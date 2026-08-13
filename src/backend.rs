use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::time::timeout;

use crate::cli::{Args, BackendKind};
use crate::record::{GenerationOutcome, ServerUsageLog};
use crate::util::{elapsed_ms, ratio, unix_seconds_now};

/// Normalized, backend-agnostic description of one generation request.
pub(crate) struct GenRequest<'a> {
    pub(crate) model: &'a str,
    /// The id this replay knows the request by. Sent in the body as well as in
    /// the `x-request-id` header because the two engines read different ones:
    /// vLLM adopts the header, SGLang takes a body-level `rid`.
    pub(crate) request_id: &'a str,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f64,
    pub(crate) stream: bool,
}

/// Server-reported token accounting, normalized across wire formats.
pub(crate) struct Usage {
    pub(crate) prompt_tokens: Option<usize>,
    pub(crate) completion_tokens: Option<usize>,
    pub(crate) total_tokens: Option<usize>,
    pub(crate) cached_prompt_tokens: Option<usize>,
}

/// Normalized view of one streamed response object (or a full non-streaming body).
pub(crate) struct StreamEvent {
    pub(crate) text_delta: Option<String>,
    /// Exact generated token ids for this chunk, when the server echoes them
    /// (vLLM `return_token_ids`). Lets us carry the real output forward without re-tokenizing.
    pub(crate) token_ids: Option<Vec<u32>>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: Option<Usage>,
}

/// Per-backend wire-protocol adapter. Pure and synchronous: it only shapes JSON, so the
/// shared async streaming engine in `GenerationClient` stays backend-agnostic and `dyn Backend`
/// remains object-safe (no `async-trait`).
pub(crate) trait Backend: Send + Sync {
    /// Path appended to `--base-url` to form the request endpoint.
    fn endpoint_suffix(&self) -> &str;
    /// Shape one generation request into this backend's request body.
    fn build_payload(&self, req: &GenRequest) -> Value;
    /// Normalize one response JSON object (a stream chunk or a full body).
    fn parse_event(&self, value: &Value) -> StreamEvent;
}

/// Build the backend adapter selected on the command line.
pub(crate) fn build_backend(kind: BackendKind) -> Box<dyn Backend> {
    match kind {
        BackendKind::Openai => Box::new(OpenAiCompletionsBackend),
        BackendKind::VllmTokens => Box::new(VllmTokensBackend),
        BackendKind::SglangTokens => Box::new(SglangTokensBackend),
    }
}

/// OpenAI-compatible `/completions` protocol. Works against vLLM and SGLang's OpenAI endpoint.
pub(crate) struct OpenAiCompletionsBackend;

impl Backend for OpenAiCompletionsBackend {
    fn endpoint_suffix(&self) -> &str {
        "/completions"
    }

    fn build_payload(&self, req: &GenRequest) -> Value {
        let mut payload = serde_json::json!({
            "model": req.model,
            // SGLang's OpenAI layer forwards this straight to GenerateReqInput,
            // so its records carry the id this replay logged. vLLM has no such
            // field; its base model allows unknown keys and ignores this one,
            // taking the id from the x-request-id header instead.
            "rid": req.request_id,
            // Submit raw token ids (OpenAI `prompt` accepts an int array): no client-side decode,
            // and the server uses the exact ids so prefix-cache keys match what we constructed.
            "prompt": req.prompt_ids,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
            // Always run decode to the trace's target length; synthetic prompts otherwise emit
            // EOS almost immediately and collapse the decode workload.
            "ignore_eos": true,
            // Echo the generated token ids (recent vLLM) so we carry the real output forward
            // exactly. Older servers ignore this; we fall back to re-encoding the output text.
            "return_token_ids": true,
        });
        if req.stream {
            // Ask for the trailing usage chunk: server token counts and prefix-cache details
            // are the whole point of this runner.
            payload["stream_options"] = serde_json::json!({"include_usage": true});
        }
        payload
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let text_delta = choice
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let token_ids = choice
            .and_then(|c| c.get("token_ids"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
                    .collect::<Vec<u32>>()
            });
        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let usage = value
            .get("usage")
            .filter(|usage| !usage.is_null())
            .map(|usage| Usage {
                prompt_tokens: usage_usize(usage, "prompt_tokens"),
                completion_tokens: usage_usize(usage, "completion_tokens"),
                total_tokens: usage_usize(usage, "total_tokens"),
                cached_prompt_tokens: usage_cached_prompt_tokens(usage),
            });
        StreamEvent {
            text_delta,
            token_ids,
            finish_reason,
            usage,
        }
    }
}

/// vLLM native token-in/token-out protocol. The server must be launched with
/// `--tokens-only`, which forces `SamplingParams.detokenize = false` for this endpoint.
pub(crate) struct VllmTokensBackend;

impl Backend for VllmTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/inference/v1/generate"
    }

    fn build_payload(&self, req: &GenRequest) -> Value {
        let mut payload = serde_json::json!({
            "model": req.model,
            // SGLang's OpenAI layer forwards this straight to GenerateReqInput,
            // so its records carry the id this replay logged. vLLM has no such
            // field; its base model allows unknown keys and ignores this one,
            // taking the id from the x-request-id header instead.
            "rid": req.request_id,
            "token_ids": req.prompt_ids,
            "sampling_params": {
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "ignore_eos": true,
            },
            "stream": req.stream,
        });
        if req.stream {
            payload["stream_options"] = serde_json::json!({"include_usage": true});
        }
        payload
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let token_ids = choice
            .and_then(|choice| choice.get("token_ids"))
            .and_then(Value::as_array)
            .map(|token_ids| {
                token_ids
                    .iter()
                    .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect::<Vec<u32>>()
            });
        let finish_reason = choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let usage = value
            .get("usage")
            .filter(|usage| !usage.is_null())
            .map(|usage| Usage {
                prompt_tokens: usage_usize(usage, "prompt_tokens"),
                completion_tokens: usage_usize(usage, "completion_tokens"),
                total_tokens: usage_usize(usage, "total_tokens"),
                cached_prompt_tokens: usage_cached_prompt_tokens(usage),
            });
        StreamEvent {
            text_delta: None,
            token_ids,
            finish_reason,
            usage,
        }
    }
}

/// SGLang native token-in/token-out `/generate` protocol.
///
/// The server must be launched with two flags. `--skip-tokenizer-init` is the
/// counterpart of vLLM's `--tokens-only`: it accepts `input_ids` and returns
/// `output_ids` without ever detokenizing. `--stream-output` (renamed
/// `--incremental-streaming-output` in newer builds) makes streamed chunks
/// disjoint deltas; SGLang's default resends the whole output every chunk,
/// which is O(n^2) on the wire and inflates late-token latency.
/// [`restates_accumulated_output`] fails the round if that default is still in
/// effect rather than silently reporting the polluted timings.
///
/// Verified against SGLang 0.5.9: `output_ids` arrives as a top-level per-chunk
/// delta, `meta_info` carries running `prompt_tokens`/`completion_tokens`/
/// `cached_tokens`, `finish_reason` is an object, and no `text` field is sent.
pub(crate) struct SglangTokensBackend;

impl Backend for SglangTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/generate"
    }

    fn build_payload(&self, req: &GenRequest) -> Value {
        // No `model` field: an SGLang server hosts exactly one model. No
        // `return_logprob` either — `output_ids` is a native top-level response
        // field, so recovering ids out of per-token logprobs would only add
        // compute and serialization to the path we are timing.
        serde_json::json!({
            "rid": req.request_id,
            "input_ids": req.prompt_ids,
            "sampling_params": {
                "max_new_tokens": req.max_tokens,
                "temperature": req.temperature,
                // Always decode to the trace's target length; synthetic prompts
                // otherwise emit EOS immediately and collapse the workload.
                "ignore_eos": true,
            },
            "stream": req.stream,
        })
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let token_ids = value
            .get("output_ids")
            .and_then(Value::as_array)
            .map(|output_ids| {
                output_ids
                    .iter()
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect::<Vec<u32>>()
            });
        let meta_info = value.get("meta_info");
        StreamEvent {
            // Under --skip-tokenizer-init the server never produces text.
            text_delta: None,
            token_ids,
            finish_reason: meta_info.and_then(sglang_finish_reason),
            usage: meta_info.and_then(sglang_usage),
        }
    }
}

/// SGLang reports `finish_reason` as an object (`{"type": "length"}`) rather
/// than the bare string the OpenAI schema uses. Accept both.
fn sglang_finish_reason(meta_info: &Value) -> Option<String> {
    let reason = meta_info.get("finish_reason")?;
    if let Some(reason) = reason.as_str() {
        return Some(reason.to_string());
    }
    reason.get("type")?.as_str().map(str::to_string)
}

/// SGLang carries token accounting in `meta_info`, not in an OpenAI `usage`
/// object. Its schema is fixed, so read the exact keys instead of searching the
/// provider alias list [`usage_cached_prompt_tokens`] needs.
///
/// `meta_info` rides on every streamed chunk with running counts. The shared
/// engine keeps the newest value for each field, so the final chunk's totals win.
fn sglang_usage(meta_info: &Value) -> Option<Usage> {
    let prompt_tokens = usage_usize(meta_info, "prompt_tokens");
    let completion_tokens = usage_usize(meta_info, "completion_tokens")
        .or_else(|| usage_usize(meta_info, "output_tokens"));
    let cached_prompt_tokens = usage_usize(meta_info, "cached_tokens");
    if prompt_tokens.is_none() && completion_tokens.is_none() && cached_prompt_tokens.is_none() {
        return None;
    }
    Some(Usage {
        prompt_tokens,
        completion_tokens,
        // SGLang does not report a combined total; derive it only when both
        // halves are present rather than reporting a partial sum.
        total_tokens: match (prompt_tokens, completion_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt.saturating_add(completion)),
            _ => None,
        },
        cached_prompt_tokens,
    })
}

/// Whether a streamed chunk restates the whole output so far instead of
/// carrying only new tokens.
///
/// Every supported backend is expected to stream disjoint deltas. Folding a
/// cumulative chunk as if it were a delta would multiply the output and wreck
/// the TPOT denominator, so the shared engine treats this as a hard failure.
/// The first chunk is exempt because an empty accumulator prefixes anything.
fn restates_accumulated_output(accumulated: &[u32], incoming: &[u32]) -> bool {
    !accumulated.is_empty()
        && incoming.len() > accumulated.len()
        && incoming.starts_with(accumulated)
}

/// Verdict on generated token ids that outnumber the server's completion count.
#[derive(Debug, PartialEq, Eq)]
enum PromptEcho {
    /// The counts agree; nothing was echoed.
    None,
    /// The leading `n` ids provably repeat the prompt's tail and can be dropped.
    Leading(usize),
    /// There are more ids than completion tokens, but the excess does not match
    /// the prompt tail, so what it represents is unknown.
    Unexplained,
}

/// Classify a leading prompt echo in the collected generated token ids.
///
/// SGLang has been reported to prepend a suffix of `input_ids` to `output_ids`
/// (sgl-project/sglang#10896). That was filed against the offline Engine API
/// rather than this streaming HTTP path, so the guard is cheap insurance:
/// it trims only what it can prove came from the prompt, and refuses to guess
/// otherwise. Carrying an echoed prefix forward would corrupt the next round's
/// context and every prefix-cache number derived from it.
fn classify_prompt_echo(
    output_ids: &[u32],
    prompt_ids: &[u32],
    completion_tokens: usize,
) -> PromptEcho {
    let Some(echoed) = output_ids.len().checked_sub(completion_tokens) else {
        return PromptEcho::None;
    };
    if echoed == 0 {
        return PromptEcho::None;
    }
    let matches_prompt_tail = prompt_ids
        .len()
        .checked_sub(echoed)
        .is_some_and(|start| prompt_ids[start..] == output_ids[..echoed]);
    if matches_prompt_tail {
        PromptEcho::Leading(echoed)
    } else {
        PromptEcho::Unexplained
    }
}

/// Backend result shared by every text-generation source. Source identity stays
/// with the executor; session executors alone carry `output_ids` to the next round.
pub(crate) struct GenerationResult {
    pub(crate) outcome: GenerationOutcome,
    pub(crate) output_ids: Vec<u32>,
}

pub(crate) fn context_limit_skip_result(
    request_id: String,
    prompt_len: usize,
    output_len_target: usize,
    max_model_len: Option<usize>,
) -> GenerationResult {
    let limit = max_model_len
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    GenerationResult {
        outcome: GenerationOutcome {
            request_id,
            output_len_actual: 0,
            output_len_text_tokens: 0,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: None,
            submit_timestamp: unix_seconds_now(),
            post_timestamp: None,
            complete_timestamp: unix_seconds_now(),
            first_token_ms: None,
            first_token_id_ms: None,
            last_token_id_ms: None,
            first_token_event_tokens: 0,
            token_event_count: 0,
            usage_event_count: 0,
            token_delivery_tpot_ms: None,
            response_complete_ms: None,
            terminal_tail_ms: None,
            total_duration_ms: 0.0,
            chunk_count: 0,
            status: "SKIPPED_CONTEXT_OVERFLOW".to_string(),
            output_preview: String::new(),
            error: Some(format!(
                "requested context {} (prompt_len {} + output_len_target {}) reaches max_model_len {}; one token of headroom is required",
                prompt_len.saturating_add(output_len_target),
                prompt_len,
                output_len_target,
                limit,
            )),
        },
        output_ids: Vec::new(),
    }
}

/// Shared streaming engine. It knows only normalized text-generation inputs and
/// outcomes; frontend/source identity remains in the executor layer.
pub(crate) struct GenerationClient {
    endpoint: String,
    client: reqwest::Client,
    tokenizer: Arc<Tokenizer>,
    model: String,
    temperature: f64,
    stream_idle_timeout_secs: u64,
    backend: Box<dyn Backend>,
}

impl GenerationClient {
    pub(crate) fn new(args: &Args, tokenizer: Arc<Tokenizer>) -> Result<Self> {
        let backend = build_backend(args.backend);
        let endpoint = format!(
            "{}{}",
            args.base_url.trim_end_matches('/'),
            backend.endpoint_suffix()
        );
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(20_000)
            .tcp_nodelay(true)
            .timeout(Duration::from_secs(3600))
            .build()?;
        Ok(Self {
            endpoint,
            client,
            tokenizer,
            model: args.model.clone(),
            temperature: args.temperature,
            stream_idle_timeout_secs: args.stream_idle_timeout_secs,
            backend,
        })
    }

    pub(crate) async fn run_step(
        &self,
        request_id: String,
        prompt_ids: &[u32],
        max_tokens: usize,
    ) -> GenerationResult {
        let submit_timestamp = unix_seconds_now();
        let start = Instant::now();

        // Submit raw token ids: no client-side decode, so even million-token prompts cost nothing
        // here and the server's prefix-cache keys match the exact ids we built.
        let payload = self.backend.build_payload(&GenRequest {
            model: &self.model,
            request_id: &request_id,
            prompt_ids,
            max_tokens,
            temperature: self.temperature,
            stream: true,
        });

        let post_timestamp = Some(unix_seconds_now());
        // Monotonic anchor at the send instant: TTFT is measured from here.
        let send_instant = Instant::now();
        let mut first_token_ms = None;
        let mut first_token_id_ms = None;
        let mut last_token_id_ms = None;
        let mut first_token_event_tokens = 0usize;
        let mut token_event_count = 0usize;
        let mut usage_event_count = 0usize;
        let mut chunk_count = 0usize;
        let mut output_text = String::new();
        let mut output_token_ids: Vec<u32> = Vec::new();
        let mut status = "SUCCESS".to_string();
        let mut error = None;
        let mut finish_reason = None;
        let mut server_prompt_tokens = None;
        let mut server_completion_tokens = None;
        let mut server_total_tokens = None;
        let mut server_cached_prompt_tokens = None;
        let mut saw_server_usage = false;

        let response = self
            .client
            .post(&self.endpoint)
            .header("x-request-id", &request_id)
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(response) if response.status().is_success() => {
                let mut stream = response.bytes_stream();
                let mut buffer = BytesMut::with_capacity(8192);
                let mut done = false;

                while !done {
                    match timeout(
                        Duration::from_secs(self.stream_idle_timeout_secs),
                        stream.next(),
                    )
                    .await
                    {
                        Ok(Some(Ok(chunk))) => {
                            buffer.extend_from_slice(&chunk);
                            while let Some(idx) = buffer.iter().position(|&b| b == b'\n') {
                                let line_bytes = buffer.split_to(idx + 1);
                                let line = String::from_utf8_lossy(&line_bytes);
                                let line = line.trim();
                                if !line.starts_with("data: ") {
                                    continue;
                                }
                                let data = line.trim_start_matches("data: ").trim();
                                if data == "[DONE]" {
                                    done = true;
                                    break;
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    let event = self.backend.parse_event(&value);
                                    let event_elapsed_ms = elapsed_ms(send_instant);
                                    let token_ids = event.token_ids.unwrap_or_default();
                                    if restates_accumulated_output(&output_token_ids, &token_ids) {
                                        status = "FAILED".to_string();
                                        error = Some(format!(
                                            "server streamed cumulative output: a chunk repeated all {} \
                                             tokens delivered so far. Launch SGLang with \
                                             --stream-output (renamed --incremental-streaming-output \
                                             in newer builds) so chunks are disjoint deltas.",
                                            output_token_ids.len(),
                                        ));
                                        done = true;
                                        break;
                                    }
                                    if !token_ids.is_empty() {
                                        if first_token_id_ms.is_none() {
                                            first_token_id_ms = Some(event_elapsed_ms);
                                            first_token_event_tokens = token_ids.len();
                                        }
                                        last_token_id_ms = Some(event_elapsed_ms);
                                        token_event_count += 1;
                                        output_token_ids.extend(token_ids);
                                    }
                                    if let Some(delta) = event.text_delta {
                                        if !delta.is_empty() {
                                            if first_token_ms.is_none() {
                                                first_token_ms = Some(event_elapsed_ms);
                                            }
                                            output_text.push_str(&delta);
                                        }
                                    }
                                    if let Some(reason) = event.finish_reason {
                                        finish_reason = Some(reason);
                                    }
                                    if let Some(usage) = event.usage {
                                        usage_event_count += 1;
                                        saw_server_usage = true;
                                        server_prompt_tokens =
                                            usage.prompt_tokens.or(server_prompt_tokens);
                                        server_completion_tokens =
                                            usage.completion_tokens.or(server_completion_tokens);
                                        server_total_tokens =
                                            usage.total_tokens.or(server_total_tokens);
                                        server_cached_prompt_tokens = usage
                                            .cached_prompt_tokens
                                            .or(server_cached_prompt_tokens);
                                    }
                                    chunk_count += 1;
                                }
                            }
                        }
                        Ok(Some(Err(err))) => {
                            status = "FAILED".to_string();
                            error = Some(format!("stream error: {err}"));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            status = "FAILED".to_string();
                            error = Some(format!(
                                "stream idle timeout after {}s",
                                self.stream_idle_timeout_secs
                            ));
                            break;
                        }
                    }
                }
            }
            Ok(response) => {
                status = "FAILED".to_string();
                error = Some(format!("HTTP {}", response.status()));
            }
            Err(err) => {
                status = "FAILED".to_string();
                error = Some(format!("request error: {err}"));
            }
        }

        // Stop the wire-response clock before output re-tokenization and log shaping.
        let response_complete_ms = post_timestamp.map(|_| elapsed_ms(send_instant));
        let token_delivery_tpot_ms = match (
            first_token_id_ms,
            last_token_id_ms,
            output_token_ids.len().checked_sub(first_token_event_tokens),
        ) {
            (Some(first), Some(last), Some(delivered_after_first_event))
                if delivered_after_first_event > 0 && last >= first =>
            {
                Some((last - first) / delivered_after_first_event as f64)
            }
            _ => None,
        };
        let terminal_tail_ms = match (last_token_id_ms, response_complete_ms) {
            (Some(last), Some(complete)) if complete >= last => Some(complete - last),
            _ => None,
        };

        // Re-encode the output text for a diagnostic token count and as a carry-forward fallback.
        let reencoded_output_ids: Vec<u32> = self
            .tokenizer
            .encode(output_text.clone(), false)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_default();
        let output_len_text_tokens = reencoded_output_ids.len();
        // Drop a leading echo of the prompt when the server streamed more ids
        // than it counted as completion tokens. Proven echoes are trimmed, an
        // unexplained excess fails the round: carrying either one forward would
        // corrupt the next prompt and every cache number derived from it.
        let mut echoed_prompt_tokens = 0usize;
        if status == "SUCCESS" {
            if let Some(completion_tokens) = server_completion_tokens {
                match classify_prompt_echo(&output_token_ids, prompt_ids, completion_tokens) {
                    PromptEcho::None => {}
                    PromptEcho::Leading(echoed) => {
                        output_token_ids.drain(..echoed);
                        echoed_prompt_tokens = echoed;
                    }
                    PromptEcho::Unexplained => {
                        error = Some(format!(
                            "server streamed {} generated token ids but reported {} completion \
                             tokens, and the {} extra leading ids do not match the prompt tail",
                            output_token_ids.len(),
                            completion_tokens,
                            output_token_ids.len().saturating_sub(completion_tokens),
                        ));
                        status = "FAILED".to_string();
                    }
                }
            }
        }
        let output_len_actual = server_completion_tokens.unwrap_or({
            if output_token_ids.is_empty() {
                output_len_text_tokens
            } else {
                output_token_ids.len()
            }
        });
        // Prefer the server's exact generated token ids (return_token_ids) for carry-forward, but
        // trust them only when their count matches the server's completion_tokens. Otherwise (an
        // older server that ignored the flag, or a shape mismatch) fall back to the re-encoded ids.
        let output_ids_exact = server_completion_tokens.map_or_else(
            || !output_token_ids.is_empty(),
            |count| output_token_ids.len() == count,
        );
        let output_ids: Vec<u32> = if output_ids_exact {
            output_token_ids
        } else {
            reencoded_output_ids
        };
        // Servers omit cached-token details when nothing was cached, so usage-present but
        // cache-detail-absent means zero cached tokens. Requires the server to report
        // prompt-token details (vLLM: --enable-prompt-tokens-details) to be meaningful.
        if server_cached_prompt_tokens.is_none() && server_prompt_tokens.is_some() {
            server_cached_prompt_tokens = Some(0);
        }
        let server_uncached_prompt_tokens =
            match (server_prompt_tokens, server_cached_prompt_tokens) {
                (Some(prompt), Some(cached)) => Some(prompt.saturating_sub(cached)),
                _ => None,
            };
        let server_prefix_hit_rate = match (server_cached_prompt_tokens, server_prompt_tokens) {
            (Some(cached), Some(prompt)) => ratio(cached, prompt),
            _ => None,
        };

        let server_usage = saw_server_usage.then_some(ServerUsageLog {
            prompt_tokens: server_prompt_tokens,
            completion_tokens: server_completion_tokens,
            total_tokens: server_total_tokens,
            cached_prompt_tokens: server_cached_prompt_tokens,
            uncached_prompt_tokens: server_uncached_prompt_tokens,
            prefix_hit_rate: server_prefix_hit_rate,
        });

        GenerationResult {
            outcome: GenerationOutcome {
                request_id,
                output_len_actual,
                output_len_text_tokens,
                echoed_prompt_tokens,
                server_usage,
                finish_reason,
                submit_timestamp,
                post_timestamp,
                complete_timestamp: unix_seconds_now(),
                first_token_ms,
                first_token_id_ms,
                last_token_id_ms,
                first_token_event_tokens,
                token_event_count,
                usage_event_count,
                token_delivery_tpot_ms,
                response_complete_ms,
                terminal_tail_ms,
                total_duration_ms: elapsed_ms(start),
                chunk_count,
                status,
                output_preview: output_text.chars().take(100).collect(),
                error,
            },
            output_ids,
        }
    }

    /// Abort early unless the server actually reports prefix-cache hits.
    ///
    /// Servers omit cached-token details when nothing is cached, so a single response cannot tell
    /// "feature disabled" apart from "cache cold". We force a guaranteed hit by sending the same
    /// probe prompt twice and require the second response to report cached tokens. This also
    /// confirms prefix caching itself is enabled server-side.
    pub(crate) async fn preflight_cache_check(&self, probe_ids: &[u32]) -> Result<()> {
        // First request warms the prefix cache; the identical second request must hit it.
        self.post_probe(probe_ids)
            .await
            .context("preflight warm-up request failed")?;
        let usage = self
            .post_probe(probe_ids)
            .await
            .context("preflight cache-hit request failed")?;

        let usage = usage.ok_or_else(|| {
            anyhow!("preflight: server response carried no usage block; cannot verify prefix-cache reporting")
        })?;
        match usage.cached_prompt_tokens {
            Some(cached) if cached > 0 => Ok(()),
            other => Err(anyhow!(
                "preflight: server reported no prefix-cache hit (prompt_tokens={:?}, cached_tokens={:?}). \
                 Launch the server with prompt-token details and prefix caching enabled \
                 (vLLM: --enable-prompt-tokens-details / ENABLE_PROMPT_TOKENS_DETAILS=1); see replay/README.md.",
                usage.prompt_tokens, other
            )),
        }
    }

    /// Send one streaming completion and return its final normalized usage, if present.
    ///
    /// Both supported backends expose prompt-cache details in the final SSE usage
    /// chunk. Keeping preflight on that same wire path also avoids depending on a
    /// backend's optional non-streaming response schema.
    async fn post_probe(&self, prompt_ids: &[u32]) -> Result<Option<Usage>> {
        let payload = self.backend.build_payload(&GenRequest {
            model: &self.model,
            // Not a trace request: named so it is obvious in a server log that
            // this row belongs to the prefix-cache preflight, not the workload.
            request_id: "req-frontend-prefix-cache-preflight",
            prompt_ids,
            max_tokens: 1,
            temperature: 0.0,
            stream: true,
        });
        let response = self
            .client
            .post(&self.endpoint)
            // vLLM DP ranks own independent prefix caches. Pin both preflight
            // requests to one rank so the second request tests the feature
            // instead of accidentally probing a different cache shard.
            // Servers that do not implement this vLLM routing header ignore it.
            .header("X-data-parallel-rank", "0")
            .json(&payload)
            .send()
            .await
            .map_err(|err| anyhow!("request error: {err}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HTTP {status}: {}",
                body.chars().take(200).collect::<String>()
            ));
        }
        let body = response
            .text()
            .await
            .map_err(|err| anyhow!("invalid streaming response: {err}"))?;
        Ok(final_usage_from_sse(self.backend.as_ref(), &body))
    }
}

fn final_usage_from_sse(backend: &dyn Backend, body: &str) -> Option<Usage> {
    let mut final_usage = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(usage) = backend.parse_event(&value).usage {
            final_usage = Some(usage);
        }
    }
    final_usage
}

fn usage_usize(usage: &Value, key: &str) -> Option<usize> {
    usage
        .get(key)?
        .as_u64()
        .and_then(|value| value.try_into().ok())
}

fn usage_cached_prompt_tokens(usage: &Value) -> Option<usize> {
    [
        &["prompt_tokens_details", "cached_tokens"][..],
        &["cached_tokens"][..],
        &["cached_input_tokens"][..],
        &["cache_read_input_tokens"][..],
        &["prompt_cached_tokens"][..],
        &["num_cached_tokens"][..],
    ]
    .into_iter()
    .find_map(|path| value_at_path(usage, path)?.as_u64()?.try_into().ok())
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    Some(cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_tokens_backend_uses_native_token_protocol() {
        let backend = VllmTokensBackend;
        let payload = backend.build_payload(&GenRequest {
            model: "meta-llama/Meta-Llama-3-8B",
            request_id: "req-1",
            prompt_ids: &[11, 22, 33],
            max_tokens: 4,
            temperature: 0.0,
            stream: true,
        });

        assert_eq!(backend.endpoint_suffix(), "/inference/v1/generate");
        assert_eq!(payload["token_ids"], serde_json::json!([11, 22, 33]));
        assert_eq!(payload["sampling_params"]["max_tokens"], 4);
        assert_eq!(payload["sampling_params"]["temperature"], 0.0);
        assert_eq!(payload["sampling_params"]["ignore_eos"], true);
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["stream_options"]["include_usage"], true);
        assert!(payload.get("prompt").is_none());
        assert!(payload.get("return_token_ids").is_none());
    }

    #[test]
    fn vllm_tokens_backend_normalizes_stream_and_usage_events() {
        let backend = VllmTokensBackend;
        let token_event = backend.parse_event(&serde_json::json!({
            "request_id": "generate-tokens-request-1",
            "usage": null,
            "choices": [{
                "index": 0,
                "finish_reason": "length",
                "token_ids": [101, 102]
            }]
        }));
        assert_eq!(token_event.token_ids, Some(vec![101, 102]));
        assert_eq!(token_event.finish_reason.as_deref(), Some("length"));
        assert!(token_event.text_delta.is_none());
        assert!(token_event.usage.is_none());

        let usage_event = backend.parse_event(&serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1}
            }
        }));
        let usage = usage_event.usage.expect("usage event");
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(5));
        assert_eq!(usage.cached_prompt_tokens, Some(1));
    }

    #[test]
    fn sglang_backend_sends_input_ids_without_model_or_logprobs() {
        let backend = SglangTokensBackend;
        let payload = backend.build_payload(&GenRequest {
            model: "ignored-by-sglang",
            request_id: "req-1",
            prompt_ids: &[7, 8, 9],
            max_tokens: 16,
            temperature: 0.0,
            stream: true,
        });

        assert_eq!(backend.endpoint_suffix(), "/generate");
        assert_eq!(payload["input_ids"], serde_json::json!([7, 8, 9]));
        assert_eq!(payload["sampling_params"]["max_new_tokens"], 16);
        assert_eq!(payload["sampling_params"]["temperature"], 0.0);
        assert_eq!(payload["sampling_params"]["ignore_eos"], true);
        assert_eq!(payload["stream"], true);
        // An SGLang server hosts one model, and output_ids is native: neither a
        // model field nor the logprob recovery path belongs in this payload.
        assert!(payload.get("model").is_none());
        assert!(payload.get("return_logprob").is_none());
        assert!(payload.get("prompt").is_none());
    }

    #[test]
    fn sglang_backend_normalizes_meta_info_into_usage() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({
            "output_ids": [101, 102],
            "meta_info": {
                "prompt_tokens": 512,
                "completion_tokens": 2,
                "cached_tokens": 496,
                "finish_reason": {"type": "length"}
            }
        }));

        assert_eq!(event.token_ids, Some(vec![101, 102]));
        // Under --skip-tokenizer-init there is no text to carry.
        assert!(event.text_delta.is_none());
        assert_eq!(event.finish_reason.as_deref(), Some("length"));
        let usage = event.usage.expect("meta_info must normalize into usage");
        assert_eq!(usage.prompt_tokens, Some(512));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.cached_prompt_tokens, Some(496));
        assert_eq!(usage.total_tokens, Some(514));
    }

    #[test]
    fn sglang_usage_falls_back_to_output_tokens_and_plain_finish_reason() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({
            "output_ids": [5],
            "meta_info": {"prompt_tokens": 4, "output_tokens": 1, "finish_reason": "stop"}
        }));

        let usage = event.usage.expect("usage");
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.cached_prompt_tokens, None);
        assert_eq!(event.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn sglang_chunk_without_meta_info_reports_no_usage() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({"output_ids": [1, 2]}));

        assert_eq!(event.token_ids, Some(vec![1, 2]));
        assert!(event.usage.is_none());
        assert!(event.finish_reason.is_none());
    }

    #[test]
    fn cumulative_streaming_is_detected_but_real_deltas_are_not() {
        // A cumulative chunk repeats everything delivered so far and then adds.
        assert!(restates_accumulated_output(&[1, 2, 3], &[1, 2, 3, 4]));
        // A disjoint delta does not, even when it happens to start with the
        // same id the accumulator did.
        assert!(!restates_accumulated_output(&[1, 2, 3], &[4, 5]));
        assert!(!restates_accumulated_output(&[1, 2, 3], &[1, 9]));
        // Re-sending the identical array without growing is not the cumulative
        // pattern this guard is for, and must not be misread as one.
        assert!(!restates_accumulated_output(&[1, 2, 3], &[1, 2, 3]));
        // The first chunk is exempt: an empty accumulator prefixes anything.
        assert!(!restates_accumulated_output(&[], &[1, 2, 3]));
    }

    #[test]
    fn prompt_echo_is_trimmed_only_when_it_matches_the_prompt_tail() {
        let prompt_ids = [10, 11, 12, 13];

        // Counts agree: nothing echoed.
        assert_eq!(
            classify_prompt_echo(&[90, 91], &prompt_ids, 2),
            PromptEcho::None
        );
        // Two extra leading ids that are exactly the prompt's last two tokens.
        assert_eq!(
            classify_prompt_echo(&[12, 13, 90, 91], &prompt_ids, 2),
            PromptEcho::Leading(2)
        );
        // Extra ids that are not the prompt tail: meaning unknown, never guess.
        assert_eq!(
            classify_prompt_echo(&[77, 78, 90, 91], &prompt_ids, 2),
            PromptEcho::Unexplained
        );
        // More excess than the prompt has tokens cannot be a prompt echo.
        assert_eq!(
            classify_prompt_echo(&[1, 2, 3, 4, 5, 90], &[10, 11], 1),
            PromptEcho::Unexplained
        );
        // Fewer ids than completion tokens is a different problem, and is left
        // to the existing output_ids_exact check rather than handled here.
        assert_eq!(
            classify_prompt_echo(&[90], &prompt_ids, 4),
            PromptEcho::None
        );
    }

    #[test]
    fn vllm_tokens_backend_reads_final_streaming_usage_for_preflight() {
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"token_ids\":[101]}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":512,",
            "\"completion_tokens\":1,\"total_tokens\":513,",
            "\"prompt_tokens_details\":{\"cached_tokens\":496}}}\n\n",
            "data: [DONE]\n\n",
        );

        let usage = final_usage_from_sse(&VllmTokensBackend, body).expect("final usage");
        assert_eq!(usage.prompt_tokens, Some(512));
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.cached_prompt_tokens, Some(496));
    }
}
