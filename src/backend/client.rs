//! The shared streaming engine. Protocol-blind: it drives one request through
//! whichever [`Backend`] was selected and folds the stream into a normalized
//! outcome.

use anyhow::Result;
use bytes::BytesMut;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::time::timeout;

use crate::cli::Args;
use crate::record::{GenerationOutcome, ServerUsageLog};
use crate::util::{elapsed_ms, ratio, unix_seconds_now};

use super::integrity::{classify_prompt_echo, PromptEcho};
use super::stream::StreamAccumulator;
use super::wire::build_backend;
use super::{Backend, GenRequest, GenerationResult, Prompt};

/// Shared streaming engine. It knows only normalized text-generation inputs and
/// outcomes; frontend/source identity remains in the executor layer.
pub(crate) struct GenerationClient {
    // `pub(super)` rather than private: the preflight gate is a sibling module
    // because it is a different concern, not because it is a different object.
    pub(super) endpoint: String,
    pub(super) client: reqwest::Client,
    tokenizer: Arc<Tokenizer>,
    pub(super) model: String,
    temperature: f64,
    stream_idle_timeout_secs: u64,
    record_timeline: bool,
    pub(super) backend: Box<dyn Backend>,
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
            record_timeline: args.timeline,
            backend,
        })
    }

    pub(crate) async fn run_step(
        &self,
        request_id: String,
        prompt: Prompt<'_>,
        max_tokens: usize,
    ) -> GenerationResult {
        let submit_timestamp = unix_seconds_now();
        let start = Instant::now();

        // Submit raw token ids: no client-side decode, so even million-token prompts cost nothing
        // here and the server's prefix-cache keys match the exact ids we built.
        let payload = self.backend.build_payload(&GenRequest {
            model: &self.model,
            request_id: &request_id,
            prompt,
            max_tokens,
            temperature: self.temperature,
            stream: true,
        });

        let post_timestamp = Some(unix_seconds_now());
        // Monotonic anchor at the send instant: TTFT is measured from here.
        let send_instant = Instant::now();

        let mut fold = StreamAccumulator::new(max_tokens, self.record_timeline);
        self.fold_response(&request_id, &payload, send_instant, &mut fold)
            .await;

        // Stop the wire-response clock before output re-tokenization and log shaping.
        let response_complete_ms = post_timestamp.map(|_| elapsed_ms(send_instant));
        let token_delivery_tpot_ms = fold.token_delivery_tpot_ms();
        let terminal_tail_ms = match (fold.last_token_id_ms, response_complete_ms) {
            (Some(last), Some(complete)) if complete >= last => Some(complete - last),
            _ => None,
        };

        let StreamAccumulator {
            first_text_ms,
            first_token_id_ms,
            last_token_id_ms,
            first_token_event_tokens,
            token_event_count,
            usage_event_count,
            chunk_count,
            output_text,
            mut output_token_ids,
            usage: mut server,
            timeline,
            finish_reason,
            failure,
            ..
        } = fold;
        // The fold reports what went wrong, not what it means for the round; the
        // integrity repair below can still fail an otherwise clean stream.
        let mut error = failure;
        let mut status = if error.is_some() { "FAILED" } else { "SUCCESS" }.to_string();

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
        let Prompt::Tokens(prompt_ids) = prompt;
        if status == "SUCCESS" {
            if let Some(completion_tokens) = server.completion_tokens {
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
        let output_len_actual = server.completion_tokens.unwrap_or({
            if output_token_ids.is_empty() {
                output_len_text_tokens
            } else {
                output_token_ids.len()
            }
        });
        // Prefer the server's exact generated token ids (return_token_ids) for carry-forward, but
        // trust them only when their count matches the server's completion_tokens. Otherwise (an
        // older server that ignored the flag, or a shape mismatch) fall back to the re-encoded ids.
        let output_ids_exact = server.completion_tokens.map_or_else(
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
        if server.cached_prompt_tokens.is_none() && server.prompt_tokens.is_some() {
            server.cached_prompt_tokens = Some(0);
        }
        let server_uncached_prompt_tokens =
            match (server.prompt_tokens, server.cached_prompt_tokens) {
                (Some(prompt), Some(cached)) => Some(prompt.saturating_sub(cached)),
                _ => None,
            };
        let server_prefix_hit_rate = match (server.cached_prompt_tokens, server.prompt_tokens) {
            (Some(cached), Some(prompt)) => ratio(cached, prompt),
            _ => None,
        };

        let server_usage = server.seen.then_some(ServerUsageLog {
            prompt_tokens: server.prompt_tokens,
            completion_tokens: server.completion_tokens,
            total_tokens: server.total_tokens,
            cached_prompt_tokens: server.cached_prompt_tokens,
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
                first_token_ms: first_text_ms,
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
            timeline,
        }
    }

    /// Send one request and drive its response into `fold`.
    ///
    /// Owns the transport — connection, SSE framing, idle timeout — and nothing
    /// else: every observation it makes goes through [`StreamAccumulator`], and
    /// every failure it can see is reported the same way, so the caller reads
    /// one place to learn how the round went.
    async fn fold_response(
        &self,
        request_id: &str,
        payload: &Value,
        send_instant: Instant,
        fold: &mut StreamAccumulator,
    ) {
        let response = self
            .client
            .post(&self.endpoint)
            .header("x-request-id", request_id)
            .json(payload)
            .send()
            .await;

        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => return fold.fail(format!("HTTP {}", response.status())),
            Err(err) => return fold.fail(format!("request error: {err}")),
        };

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
                        // A line that is not valid JSON is skipped rather than
                        // failed: some servers interleave keep-alive comments.
                        if let Ok(value) = serde_json::from_str::<Value>(data) {
                            let event = self.backend.parse_event(&value);
                            if fold.absorb(event, elapsed_ms(send_instant)).is_break() {
                                done = true;
                                break;
                            }
                        }
                    }
                }
                Ok(Some(Err(err))) => return fold.fail(format!("stream error: {err}")),
                Ok(None) => return,
                Err(_) => {
                    return fold.fail(format!(
                        "stream idle timeout after {}s",
                        self.stream_idle_timeout_secs
                    ))
                }
            }
        }
    }
}
