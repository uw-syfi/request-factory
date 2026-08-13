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

use super::integrity::{classify_prompt_echo, restates_accumulated_output, PromptEcho};
use super::wire::build_backend;
use super::{Backend, GenRequest, GenerationResult};

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
}
