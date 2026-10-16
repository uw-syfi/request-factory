//! The shared streaming engine. Protocol-blind: it drives one request through
//! whichever [`Backend`] was selected and folds the stream into a normalized
//! outcome.

use anyhow::Result;
use bytes::BytesMut;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::time::timeout;

use base64::Engine as _;
use std::path::PathBuf;

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
    /// Where each request's routed experts are written. `Some` turns the run
    /// non-streaming: the routing is one array for the whole generation, so it
    /// can only ride on a completed body.
    routed_experts_dir: Option<PathBuf>,
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
        let routed_experts_dir = args.routed_experts_dir.as_ref().map(PathBuf::from);
        if let Some(directory) = &routed_experts_dir {
            // Once, here, rather than per request: a run that cannot write its
            // captures should fail before it spends a server's worth of GPU
            // time producing them.
            std::fs::create_dir_all(directory)?;
        }
        Ok(Self {
            endpoint,
            client,
            tokenizer,
            model: args.model.clone(),
            temperature: args.temperature,
            stream_idle_timeout_secs: args.stream_idle_timeout_secs,
            record_timeline: args.timeline,
            routed_experts_dir,
            backend,
        })
    }

    pub(crate) async fn run_step(
        &self,
        request_id: String,
        prompt: Prompt<'_>,
        max_tokens: usize,
    ) -> GenerationResult {
        let start = Instant::now();

        // Submit raw token ids: no client-side decode, so the server's prefix-cache keys match
        // the exact ids we built.
        let capturing_routes = self.routed_experts_dir.is_some();
        let body = self.prepare_body(&GenRequest {
            model: &self.model,
            request_id: &request_id,
            prompt,
            max_tokens,
            temperature: self.temperature,
            stream: !capturing_routes,
            // The last prompt token, not the first generated one: its forward is
            // the step that produced token 0, so starting a row earlier is what
            // makes every later row attributable to an accepted token.
            routed_experts_prompt_start: capturing_routes
                .then(|| prompt.token_len().saturating_sub(1)),
        });
        let body_build_ms = elapsed_ms(start);

        // Every clock starts here, after the body is ready for the wire. Submit
        // and post are one stamp: nothing the client does sits between them any
        // more, and TTFT (`send_instant`) has the same origin.
        let sent_at = unix_seconds_now();
        let submit_timestamp = sent_at;
        let post_timestamp = Some(sent_at);
        // Monotonic anchor at the send instant: TTFT is measured from here.
        let send_instant = Instant::now();

        let mut fold = StreamAccumulator::new(max_tokens, self.record_timeline);
        if let Some(directory) = &self.routed_experts_dir {
            self.fold_body(&request_id, body, send_instant, &mut fold, directory)
                .await;
        } else {
            self.fold_response(&request_id, body, send_instant, &mut fold)
                .await;
        }

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
                body_build_ms: Some(body_build_ms),
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

    /// Build this request's body and serialize it to the bytes that go on the wire.
    ///
    /// Done before any timestamp is taken, and just in time rather than ahead of
    /// the request's due time: a 512k-token prompt is a ~3.5 MB body plus a JSON
    /// tree several times that size, and a trace full of them prebuilt would hold
    /// all of it at once. Encoding a body that size takes milliseconds of client
    /// work, which must not land inside TTFT. The JSON tree is dropped here too,
    /// before the stamp, for the same reason.
    fn prepare_body(&self, request: &GenRequest) -> Vec<u8> {
        encode_body(&self.backend.build_payload(request))
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
        body: Vec<u8>,
        send_instant: Instant,
        fold: &mut StreamAccumulator,
    ) {
        let response = self.post_body(request_id, body).send().await;

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

    /// The POST for one prepared body: the same request `RequestBuilder::json`
    /// would make, minus the encoding it would do after the send stamp.
    fn post_body(&self, request_id: &str, body: Vec<u8>) -> reqwest::RequestBuilder {
        self.client
            .post(&self.endpoint)
            .header("x-request-id", request_id)
            .header(CONTENT_TYPE, "application/json")
            .body(body)
    }

    /// Drive one request as a single non-streaming response and persist the
    /// routed experts it carries.
    ///
    /// The routing is one array covering the whole generation, so no chunk
    /// could carry it and there is nothing to fold incrementally. The response
    /// still goes through the same `parse_event`, which is already specified
    /// over "a stream chunk or a full body", so the outcome this produces is
    /// the same shape the streaming path produces -- minus the per-event times,
    /// which do not exist here. That is the price of the capture, and the
    /// reason it is a pass of its own.
    async fn fold_body(
        &self,
        request_id: &str,
        body: Vec<u8>,
        send_instant: Instant,
        fold: &mut StreamAccumulator,
        directory: &std::path::Path,
    ) {
        let response = self.post_body(request_id, body).send().await;
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => return fold.fail(format!("HTTP {}", response.status())),
            Err(err) => return fold.fail(format!("request error: {err}")),
        };
        let body = match response.bytes().await {
            Ok(body) => body,
            Err(err) => return fold.fail(format!("response error: {err}")),
        };
        let value: Value = match serde_json::from_slice(&body) {
            Ok(value) => value,
            Err(err) => return fold.fail(format!("response is not JSON: {err}")),
        };
        // A single body is the whole stream, so there is no continuation to
        // break out of; the accumulator's stop signal has nothing left to stop.
        let _ = fold.absorb(self.backend.parse_event(&value), elapsed_ms(send_instant));

        let Some(encoded) = self.backend.routed_experts(&value) else {
            // Not a warning: a capture that silently produced no routes would
            // be discovered only as an empty corpus, a whole GPU run later.
            return fold.fail(
                "response carries no routed_experts; start the server with \
                 --enable-return-routed-experts"
                    .to_string(),
            );
        };
        // Written through verbatim. The payload is already a `.npy` file, so
        // the packer reads the server's own array rather than one this process
        // re-encoded and could have transposed.
        match base64::engine::general_purpose::STANDARD.decode(encoded) {
            Ok(routes) => {
                if let Err(err) =
                    std::fs::write(directory.join(format!("{request_id}.npy")), routes)
                {
                    fold.fail(format!("writing routed experts: {err}"));
                }
            }
            Err(err) => fold.fail(format!("routed_experts is not base64: {err}")),
        }
    }
}

/// Serialize one request body exactly as `RequestBuilder::json` would.
fn encode_body(payload: &Value) -> Vec<u8> {
    // A `Value` always serializes: its map keys are strings by construction,
    // which is the only way `to_vec` on it can fail.
    serde_json::to_vec(payload).expect("a serde_json::Value always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokenizers::models::wordlevel::WordLevel;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn client(base_url: &str) -> GenerationClient {
        let args = Args::try_parse_from([
            "runner",
            "--trace",
            "trace.csv",
            "--text-file",
            "corpus.txt",
            "--tokenizer",
            "tokenizer",
            "--model",
            "model",
            "--backend",
            "vllm-tokens",
            "--base-url",
            base_url,
        ])
        .unwrap();
        GenerationClient::new(&args, Arc::new(Tokenizer::new(WordLevel::default()))).unwrap()
    }

    fn request<'a>(prompt_ids: &'a [u32]) -> GenRequest<'a> {
        GenRequest {
            model: "model",
            request_id: "r0",
            prompt: Prompt::Tokens(prompt_ids),
            max_tokens: 1,
            temperature: 0.0,
            stream: true,
            routed_experts_prompt_start: None,
        }
    }

    /// The prepared body must be the request `RequestBuilder::json` would have
    /// sent: same bytes, same content type. Only where the encoding happens moved.
    #[test]
    fn prepared_body_is_the_request_json_would_send() {
        let client = client("http://127.0.0.1:9");
        let prompt_ids: Vec<u32> = (0..1000).collect();
        let payload = client.backend.build_payload(&request(&prompt_ids));

        let via_json = client
            .client
            .post(&client.endpoint)
            .json(&payload)
            .build()
            .unwrap();
        let prepared = client
            .post_body("r0", client.prepare_body(&request(&prompt_ids)))
            .build()
            .unwrap();

        assert_eq!(
            prepared.body().and_then(reqwest::Body::as_bytes),
            via_json.body().and_then(reqwest::Body::as_bytes),
        );
        assert_eq!(
            prepared.headers().get(CONTENT_TYPE),
            via_json.headers().get(CONTENT_TYPE),
        );
        assert_eq!(prepared.headers()["x-request-id"], "r0");
    }

    /// Submit and post are one stamp taken after the body is ready, so the time
    /// spent building it is visible as `body_build_ms` and nowhere in TTFT.
    #[tokio::test]
    async fn stamps_are_taken_after_the_body_is_serialized() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let header_end = loop {
                let mut chunk = [0u8; 65536];
                let read = stream.read(&mut chunk).await.unwrap();
                request.extend_from_slice(&chunk[..read]);
                if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break at + 4;
                }
            };
            let head = String::from_utf8_lossy(&request[..header_end]).to_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            while request.len() < header_end + length {
                let mut chunk = [0u8; 65536];
                let read = stream.read(&mut chunk).await.unwrap();
                request.extend_from_slice(&chunk[..read]);
            }
            let received_at = unix_seconds_now();
            let events = "data: {\"choices\":[{\"index\":0,\"token_ids\":[7]}]}\n\n\
                          data: [DONE]\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{events}",
                events.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            (request[header_end..].to_vec(), received_at)
        });

        let client = client(&url);
        // Large enough that encoding takes a measurable time.
        let prompt_ids: Vec<u32> = (0..200_000).map(|i| i * 7 % 150_000).collect();
        let entered = unix_seconds_now();
        let result = client
            .run_step("r0".to_string(), Prompt::Tokens(&prompt_ids), 1)
            .await;
        let (body, received_at) = server.await.unwrap();
        let outcome = result.outcome;

        assert_eq!(outcome.status, "SUCCESS", "{:?}", outcome.error);
        assert_eq!(body, client.prepare_body(&request(&prompt_ids)));
        let build_ms = outcome.body_build_ms.unwrap();
        assert!(build_ms > 0.0);
        assert_eq!(outcome.post_timestamp, Some(outcome.submit_timestamp));
        // The stamp follows the build: entry plus the build time cannot be later
        // than it (allowing for `elapsed_ms`'s 0.01 ms rounding).
        assert!(outcome.submit_timestamp * 1e3 >= entered * 1e3 + build_ms - 0.01);
        assert!(outcome.submit_timestamp <= received_at);
        assert!(outcome.total_duration_ms >= build_ms + outcome.first_token_id_ms.unwrap() - 0.02);
    }
}
