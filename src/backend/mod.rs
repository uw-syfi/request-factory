//! The wire boundary: normalized request/response vocabulary, the per-protocol
//! adapters that shape it, and the streaming engine that drives one request.
//!
//! The split here is deliberate. `wire/` is pure JSON shaping and knows nothing
//! about time, concurrency, or failure policy. `client` owns the async engine and
//! is protocol-blind. `integrity` holds the checks that decide whether a
//! response can be believed at all, which is a policy question rather than a
//! parsing one. Nothing above this module sees a `serde_json::Value`.

mod client;
mod dialect;
mod integrity;
mod media_client;
mod preflight;
mod stream;
mod wire;

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde_json::Value;

use crate::record::GenerationOutcome;
use crate::schema::Modality;
use crate::timeline::TimelineEvent;
use crate::util::unix_seconds_now;

pub(crate) use client::GenerationClient;
pub(crate) use dialect::{dialect_for, Dialect};
pub(crate) use media_client::MediaClient;

/// What one request sends as its input.
///
/// One variant today, and an enum anyway on purpose. The second variant is
/// interleaved text and media; when it lands, every place that has to decide how
/// to shape or measure a prompt should be a place the compiler names rather than
/// one a reader has to go find. Each site therefore destructures — an
/// irrefutable `let` today, a compile error the day the variant appears.
///
/// Borrowed rather than owned: prompts here reach a million tokens, and the
/// session executor still needs this one after the send, to carry forward as the
/// next round's context.
#[derive(Clone, Copy)]
pub(crate) enum Prompt<'a> {
    /// Token ids built by the caller and sent verbatim, so the server's
    /// prefix-cache keys match the exact ids the workload planned.
    Tokens(&'a [u32]),
    /// Interleaved content already resolved and encoded before replay starts.
    Parts(&'a [PreparedInputPart]),
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedInputPart {
    System(String),
    Text(String),
    Media {
        modality: Modality,
        data_url: String,
    },
}

/// A failed response's status *and* what the server said about it.
///
/// The status alone is rarely the answer: vLLM's 404 body carries "The model
/// `x` does not exist", which is the whole diagnosis.
pub(crate) async fn http_failure(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    match error_detail(&body) {
        Some(detail) => format!("HTTP {status}: {detail}"),
        None => format!("HTTP {status}"),
    }
}

/// The human-readable part of an error body, unwrapped from whichever envelope
/// carries it. Truncated because an error body can be an HTML page, and one
/// line per failed request is enough to act on.
fn error_detail(body: &str) -> Option<String> {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.to_string());
    let detail = detail.trim();
    if detail.is_empty() {
        return None;
    }
    Some(detail.chars().take(300).collect())
}

/// Shape role-aware, interleaved input parts the way one dialect expects them.
///
/// Returns the whole input half of the request body, not just `messages`,
/// because where media goes is itself a dialect decision: two of the three
/// encodings put it in the message content and the third hangs it off the
/// request root.
pub(crate) fn chat_inputs(
    parts: &[PreparedInputPart],
    encoding: dialect::MediaInput,
) -> Result<serde_json::Map<String, Value>> {
    use dialect::MediaInput;

    let mut messages = Vec::new();
    let mut user_content = Vec::new();
    let mut lists: BTreeMap<&'static str, Vec<Value>> = BTreeMap::new();
    let mut saw_user = false;
    for part in parts {
        match part {
            PreparedInputPart::System(text) if !saw_user => {
                messages.push(serde_json::json!({"role": "system", "content": text}));
            }
            PreparedInputPart::System(_) => bail!("system inputs must precede user inputs"),
            PreparedInputPart::Text(text) => {
                saw_user = true;
                user_content.push(serde_json::json!({"type": "text", "text": text}));
            }
            PreparedInputPart::Media { modality, data_url } => {
                saw_user = true;
                match encoding {
                    MediaInput::UrlParts => user_content.push(url_part(*modality, data_url)?),
                    MediaInput::OpenAiParts => user_content.push(openai_part(*modality, data_url)?),
                    MediaInput::TopLevelLists => {
                        lists
                            .entry(media_list_key(*modality)?)
                            .or_default()
                            .push(Value::String(data_url.clone()));
                    }
                }
            }
        }
    }
    if !saw_user {
        bail!("request has no user input")
    }
    // A media-only request under `TopLevelLists` still needs a user turn, and
    // that turn's content is a plain string there rather than an array.
    let content = if matches!(encoding, MediaInput::TopLevelLists) {
        Value::String(
            user_content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        if user_content.is_empty() {
            bail!("request has no user input")
        }
        Value::Array(user_content)
    };
    messages.push(serde_json::json!({"role": "user", "content": content}));

    let mut body = serde_json::Map::new();
    body.insert("messages".into(), Value::Array(messages));
    for (key, values) in lists {
        body.insert(key.into(), Value::Array(values));
    }
    Ok(body)
}

/// vLLM's `_url` family, which is also the only spelling video has anywhere.
fn url_part(modality: Modality, data_url: &str) -> Result<Value> {
    let key = match modality {
        Modality::Image => "image_url",
        Modality::Audio => "audio_url",
        Modality::Video => "video_url",
        Modality::Text | Modality::Tensor => {
            bail!("unsupported prepared media modality {modality:?}")
        }
    };
    Ok(serde_json::json!({"type": key, key: {"url": data_url}}))
}

/// OpenAI proper: images by URL, audio as `input_audio`, and no video at all.
fn openai_part(modality: Modality, data_url: &str) -> Result<Value> {
    match modality {
        Modality::Image => Ok(serde_json::json!({
            "type": "image_url", "image_url": {"url": data_url}
        })),
        Modality::Audio => {
            let (format, data) = split_data_url(data_url)?;
            Ok(serde_json::json!({
                "type": "input_audio", "input_audio": {"data": data, "format": format}
            }))
        }
        Modality::Video => bail!("the openai dialect has no video input content part"),
        Modality::Text | Modality::Tensor => {
            bail!("unsupported prepared media modality {modality:?}")
        }
    }
}

fn media_list_key(modality: Modality) -> Result<&'static str> {
    match modality {
        Modality::Image => Ok("images"),
        Modality::Audio => Ok("audios"),
        Modality::Video => Ok("videos"),
        Modality::Text | Modality::Tensor => {
            bail!("unsupported prepared media modality {modality:?}")
        }
    }
}

/// `data:audio/wav;base64,AAAA` -> `("wav", "AAAA")`. `input_audio` wants the
/// bare payload and a bare format name, not the URL wrapper.
fn split_data_url(data_url: &str) -> Result<(String, String)> {
    let rest = data_url
        .strip_prefix("data:")
        .ok_or_else(|| anyhow::anyhow!("input_audio requires a data URL, got {data_url:.32}"))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("malformed data URL"))?;
    let mime = meta.split(';').next().unwrap_or_default();
    let format = mime.rsplit('/').next().unwrap_or_default();
    if format.is_empty() {
        bail!("data URL carries no media subtype")
    }
    Ok((format.to_string(), payload.to_string()))
}

/// Normalized, backend-agnostic description of one generation request.
pub(crate) struct GenRequest<'a> {
    pub(crate) model: &'a str,
    /// The id this replay knows the request by. Sent in the body as well as in
    /// the `x-request-id` header because the two engines read different ones:
    /// vLLM adopts the header, SGLang takes a body-level `rid`.
    pub(crate) request_id: &'a str,
    pub(crate) prompt: Prompt<'a>,
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
    /// What an operator should do when this transport's server reports no
    /// cached prompt tokens. Backend-specific because the remedy is: SGLang's
    /// OpenAI layer never reports them on any flag, while vLLM's does behind
    /// one, so identical advice would send half of them to the wrong place.
    fn prefix_cache_remedy(&self) -> &'static str {
        "Launch the server with prompt-token details and prefix caching enabled \
         (vLLM: --enable-prompt-tokens-details / ENABLE_PROMPT_TOKENS_DETAILS=1); see README.md."
    }
    /// Shape one generation request into this backend's request body.
    fn build_payload(&self, req: &GenRequest) -> Result<Value>;
    /// Normalize one response JSON object (a stream chunk or a full body).
    fn parse_event(&self, value: &Value) -> StreamEvent;
}

/// Backend result shared by every text-generation source. Source identity stays
/// with the executor; session executors alone carry `output_ids` to the next round.
pub(crate) struct GenerationResult {
    pub(crate) outcome: GenerationOutcome,
    pub(crate) output_ids: Vec<u32>,
    /// Every arrival on this request's stream, in order. Empty when the run is
    /// not recording a timeline, and for a request that never reached the wire.
    pub(crate) timeline: Vec<TimelineEvent>,
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
            output_modality: Modality::Text,
            output_len_actual: 0,
            output_len_text_tokens: 0,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: None,
            submit_timestamp: unix_seconds_now(),
            post_timestamp: None,
            complete_timestamp: unix_seconds_now(),
            first_output_ms: None,
            last_output_ms: None,
            output_bytes: 0,
            output_chunk_count: 0,
            output_duration_ms: None,
            real_time_factor: None,
            artifact_path: None,
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
        timeline: Vec::new(),
    }
}

pub(crate) fn request_build_failure_result(
    request_id: String,
    error: impl std::fmt::Display,
) -> GenerationResult {
    let now = unix_seconds_now();
    GenerationResult {
        outcome: GenerationOutcome {
            request_id,
            output_modality: Modality::Text,
            output_len_actual: 0,
            output_len_text_tokens: 0,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: None,
            submit_timestamp: now,
            post_timestamp: None,
            complete_timestamp: now,
            first_output_ms: None,
            last_output_ms: None,
            output_bytes: 0,
            output_chunk_count: 0,
            output_duration_ms: None,
            real_time_factor: None,
            artifact_path: None,
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
            status: "FAILED".into(),
            output_preview: String::new(),
            error: Some(format!("request build error: {error}")),
        },
        output_ids: Vec::new(),
        timeline: Vec::new(),
    }
}

#[cfg(test)]
mod error_body_tests {
    use super::error_detail;

    #[test]
    fn an_error_envelope_is_unwrapped_to_its_message() {
        // Shape observed from a live vLLM on an unknown model. Reporting only
        // "HTTP 404" hands the operator nothing they can act on.
        let detail = error_detail(
            r#"{"error":{"message":"The model `x` does not exist.","type":"NotFoundError","code":404}}"#,
        )
        .unwrap();
        assert_eq!(detail, "The model `x` does not exist.");
    }

    #[test]
    fn a_bare_message_and_a_plain_body_both_survive() {
        assert_eq!(
            error_detail(r#"{"message":"overloaded"}"#).unwrap(),
            "overloaded"
        );
        assert_eq!(
            error_detail("upstream connect error").unwrap(),
            "upstream connect error"
        );
    }

    #[test]
    fn an_empty_body_adds_nothing_to_the_status() {
        assert!(error_detail("").is_none());
        assert!(error_detail("   \n ").is_none());
    }

    #[test]
    fn a_page_sized_body_is_truncated() {
        let detail = error_detail(&"x".repeat(5_000)).unwrap();
        assert_eq!(detail.chars().count(), 300);
    }
}
