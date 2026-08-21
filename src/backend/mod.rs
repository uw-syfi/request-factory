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
mod realtime_client;
mod stream;
mod wire;

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::Serialize;
use serde_json::Value;

use crate::record::GenerationOutcome;
use crate::schema::Modality;
use crate::timeline::TimelineEvent;
use crate::util::unix_seconds_now;

pub(crate) use client::GenerationClient;
pub(crate) use dialect::{dialect_for, Dialect};
pub(crate) use media_client::MediaClient;
pub(crate) use realtime_client::RealtimeClient;

#[cfg(feature = "bench-internals")]
pub(crate) fn bench_serialize_vllm_request(prompt_ids: &[u32]) -> Vec<u8> {
    wire::VllmTokensBackend
        .serialize_payload(&GenRequest {
            model: "bench-model",
            request_id: "bench-request",
            prompt: Prompt::Tokens(prompt_ids),
            max_tokens: 32,
            temperature: 0.0,
            stream: true,
        })
        .expect("benchmark request must serialize")
}

/// Chat-path serialization for the benchmark that guards this body against
/// drifting back to a DOM. Takes the shape rather than the parts because
/// `PreparedInputPart` does not leave the crate.
#[cfg(feature = "bench-internals")]
pub(crate) fn bench_serialize_chat_request(dialect: &str, text: &str, images: usize) -> Vec<u8> {
    let mut parts = vec![PreparedInputPart::Text(text.to_string())];
    for _ in 0..images {
        parts.push(PreparedInputPart::Media {
            modality: Modality::Image,
            data_url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
        });
    }
    wire::OpenAiChatBackend(dialect_for(dialect).expect("known dialect"))
        .serialize_payload(&GenRequest {
            model: "bench-model",
            request_id: "bench-request",
            prompt: Prompt::Parts(&parts),
            max_tokens: 32,
            temperature: 0.0,
            stream: true,
        })
        .expect("benchmark request must serialize")
}

#[cfg(feature = "bench-internals")]
pub(crate) fn bench_parse_vllm_event(data: &[u8]) -> usize {
    let event = wire::VllmTokensBackend
        .parse_event(data)
        .expect("benchmark event must parse");
    event.token_ids.as_ref().map_or(0, Vec::len)
        + usize::from(event.finish_reason.is_some())
        + usize::from(event.usage.is_some())
}

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
/// Covers the whole input half of the request body, not just `messages`,
/// because where media goes is itself a dialect decision: two of the three
/// encodings put it in the message content and the third hangs it off the
/// request root.
///
/// Borrowed and `Serialize` rather than a `Value` builder because two callers
/// want different renderings of the same shaping. Generation serializes it
/// straight into body bytes, where a DOM would allocate a tree only for the
/// HTTP client to walk it again; the media surfaces need a real `Value`, since
/// an operator's `model_params` is arbitrary JSON that has to be merged in.
/// Writing it once and rendering twice is what keeps those two from drifting.
pub(crate) struct ChatInputs<'a> {
    encoding: dialect::MediaInput,
    /// System turns, in trace order. Each is one message.
    system: Vec<&'a str>,
    /// The user turn's parts, in trace order.
    user: Vec<UserPart<'a>>,
}

enum UserPart<'a> {
    Text(&'a str),
    Media {
        modality: Modality,
        data_url: &'a str,
    },
}

impl<'a> ChatInputs<'a> {
    /// Validate the parts against the encoding *before* any serializing.
    ///
    /// Serialization is infallible by the time it runs: a `Serializer` can only
    /// report failure as a serde error, which would surface a dialect mistake
    /// as if the JSON writer had broken. An unsupported modality is an operator
    /// error and reads like one here.
    pub(crate) fn plan(
        parts: &'a [PreparedInputPart],
        encoding: dialect::MediaInput,
    ) -> Result<Self> {
        let mut system = Vec::new();
        let mut user = Vec::new();
        let mut saw_user = false;
        for part in parts {
            match part {
                PreparedInputPart::System(text) if !saw_user => system.push(text.as_str()),
                PreparedInputPart::System(_) => bail!("system inputs must precede user inputs"),
                PreparedInputPart::Text(text) => {
                    saw_user = true;
                    user.push(UserPart::Text(text));
                }
                PreparedInputPart::Media { modality, data_url } => {
                    saw_user = true;
                    // Reject here, where the dialect is in hand, rather than
                    // letting an unnamed modality reach the wire as a field the
                    // server will quietly ignore.
                    media_key(*modality, encoding)?;
                    user.push(UserPart::Media {
                        modality: *modality,
                        data_url,
                    });
                }
            }
        }
        if !saw_user {
            bail!("request has no user input")
        }
        // Under `TopLevelLists` the user turn is a plain string, so a
        // media-only request still has a turn; the other two encodings build a
        // content array and would emit an empty one.
        if !matches!(encoding, dialect::MediaInput::TopLevelLists)
            && !user.iter().any(|part| matches!(part, UserPart::Text(_)))
            && user.is_empty()
        {
            bail!("request has no user input")
        }
        Ok(Self {
            encoding,
            system,
            user,
        })
    }

    /// The same shaping as a `Value` object, for the media surfaces that must
    /// merge arbitrary operator JSON into it.
    pub(crate) fn to_object(&self) -> Result<serde_json::Map<String, Value>> {
        match serde_json::to_value(self)? {
            Value::Object(map) => Ok(map),
            other => bail!("chat inputs rendered as {other} rather than an object"),
        }
    }
}

impl ChatInputs<'_> {
    /// Write the input half of a chat body into a map the caller already opened.
    ///
    /// Taking an open map rather than returning one is what lets a backend
    /// flatten these entries in beside `model` and the decode cap: under one
    /// encoding the input half is more than `messages`, so it cannot be a
    /// single nested field.
    pub(crate) fn serialize_entries<M: SerializeMap>(
        &self,
        map: &mut M,
    ) -> std::result::Result<(), M::Error> {
        map.serialize_entry("messages", &Messages(self))?;
        if matches!(self.encoding, dialect::MediaInput::TopLevelLists) {
            // Grouped by key so `images` / `audios` / `videos` each arrive as
            // one array, whatever order the trace interleaved them in.
            let mut lists: BTreeMap<&'static str, Vec<&str>> = BTreeMap::new();
            for part in &self.user {
                if let UserPart::Media { modality, data_url } = part {
                    // `plan` already proved every modality has a key here.
                    if let Ok(key) = media_list_key(*modality) {
                        lists.entry(key).or_default().push(data_url);
                    }
                }
            }
            for (key, urls) in lists {
                map.serialize_entry(key, &urls)?;
            }
        }
        Ok(())
    }
}

impl Serialize for ChatInputs<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        self.serialize_entries(&mut map)?;
        map.end()
    }
}

/// The `messages` array: every system turn, then one user turn.
struct Messages<'a>(&'a ChatInputs<'a>);

impl Serialize for Messages<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let inputs = self.0;
        let mut seq = serializer.serialize_seq(Some(inputs.system.len() + 1))?;
        for text in &inputs.system {
            seq.serialize_element(&Message {
                role: "system",
                content: MessageContent::Text(text),
            })?;
        }
        seq.serialize_element(&Message {
            role: "user",
            content: MessageContent::User(inputs),
        })?;
        seq.end()
    }
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: MessageContent<'a>,
}

enum MessageContent<'a> {
    Text(&'a str),
    User(&'a ChatInputs<'a>),
}

impl Serialize for MessageContent<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let inputs = match self {
            Self::Text(text) => return serializer.serialize_str(text),
            Self::User(inputs) => inputs,
        };
        if matches!(inputs.encoding, dialect::MediaInput::TopLevelLists) {
            // Media left the turn for the request root, so what remains is
            // text; joined rather than arrayed because this encoding's
            // `content` is a string.
            let joined = inputs
                .user
                .iter()
                .filter_map(|part| match part {
                    UserPart::Text(text) => Some(*text),
                    UserPart::Media { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return serializer.serialize_str(&joined);
        }
        let mut seq = serializer.serialize_seq(Some(inputs.user.len()))?;
        for part in &inputs.user {
            match part {
                UserPart::Text(text) => seq.serialize_element(&TextPart { text })?,
                UserPart::Media { modality, data_url } => {
                    seq.serialize_element(&MediaPart {
                        modality: *modality,
                        data_url,
                        encoding: inputs.encoding,
                    })?;
                }
            }
        }
        seq.end()
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "text")]
struct TextPart<'a> {
    text: &'a str,
}

/// One media content part, spelled the way the encoding spells it.
struct MediaPart<'a> {
    modality: Modality,
    data_url: &'a str,
    encoding: dialect::MediaInput,
}

impl Serialize for MediaPart<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        // `plan` rejected any modality this encoding cannot name, so the key is
        // known to exist by the time a part is written.
        let key = media_key(self.modality, self.encoding).map_err(serde::ser::Error::custom)?;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", key)?;
        if key == "input_audio" {
            // OpenAI proper wants the bare payload and a bare format name
            // rather than the data-URL wrapper the `_url` family carries.
            let (format, data) =
                split_data_url(self.data_url).map_err(serde::ser::Error::custom)?;
            map.serialize_entry(key, &InputAudio { data, format })?;
        } else {
            map.serialize_entry(key, &MediaUrl { url: self.data_url })?;
        }
        map.end()
    }
}

#[derive(Serialize)]
struct MediaUrl<'a> {
    url: &'a str,
}

#[derive(Serialize)]
struct InputAudio<'a> {
    data: &'a str,
    format: &'a str,
}

/// The content-part key one encoding uses for one modality, or an error naming
/// what that dialect cannot carry.
///
/// `TopLevelLists` has no content-part key at all; it answers for whether the
/// modality is carryable, which is what `plan` asks.
fn media_key(modality: Modality, encoding: dialect::MediaInput) -> Result<&'static str> {
    use dialect::MediaInput;
    match encoding {
        // vLLM's `_url` family, which is also the only spelling video has anywhere.
        MediaInput::UrlParts => match modality {
            Modality::Image => Ok("image_url"),
            Modality::Audio => Ok("audio_url"),
            Modality::Video => Ok("video_url"),
            Modality::Text | Modality::Tensor => {
                bail!("unsupported prepared media modality {modality:?}")
            }
        },
        // OpenAI proper: images by URL, audio as `input_audio`, and no video at all.
        MediaInput::OpenAiParts => match modality {
            Modality::Image => Ok("image_url"),
            Modality::Audio => Ok("input_audio"),
            Modality::Video => bail!("the openai dialect has no video input content part"),
            Modality::Text | Modality::Tensor => {
                bail!("unsupported prepared media modality {modality:?}")
            }
        },
        MediaInput::TopLevelLists => media_list_key(modality),
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

/// `data:audio/wav;base64,AAAA` -> `("wav", "AAAA")`. Borrowed from the input:
/// `input_audio` wants the bare payload and a bare format name, and copying a
/// base64 audio blob to strip four characters is the kind of allocation this
/// path exists to avoid.
fn split_data_url(data_url: &str) -> Result<(&str, &str)> {
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
    Ok((format, payload))
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
    /// Serialize one generation request directly into this backend's JSON wire
    /// representation. Keeping the DOM out of this hot path avoids allocating
    /// a tree only for reqwest to immediately walk it again.
    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>>;
    /// Deserialize and normalize one response JSON object directly from its
    /// SSE bytes.
    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent>;
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
