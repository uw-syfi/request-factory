//! The wire boundary: normalized request/response vocabulary, the per-protocol
//! adapters that shape it, and the streaming engine that drives one request.
//!
//! The split here is deliberate. `wire/` is pure JSON shaping and knows nothing
//! about time, concurrency, or failure policy. `client` owns the async engine and
//! is protocol-blind. `integrity` holds the checks that decide whether a
//! response can be believed at all, which is a policy question rather than a
//! parsing one. Nothing above this module sees a `serde_json::Value`.

mod client;
mod integrity;
mod preflight;
mod stream;
mod wire;

use serde_json::Value;

use crate::record::GenerationOutcome;
use crate::timeline::TimelineEvent;
use crate::util::unix_seconds_now;

pub(crate) use client::GenerationClient;

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
}

impl Prompt<'_> {
    /// Tokens the server will charge as this prompt's length.
    pub(crate) fn token_len(&self) -> usize {
        let Self::Tokens(prompt_ids) = self;
        prompt_ids.len()
    }
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
    /// Shape one generation request into this backend's request body.
    fn build_payload(&self, req: &GenRequest) -> Value;
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
        timeline: Vec::new(),
    }
}
