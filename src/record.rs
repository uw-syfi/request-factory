use serde::Serialize;

use crate::trace::{IndependentRequest, SessionStep};
use crate::util::prefix_hit_rate;

const STEP_LOG_SCHEMA_VERSION: u32 = 7;

/// One JSONL record: a typed source plus measurements shared by text generation.
///
/// Source-specific fields live inside the tagged [`SourceRecord`] variants. The
/// common envelope never grows nullable session/multimodal fields as new
/// frontends are added.
#[derive(Debug, Serialize)]
pub(crate) struct StepLog {
    schema_version: u32,
    pub(crate) source: SourceRecord,
    pub(crate) outcome: GenerationOutcome,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub(crate) enum SourceRecord {
    IndependentRequest(IndependentRequestSource),
    SessionRound(SessionRoundSource),
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionRoundSource {
    pub(crate) session_id: String,
    pub(crate) round_idx: usize,
    pub(crate) prefix_len: usize,
    pub(crate) input_len: usize,
    /// Trace-reported total prompt target (`prefix_len + input_len`).
    pub(crate) target_prompt_len: usize,
    pub(crate) prompt_len: usize,
    pub(crate) session_context_policy: String,
    /// Cache-reusable prefix and newly appended tokens actually constructed by
    /// the selected policy. These, rather than the raw trace split, define the
    /// planned cache hit rate.
    pub(crate) derived_prefix_len: usize,
    pub(crate) derived_append_len: usize,
    pub(crate) major_compaction: bool,
    pub(crate) planned_prefix_hit_rate: Option<f64>,
    pub(crate) output_len_target: usize,
    pub(crate) tool_wait_after_ms: f64,
    pub(crate) arrival_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct IndependentRequestSource {
    pub(crate) id: String,
    pub(crate) input_len: usize,
    pub(crate) prompt_len: usize,
    pub(crate) output_len_target: usize,
    pub(crate) arrival_time_ms: f64,
    /// Delay from the trace's scheduled arrival to this Tokio task resuming.
    /// Measured before the optional concurrency semaphore, so an intentional
    /// client-side concurrency cap is not misreported as runtime scheduler lag.
    pub(crate) arrival_release_lag_ms: f64,
}

/// Measurements that have the same meaning for every text-generation source.
#[derive(Debug, Serialize)]
pub(crate) struct GenerationOutcome {
    pub(crate) request_id: String,
    pub(crate) output_len_actual: usize,
    pub(crate) output_len_text_tokens: usize,
    /// Leading generated ids that repeated the prompt tail and were dropped
    /// before carry-forward. Non-zero only on servers that echo part of the
    /// prompt into their generated ids; see `classify_prompt_echo`.
    pub(crate) echoed_prompt_tokens: usize,
    /// `None` means the server returned no usage object. Fields inside a present
    /// usage object remain optional when that server does not expose the metric.
    pub(crate) server_usage: Option<ServerUsageLog>,
    pub(crate) finish_reason: Option<String>,
    /// Wall-clock seconds when the request entered the client.
    pub(crate) submit_timestamp: f64,
    /// Wall-clock seconds immediately before HTTP send; absent if never sent.
    pub(crate) post_timestamp: Option<f64>,
    /// Wall-clock seconds when the response completed or the request was skipped.
    pub(crate) complete_timestamp: f64,
    /// Legacy TTFT measured from HTTP send to the first non-empty text event.
    pub(crate) first_token_ms: Option<f64>,
    /// Time from HTTP send to the first event carrying generated token ids.
    pub(crate) first_token_id_ms: Option<f64>,
    /// Time from HTTP send to the last event carrying generated token ids.
    pub(crate) last_token_id_ms: Option<f64>,
    /// Tokens delivered in the first timed event. They share one observable
    /// arrival boundary and are excluded from token-delivery TPOT's denominator.
    pub(crate) first_token_event_tokens: usize,
    /// Number of parsed events carrying at least one generated token id.
    pub(crate) token_event_count: usize,
    /// Number of parsed events carrying a usage object.
    pub(crate) usage_event_count: usize,
    /// Average client-observed delivery time for tokens arriving after the
    /// first token event. Unlike the legacy formula, this excludes terminal
    /// usage/[DONE] and client post-processing.
    pub(crate) token_delivery_tpot_ms: Option<f64>,
    /// Time from HTTP send until the response stream reached [DONE] or EOF,
    /// before output re-tokenization and logging bookkeeping.
    pub(crate) response_complete_ms: Option<f64>,
    /// Response-completion delay after the last generated-token-id event.
    pub(crate) terminal_tail_ms: Option<f64>,
    pub(crate) total_duration_ms: f64,
    /// All parsed JSON SSE objects, including usage-only objects. This is not a
    /// token count; use server usage and `token_event_count` for their contracts.
    pub(crate) chunk_count: usize,
    pub(crate) status: String,
    pub(crate) output_preview: String,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ServerUsageLog {
    pub(crate) prompt_tokens: Option<usize>,
    pub(crate) completion_tokens: Option<usize>,
    pub(crate) total_tokens: Option<usize>,
    pub(crate) cached_prompt_tokens: Option<usize>,
    pub(crate) uncached_prompt_tokens: Option<usize>,
    pub(crate) prefix_hit_rate: Option<f64>,
}

impl StepLog {
    pub(crate) fn session_round(
        step: &SessionStep,
        prompt_len: usize,
        session_context_policy: &str,
        derived_prefix_len: usize,
        derived_append_len: usize,
        major_compaction: bool,
        outcome: GenerationOutcome,
    ) -> Self {
        Self {
            schema_version: STEP_LOG_SCHEMA_VERSION,
            source: SourceRecord::SessionRound(SessionRoundSource {
                session_id: step.session_id.clone(),
                round_idx: step.round_idx,
                prefix_len: step.prefix_len,
                input_len: step.input_len,
                target_prompt_len: step.prefix_len.saturating_add(step.input_len),
                prompt_len,
                session_context_policy: session_context_policy.to_string(),
                derived_prefix_len,
                derived_append_len,
                major_compaction,
                planned_prefix_hit_rate: Some(prefix_hit_rate(derived_prefix_len, prompt_len)),
                output_len_target: step.output_len,
                tool_wait_after_ms: step.tool_wait_after_ms,
                arrival_time_ms: step.arrival_time,
            }),
            outcome,
        }
    }

    pub(crate) fn independent_request(
        request: &IndependentRequest,
        prompt_len: usize,
        arrival_release_lag_ms: f64,
        outcome: GenerationOutcome,
    ) -> Self {
        Self {
            schema_version: STEP_LOG_SCHEMA_VERSION,
            source: SourceRecord::IndependentRequest(IndependentRequestSource {
                id: request.id.clone(),
                input_len: request.input_len,
                prompt_len,
                output_len_target: request.output_len,
                arrival_time_ms: request.arrival_time,
                arrival_release_lag_ms,
            }),
            outcome,
        }
    }
}

impl GenerationOutcome {
    pub(crate) fn is_success(&self) -> bool {
        self.status == "SUCCESS"
    }

    pub(crate) fn is_context_overflow(&self) -> bool {
        self.status == "SKIPPED_CONTEXT_OVERFLOW"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome() -> GenerationOutcome {
        GenerationOutcome {
            request_id: "request-1".to_string(),
            output_len_actual: 4,
            output_len_text_tokens: 4,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: Some("length".to_string()),
            submit_timestamp: 1.0,
            post_timestamp: Some(1.1),
            complete_timestamp: 1.2,
            first_token_ms: Some(10.0),
            first_token_id_ms: Some(10.0),
            last_token_id_ms: Some(19.0),
            first_token_event_tokens: 1,
            token_event_count: 4,
            usage_event_count: 1,
            token_delivery_tpot_ms: Some(3.0),
            response_complete_ms: Some(19.5),
            terminal_tail_ms: Some(0.5),
            total_duration_ms: 20.0,
            chunk_count: 4,
            status: "SUCCESS".to_string(),
            output_preview: String::new(),
            error: None,
        }
    }

    #[test]
    fn session_record_keeps_session_fields_inside_tagged_source() {
        let step = SessionStep {
            session_id: "s1".to_string(),
            arrival_time: 0.0,
            round_idx: 2,
            prefix_len: 8,
            input_len: 4,
            output_len: 3,
            tool_wait_after_ms: 5.0,
        };
        let value = serde_json::to_value(StepLog::session_round(
            &step,
            12,
            "trace_reported",
            8,
            4,
            false,
            outcome(),
        ))
        .unwrap();

        assert_eq!(value["source"]["type"], "session_round");
        assert_eq!(value["schema_version"], 7);
        assert_eq!(value["source"]["data"]["prefix_len"], 8);
        assert_eq!(value["source"]["data"]["derived_prefix_len"], 8);
        assert_eq!(value["source"]["data"]["derived_append_len"], 4);
        assert_eq!(value["source"]["data"]["target_prompt_len"], 12);
        assert!(value["source"]["data"].get("id").is_none());
    }

    #[test]
    fn independent_record_has_no_session_placeholder_fields() {
        let request = IndependentRequest {
            id: "r1".to_string(),
            input_len: 12,
            output_len: 3,
            arrival_time: 0.0,
        };
        let value =
            serde_json::to_value(StepLog::independent_request(&request, 12, 0.25, outcome()))
                .unwrap();

        assert_eq!(value["source"]["type"], "independent_request");
        let data = value["source"]["data"].as_object().unwrap();
        assert_eq!(data["id"], "r1");
        assert_eq!(data["arrival_release_lag_ms"], 0.25);
        assert!(!data.contains_key("session_id"));
        assert!(!data.contains_key("round_idx"));
        assert!(!data.contains_key("prefix_len"));
        assert!(!data.contains_key("tool_wait_after_ms"));
    }
}
