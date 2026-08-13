use serde::Serialize;

use crate::trace::{IndependentRequest, ReplayWorkload, SessionPlans};
use crate::util::reaches_context_limit;

/// Source-specific dry-run summaries. Variants intentionally retain different
/// fields: independent requests do not have session prefix/tool metrics.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WorkloadSummary {
    Sessions(SessionWorkloadSummary),
    IndependentRequests(IndependentRequestSummary),
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SessionWorkloadSummary {
    sessions: usize,
    rounds: usize,
    first_context_overflow_round_idx: Option<usize>,
    first_context_overflow_prompt_len: Option<usize>,
    first_context_overflow_requested_total_len: Option<usize>,
    max_prompt_len: usize,
    max_prefix_len: usize,
    max_input_len: usize,
    max_output_len: usize,
    total_output_len: usize,
    max_arrival_time_ms: f64,
    total_tool_wait_after_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct IndependentRequestSummary {
    requests: usize,
    first_context_overflow_request_id: Option<String>,
    first_context_overflow_input_len: Option<usize>,
    first_context_overflow_requested_total_len: Option<usize>,
    max_input_len: usize,
    max_output_len: usize,
    total_output_len: usize,
    max_arrival_time_ms: f64,
}

impl WorkloadSummary {
    pub(crate) fn from_workload(workload: &ReplayWorkload, max_model_len: Option<usize>) -> Self {
        match workload {
            ReplayWorkload::Sessions(sessions) => Self::Sessions(
                SessionWorkloadSummary::from_sessions(sessions, max_model_len),
            ),
            ReplayWorkload::IndependentRequests(requests) => Self::IndependentRequests(
                IndependentRequestSummary::from_requests(requests, max_model_len),
            ),
        }
    }

    pub(crate) fn total_steps(&self) -> usize {
        match self {
            Self::Sessions(summary) => summary.rounds,
            Self::IndependentRequests(summary) => summary.requests,
        }
    }

    pub(crate) fn max_prompt_len(&self) -> usize {
        match self {
            Self::Sessions(summary) => summary.max_prompt_len,
            Self::IndependentRequests(summary) => summary.max_input_len,
        }
    }

    pub(crate) fn print(&self) {
        match self {
            Self::Sessions(summary) => summary.print(),
            Self::IndependentRequests(summary) => summary.print(),
        }
    }
}

impl SessionWorkloadSummary {
    fn from_sessions(sessions: &SessionPlans, max_model_len: Option<usize>) -> Self {
        let mut summary = Self {
            sessions: sessions.len(),
            rounds: 0,
            first_context_overflow_round_idx: None,
            first_context_overflow_prompt_len: None,
            first_context_overflow_requested_total_len: None,
            max_prompt_len: 0,
            max_prefix_len: 0,
            max_input_len: 0,
            max_output_len: 0,
            total_output_len: 0,
            max_arrival_time_ms: 0.0,
            total_tool_wait_after_ms: 0.0,
        };
        for (_, steps) in sessions.iter() {
            for step in steps {
                let prompt_len = step.prefix_len.saturating_add(step.input_len);
                summary.rounds += 1;
                summary.max_prompt_len = summary.max_prompt_len.max(prompt_len);
                summary.max_prefix_len = summary.max_prefix_len.max(step.prefix_len);
                summary.max_input_len = summary.max_input_len.max(step.input_len);
                summary.max_output_len = summary.max_output_len.max(step.output_len);
                summary.total_output_len += step.output_len;
                summary.max_arrival_time_ms = summary.max_arrival_time_ms.max(step.arrival_time);
                summary.total_tool_wait_after_ms += step.tool_wait_after_ms;
                if max_model_len
                    .is_some_and(|limit| reaches_context_limit(prompt_len, step.output_len, limit))
                    && summary.first_context_overflow_round_idx.is_none()
                {
                    summary.first_context_overflow_round_idx = Some(step.round_idx);
                    summary.first_context_overflow_prompt_len = Some(prompt_len);
                    summary.first_context_overflow_requested_total_len =
                        Some(prompt_len.saturating_add(step.output_len));
                }
            }
        }
        summary
    }

    fn print(&self) {
        eprintln!(
            "session workload | sessions={} rounds={} max_prompt_len={} max_prefix_len={} max_input_len={} max_output_len={} total_output_len={} max_arrival_time_ms={:.3} total_tool_wait_after_ms={:.3}",
            self.sessions,
            self.rounds,
            self.max_prompt_len,
            self.max_prefix_len,
            self.max_input_len,
            self.max_output_len,
            self.total_output_len,
            self.max_arrival_time_ms,
            self.total_tool_wait_after_ms,
        );
    }
}

impl IndependentRequestSummary {
    fn from_requests(requests: &[IndependentRequest], max_model_len: Option<usize>) -> Self {
        let mut summary = Self {
            requests: requests.len(),
            first_context_overflow_request_id: None,
            first_context_overflow_input_len: None,
            first_context_overflow_requested_total_len: None,
            max_input_len: 0,
            max_output_len: 0,
            total_output_len: 0,
            max_arrival_time_ms: 0.0,
        };
        for request in requests {
            summary.max_input_len = summary.max_input_len.max(request.input_len);
            summary.max_output_len = summary.max_output_len.max(request.output_len);
            summary.total_output_len += request.output_len;
            summary.max_arrival_time_ms = summary.max_arrival_time_ms.max(request.arrival_time);
            if max_model_len.is_some_and(|limit| {
                reaches_context_limit(request.input_len, request.output_len, limit)
            }) && summary.first_context_overflow_request_id.is_none()
            {
                summary.first_context_overflow_request_id = Some(request.id.clone());
                summary.first_context_overflow_input_len = Some(request.input_len);
                summary.first_context_overflow_requested_total_len =
                    Some(request.input_len.saturating_add(request.output_len));
            }
        }
        summary
    }

    fn print(&self) {
        eprintln!(
            "independent-request workload | requests={} max_input_len={} max_output_len={} total_output_len={} max_arrival_time_ms={:.3}",
            self.requests,
            self.max_input_len,
            self.max_output_len,
            self.total_output_len,
            self.max_arrival_time_ms,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::SessionStep;

    #[test]
    fn session_static_limit_check_includes_output_and_reserves_headroom() {
        let step = SessionStep {
            request_id: "session_s1_round_000000".to_string(),
            session_id: "session".to_string(),
            arrival_time: 0.0,
            round_idx: 3,
            prefix_len: 80,
            input_len: 10,
            output_len: 10,
            tool_wait_after_ms: 0.0,
        };
        let sessions: SessionPlans = vec![("session".to_string(), vec![step])];

        let summary = SessionWorkloadSummary::from_sessions(&sessions, Some(100));

        assert_eq!(summary.first_context_overflow_round_idx, Some(3));
        assert_eq!(summary.first_context_overflow_prompt_len, Some(90));
        assert_eq!(
            summary.first_context_overflow_requested_total_len,
            Some(100)
        );
    }

    #[test]
    fn independent_static_limit_check_allows_one_token_of_headroom() {
        let request = IndependentRequest {
            id: "request".to_string(),
            input_len: 90,
            output_len: 9,
            arrival_time: 0.0,
        };

        let summary = IndependentRequestSummary::from_requests(&[request], Some(100));

        assert!(summary.first_context_overflow_request_id.is_none());
        assert!(summary.first_context_overflow_requested_total_len.is_none());
    }
}
