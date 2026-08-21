//! Runtime operations over fully parsed input files.
//!
//! Format modules own CSV decoding and structural validation. This module starts
//! after that boundary: it selects a format loader, applies the operator's
//! top-level item limit and arrival-rate override, and summarizes the replayable
//! workload. It deliberately does not define another row schema.

use anyhow::{anyhow, bail, Result};
use serde::Serialize;

use crate::schema::format::text_generation::independent::{self, IndependentRequest};
use crate::schema::format::text_generation::session::{self, SessionPlans};
use crate::schema::{InputFileFormat, InputFileSchema, RequestFamily, TraceTag};
use crate::schema::{Modality, OutputSpec, RequestSpec};
use crate::util::reaches_context_limit;

/// Typed replay plans produced by the format loaders.
pub(crate) enum ReplayWorkload {
    Sessions(SessionPlans),
    IndependentRequests(Vec<IndependentRequest>),
    MultimodalRequests(Vec<RequestSpec>),
}

/// The result of rescaling trace arrival offsets to a requested workload-unit rate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrivalRateAdjustment {
    pub(crate) trace_rate: f64,
    pub(crate) target_rate: f64,
    pub(crate) time_scale: f64,
}

/// Load and validate the whole file, then apply the runtime's item limit.
///
/// Validation intentionally precedes truncation: `--max-items 1` must not hide
/// malformed rows later in the declared input artifact.
pub(crate) fn load_workload(
    path: &str,
    input_file_schema: &InputFileSchema,
    max_items: Option<usize>,
) -> Result<ReplayWorkload> {
    reject_unsupported_replay(input_file_schema)?;
    let mut workload = match input_file_schema.input_file_format {
        InputFileFormat::TextGenerationIndependent => ReplayWorkload::IndependentRequests(
            independent::load_requests(path, input_file_schema)?,
        ),
        InputFileFormat::TextGenerationSessionExecutionV2 => {
            ReplayWorkload::Sessions(session::load(path, input_file_schema)?)
        }
        InputFileFormat::MultimodalIndependentV1 => ReplayWorkload::MultimodalRequests(
            crate::schema::format::multimodal_independent::load(path)?,
        ),
        unsupported => bail!(
            "input file format {:?} is valid, but this HTTP client has no request builder for {:?}",
            unsupported.name(),
            unsupported.request_family(),
        ),
    };
    workload.truncate(max_items);
    Ok(workload)
}

/// Refuse valid input declarations that this HTTP replay client cannot execute.
fn reject_unsupported_replay(input_file_schema: &InputFileSchema) -> Result<()> {
    if input_file_schema.request_family() != RequestFamily::TextGeneration
        && input_file_schema.input_file_format != InputFileFormat::MultimodalIndependentV1
    {
        bail!(
            "request family {:?} is defined by the shared input schema, but this client can only \
             submit text generation: no media prompt builder exists yet",
            input_file_schema.request_family(),
        );
    }
    if input_file_schema.carries(TraceTag::Speculative) {
        bail!(
            "the speculative tag declares an acceptance rate for simulation; a replay client \
             measures the real server's acceptance rate and cannot impose one"
        );
    }
    Ok(())
}

impl ReplayWorkload {
    fn truncate(&mut self, max_items: Option<usize>) {
        let Some(max_items) = max_items else {
            return;
        };
        match self {
            Self::Sessions(sessions) => sessions.truncate(max_items),
            Self::IndependentRequests(requests) => requests.truncate(max_items),
            Self::MultimodalRequests(requests) => requests.truncate(max_items),
        }
    }

    pub(crate) fn unit_count(&self) -> usize {
        match self {
            Self::Sessions(sessions) => sessions.len(),
            Self::IndependentRequests(requests) => requests.len(),
            Self::MultimodalRequests(requests) => requests.len(),
        }
    }

    pub(crate) fn unit_label(&self) -> &'static str {
        match self {
            Self::Sessions(_) => "sessions",
            Self::IndependentRequests(_) | Self::MultimodalRequests(_) => "requests",
        }
    }

    pub(crate) fn arrival_rate(&self) -> Option<f64> {
        match self {
            Self::Sessions(sessions) => arrival_rate(
                sessions
                    .iter()
                    .filter_map(|(_, rounds)| rounds.first())
                    .map(|round| round.arrival_time),
            ),
            Self::IndependentRequests(requests) => {
                arrival_rate(requests.iter().map(|request| request.arrival_time))
            }
            Self::MultimodalRequests(requests) => {
                arrival_rate(requests.iter().map(|request| request.arrival_time_ms))
            }
        }
    }

    pub(crate) fn apply_arrival_rate(&mut self, target_rate: f64) -> Result<ArrivalRateAdjustment> {
        let trace_rate = self.arrival_rate().ok_or_else(|| {
            anyhow!(
                "cannot apply --rate: the selected workload needs at least two items with distinct arrival times"
            )
        })?;
        validate_target_rate(target_rate)?;
        let time_scale = trace_rate / target_rate;
        if !time_scale.is_finite() {
            return Err(anyhow!(
                "--rate produced a non-finite arrival-time scale; choose a less extreme value"
            ));
        }
        match self {
            Self::Sessions(sessions) => {
                for round in sessions.iter_mut().flat_map(|(_, rounds)| rounds) {
                    round.arrival_time *= time_scale;
                }
            }
            Self::IndependentRequests(requests) => {
                for request in requests {
                    request.arrival_time *= time_scale;
                }
            }
            Self::MultimodalRequests(requests) => {
                for request in requests {
                    request.arrival_time_ms *= time_scale;
                }
            }
        }
        Ok(ArrivalRateAdjustment {
            trace_rate,
            target_rate,
            time_scale,
        })
    }
}

fn arrival_rate(arrivals: impl Iterator<Item = f64>) -> Option<f64> {
    let mut arrivals = arrivals.map(|arrival| arrival.max(0.0));
    let first = arrivals.next()?;
    let mut min_arrival_ms = first;
    let mut max_arrival_ms = first;
    let mut count = 1usize;

    for arrival_ms in arrivals {
        if !arrival_ms.is_finite() {
            return None;
        }
        min_arrival_ms = min_arrival_ms.min(arrival_ms);
        max_arrival_ms = max_arrival_ms.max(arrival_ms);
        count += 1;
    }

    let span_seconds = (max_arrival_ms - min_arrival_ms) / 1000.0;
    if count < 2 || !first.is_finite() || span_seconds <= 0.0 {
        return None;
    }
    Some((count - 1) as f64 / span_seconds)
}

fn validate_target_rate(target_rate: f64) -> Result<()> {
    if !target_rate.is_finite() || target_rate <= 0.0 {
        return Err(anyhow!(
            "--rate must be a finite value greater than 0, got {target_rate}"
        ));
    }
    Ok(())
}

/// Source-specific dry-run summaries. Variants intentionally retain different
/// fields: independent requests do not have session prefix/tool metrics.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WorkloadSummary {
    Sessions(SessionWorkloadSummary),
    IndependentRequests(IndependentRequestSummary),
    MultimodalRequests(MultimodalRequestSummary),
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

#[derive(Debug, Clone, Serialize)]
pub(crate) struct MultimodalRequestSummary {
    requests: usize,
    assets: usize,
    input_parts_by_modality: std::collections::BTreeMap<String, usize>,
    output_specs_by_modality: std::collections::BTreeMap<String, usize>,
    max_text_output_tokens: usize,
    total_text_output_tokens: usize,
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
            ReplayWorkload::MultimodalRequests(requests) => {
                Self::MultimodalRequests(MultimodalRequestSummary::from_requests(requests))
            }
        }
    }

    pub(crate) fn total_steps(&self) -> usize {
        match self {
            Self::Sessions(summary) => summary.rounds,
            Self::IndependentRequests(summary) => summary.requests,
            Self::MultimodalRequests(summary) => summary.requests,
        }
    }

    pub(crate) fn max_prompt_len(&self) -> usize {
        match self {
            Self::Sessions(summary) => summary.max_prompt_len,
            Self::IndependentRequests(summary) => summary.max_input_len,
            Self::MultimodalRequests(_) => 0,
        }
    }

    /// Whether this workload's result depends on server-side prefix reuse.
    ///
    /// A session round replays a prefix its earlier rounds already sent, so a
    /// server that silently fails to report — or to perform — cache hits
    /// invalidates the measurement outright. Independent requests each seed at
    /// a distinct pool offset precisely so cross-unit sharing never happens;
    /// their hit rate is ~0 by construction, and holding them to the session
    /// requirement would block runs that do not use the feature at all.
    pub(crate) fn depends_on_prefix_cache(&self) -> bool {
        match self {
            Self::Sessions(_) => true,
            Self::IndependentRequests(_) => false,
        }
    }

    /// Workload units this trace offers — what `--rate` counts.
    ///
    /// A session, not a round: `--rate` rescales session arrivals, and a
    /// session's later rounds are submitted by the session itself as it goes.
    pub(crate) fn workload_units(&self) -> usize {
        match self {
            Self::Sessions(summary) => summary.sessions,
            Self::IndependentRequests(summary) => summary.requests,
            Self::MultimodalRequests(summary) => summary.requests,
        }
    }

    /// Steps one workload unit contributes, on average.
    ///
    /// The conversion between the two rates anyone comparing offered load to
    /// delivered work needs. `--rate` is in units per second and a run's
    /// throughput is in steps per second; on a session trace averaging two
    /// rounds each, those differ by a factor of two, and comparing them
    /// directly reads a saturated server as keeping up with room to spare.
    ///
    /// It is the trace's ratio, not the run's: it says what a server that kept
    /// up perfectly would have to deliver, which is exactly the reference a
    /// saturation test needs. One for an independent trace, where a unit *is* a
    /// step.
    pub(crate) fn steps_per_workload_unit(&self) -> f64 {
        let units = self.workload_units();
        if units == 0 {
            return 1.0;
        }
        self.total_steps() as f64 / units as f64
    }

    pub(crate) fn print(&self) {
        match self {
            Self::Sessions(summary) => summary.print(),
            Self::IndependentRequests(summary) => summary.print(),
            Self::MultimodalRequests(summary) => summary.print(),
        }
    }
}

impl MultimodalRequestSummary {
    fn from_requests(requests: &[RequestSpec]) -> Self {
        let mut summary = Self {
            requests: requests.len(),
            assets: 0,
            input_parts_by_modality: Default::default(),
            output_specs_by_modality: Default::default(),
            max_text_output_tokens: 0,
            total_text_output_tokens: 0,
            max_arrival_time_ms: 0.0,
        };
        for request in requests {
            summary.assets += request.assets().count();
            summary.max_arrival_time_ms = summary.max_arrival_time_ms.max(request.arrival_time_ms);
            for input in &request.inputs {
                *summary
                    .input_parts_by_modality
                    .entry(modality_name(input.modality()).into())
                    .or_default() += 1;
            }
            for output in &request.outputs {
                *summary
                    .output_specs_by_modality
                    .entry(modality_name(output.modality()).into())
                    .or_default() += 1;
                if let OutputSpec::Text { max_tokens } = output {
                    summary.max_text_output_tokens =
                        summary.max_text_output_tokens.max(*max_tokens);
                    summary.total_text_output_tokens += max_tokens;
                }
            }
        }
        summary
    }

    fn print(&self) {
        eprintln!(
            "multimodal workload | requests={} assets={} inputs={:?} outputs={:?} max_text_output_tokens={} total_text_output_tokens={} max_arrival_time_ms={:.3}",
            self.requests,
            self.assets,
            self.input_parts_by_modality,
            self.output_specs_by_modality,
            self.max_text_output_tokens,
            self.total_text_output_tokens,
            self.max_arrival_time_ms,
        );
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Text => "text",
        Modality::Image => "image",
        Modality::Audio => "audio",
        Modality::Video => "video",
        Modality::Tensor => "tensor",
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
    use crate::schema::format::text_generation::session::{request_id, SessionRound};
    use std::fs;

    fn round(session_id: &str, arrival_time: f64, round_idx: usize) -> SessionRound {
        SessionRound {
            request_id: request_id(session_id, round_idx),
            session_id: session_id.to_string(),
            arrival_time,
            round_idx,
            prefix_len: 0,
            input_len: 1,
            output_len: 1,
            tool_wait_after_ms: 0.0,
            slo: Default::default(),
            priority: Default::default(),
            speculative: Default::default(),
        }
    }

    fn session_workload(arrivals: &[f64]) -> ReplayWorkload {
        ReplayWorkload::Sessions(
            arrivals
                .iter()
                .enumerate()
                .map(|(index, &arrival_time)| {
                    let session_id = index.to_string();
                    (
                        session_id.clone(),
                        vec![
                            round(&session_id, arrival_time, 0),
                            round(&session_id, arrival_time, 1),
                        ],
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn rescales_session_starts_and_every_round_to_the_target_rate() {
        let mut workload = session_workload(&[0.0, 500.0, 1_000.0]);
        let adjustment = workload.apply_arrival_rate(4.0).unwrap();

        assert_eq!(adjustment.trace_rate, 2.0);
        assert_eq!(adjustment.target_rate, 4.0);
        assert_eq!(adjustment.time_scale, 0.5);
        assert_eq!(workload.arrival_rate(), Some(4.0));
        let ReplayWorkload::Sessions(sessions) = workload else {
            panic!("expected session workload")
        };
        assert!(sessions[1]
            .1
            .iter()
            .all(|round| round.arrival_time == 250.0));
    }

    #[test]
    fn rejects_invalid_or_unmeasurable_target_rates() {
        let mut valid = session_workload(&[0.0, 1_000.0]);
        assert!(valid.apply_arrival_rate(0.0).is_err());
        assert!(valid.apply_arrival_rate(f64::NAN).is_err());

        let mut simultaneous = session_workload(&[0.0, 0.0]);
        assert!(simultaneous.apply_arrival_rate(1.0).is_err());
    }

    #[test]
    fn validates_the_whole_file_before_applying_the_item_limit() {
        let path = std::env::temp_dir().join(format!(
            "req_frontend_validate_before_truncate_{}.csv",
            std::process::id()
        ));
        fs::write(
            &path,
            "id,arrival_time,input_len,output_len,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms\n\
             valid,0,16,4,100,,\n\
             invalid,1,16,4,0,,\n",
        )
        .unwrap();
        let input_file_schema = InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Slo],
        )
        .unwrap();

        let error = load_workload(path.to_str().unwrap(), &input_file_schema, Some(1))
            .err()
            .expect("the unselected malformed row must still fail validation");
        fs::remove_file(path).ok();

        assert!(error.to_string().contains("line 3"), "{error}");
    }

    #[test]
    fn applies_the_item_limit_after_loading_a_valid_file() {
        let path = std::env::temp_dir().join(format!(
            "req_frontend_truncate_valid_{}.csv",
            std::process::id()
        ));
        fs::write(
            &path,
            "id,arrival_time,input_len,output_len\nfirst,0,16,4\nsecond,1,16,4\n",
        )
        .unwrap();

        let workload = load_workload(
            path.to_str().unwrap(),
            &InputFileSchema::text_generation_independent(),
            Some(1),
        )
        .unwrap();
        fs::remove_file(path).ok();

        let ReplayWorkload::IndependentRequests(requests) = workload else {
            panic!("expected independent requests")
        };
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].id, "first");
    }

    #[test]
    fn session_static_limit_check_includes_output_and_reserves_headroom() {
        let step = SessionRound {
            request_id: "session_s1_round_000000".to_string(),
            session_id: "session".to_string(),
            arrival_time: 0.0,
            round_idx: 3,
            prefix_len: 80,
            input_len: 10,
            output_len: 10,
            tool_wait_after_ms: 0.0,
            slo: Default::default(),
            priority: Default::default(),
            speculative: Default::default(),
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
            slo: Default::default(),
            priority: Default::default(),
        };

        let summary = IndependentRequestSummary::from_requests(&[request], Some(100));

        assert!(summary.first_context_overflow_request_id.is_none());
        assert!(summary.first_context_overflow_requested_total_len.is_none());
    }
}
