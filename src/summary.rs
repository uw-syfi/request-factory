use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use tokio::sync::mpsc;

use crate::record::{GenerationOutcome, SessionRoundSource, SourceRecord, StepLog};
use crate::schema::slo::SloSummary;
use crate::timeline::TimelineSummary;
use crate::trace::ReplayWorkload;
use crate::util::ratio;
use crate::workload::WorkloadSummary;

/// Replay summaries retain the workload type instead of exposing session-only
/// cache statistics as an optional block on every source.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ReplaySummary {
    Sessions {
        common: CommonReplaySummary,
        prefix_cache: PrefixCacheSummary,
    },
    IndependentRequests {
        common: CommonReplaySummary,
    },
}

/// Statistics with identical meaning for every text-generation source.
#[derive(Debug, Default, Serialize)]
pub(crate) struct CommonReplaySummary {
    attempted_steps: usize,
    success_steps: usize,
    failed_steps: usize,
    target_output_tokens: usize,
    actual_output_tokens: usize,
    output_mismatch_steps: usize,
    output_token_delta: i64,
    total_duration_ms_sum: f64,
    total_duration_ms_avg: f64,
    total_duration_ms_p50: f64,
    total_duration_ms_p90: f64,
    total_duration_ms_max: f64,
    ttft_ms_avg: Option<f64>,
    ttft_ms_p50: Option<f64>,
    ttft_ms_p90: Option<f64>,
    ttft_ms_max: Option<f64>,
    ttft_token_id_steps: usize,
    ttft_text_fallback_steps: usize,
    run_duration_ms: Option<f64>,
    request_throughput_per_s: Option<f64>,
    output_token_throughput_per_s: Option<f64>,
    tpot_measured_steps: usize,
    tpot_ms_avg: Option<f64>,
    tpot_ms_p50: Option<f64>,
    tpot_ms_p90: Option<f64>,
    tpot_ms_max: Option<f64>,
    completion_amortized_tpot_measured_steps: usize,
    completion_amortized_tpot_ms_avg: Option<f64>,
    completion_amortized_tpot_ms_p50: Option<f64>,
    completion_amortized_tpot_ms_p90: Option<f64>,
    completion_amortized_tpot_ms_max: Option<f64>,
    context_overflow_steps: usize,
}

/// Raw observations needed for cross-request metrics.
///
/// Keep these out of the serialized schema: the summary exposes only stable,
/// named metrics rather than implementation accumulators or timestamp bounds.
#[derive(Debug, Default)]
struct ReplayMeasurements {
    total_durations_ms: Vec<f64>,
    ttfts_ms: Vec<f64>,
    tpots_ms: Vec<f64>,
    completion_amortized_tpots_ms: Vec<f64>,
    first_submit_timestamp: Option<f64>,
    last_complete_timestamp: Option<f64>,
    successful_output_tokens: usize,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct PrefixCacheSummary {
    planned_prefix_tokens: usize,
    planned_prompt_tokens: usize,
    planned_prefix_hit_rate: Option<f64>,
    measured_cache_steps: usize,
    measured_server_cached_prompt_tokens: usize,
    measured_server_prompt_tokens: usize,
    planned_prefix_tokens_for_measured_cache_steps: usize,
    planned_prompt_tokens_for_measured_cache_steps: usize,
    planned_prefix_hit_rate_for_measured_cache_steps: Option<f64>,
    server_prefix_hit_rate: Option<f64>,
    server_prefix_hit_rate_delta: Option<f64>,
}

impl ReplaySummary {
    pub(crate) fn empty_for(workload: &ReplayWorkload) -> Self {
        match workload {
            ReplayWorkload::Sessions(_) => Self::Sessions {
                common: CommonReplaySummary::default(),
                prefix_cache: PrefixCacheSummary::default(),
            },
            ReplayWorkload::IndependentRequests(_) => Self::IndependentRequests {
                common: CommonReplaySummary::default(),
            },
        }
    }

    fn add(&mut self, record: &StepLog) {
        match (self, &record.source) {
            (
                Self::Sessions {
                    common,
                    prefix_cache,
                },
                SourceRecord::SessionRound(source),
            ) => {
                common.add(source.output_len_target, &record.outcome);
                prefix_cache.add(source, &record.outcome);
            }
            (Self::IndependentRequests { common }, SourceRecord::IndependentRequest(source)) => {
                common.add(source.output_len_target, &record.outcome)
            }
            _ => unreachable!("log source must match the selected replay workload"),
        }
    }

    fn finalize(&mut self, measurements: &mut ReplayMeasurements) {
        let common = match self {
            Self::Sessions { common, .. } | Self::IndependentRequests { common } => common,
        };
        common.finalize(measurements);

        if let Self::Sessions { prefix_cache, .. } = self {
            prefix_cache.finalize();
        }
    }
}

impl CommonReplaySummary {
    fn add(&mut self, output_len_target: usize, outcome: &GenerationOutcome) {
        self.attempted_steps += 1;
        if outcome.is_success() {
            self.success_steps += 1;
        } else {
            self.failed_steps += 1;
        }
        if outcome.is_context_overflow() {
            self.context_overflow_steps += 1;
        }
        self.target_output_tokens += output_len_target;
        self.actual_output_tokens += outcome.output_len_actual;
        if outcome.is_success() && outcome.output_len_actual != output_len_target {
            self.output_mismatch_steps += 1;
            self.output_token_delta += outcome.output_len_actual as i64 - output_len_target as i64;
        }
        self.total_duration_ms_sum += outcome.total_duration_ms;
        if outcome.first_token_id_ms.is_some() {
            self.ttft_token_id_steps += 1;
        } else if outcome.first_token_ms.is_some() {
            self.ttft_text_fallback_steps += 1;
        }
    }

    fn finalize(&mut self, measurements: &mut ReplayMeasurements) {
        if !measurements.total_durations_ms.is_empty() {
            measurements
                .total_durations_ms
                .sort_by(|a, b| a.total_cmp(b));
            self.total_duration_ms_avg =
                self.total_duration_ms_sum / measurements.total_durations_ms.len() as f64;
            self.total_duration_ms_p50 = percentile_sorted(&measurements.total_durations_ms, 0.50);
            self.total_duration_ms_p90 = percentile_sorted(&measurements.total_durations_ms, 0.90);
            self.total_duration_ms_max = measurements
                .total_durations_ms
                .last()
                .copied()
                .unwrap_or(0.0);
        }

        if !measurements.ttfts_ms.is_empty() {
            measurements.ttfts_ms.sort_by(|a, b| a.total_cmp(b));
            let sum: f64 = measurements.ttfts_ms.iter().sum();
            self.ttft_ms_avg = Some(sum / measurements.ttfts_ms.len() as f64);
            self.ttft_ms_p50 = Some(percentile_sorted(&measurements.ttfts_ms, 0.50));
            self.ttft_ms_p90 = Some(percentile_sorted(&measurements.ttfts_ms, 0.90));
            self.ttft_ms_max = measurements.ttfts_ms.last().copied();
        }

        if !measurements.tpots_ms.is_empty() {
            measurements.tpots_ms.sort_by(|a, b| a.total_cmp(b));
            self.tpot_measured_steps = measurements.tpots_ms.len();
            let sum: f64 = measurements.tpots_ms.iter().sum();
            self.tpot_ms_avg = Some(sum / self.tpot_measured_steps as f64);
            self.tpot_ms_p50 = Some(percentile_sorted(&measurements.tpots_ms, 0.50));
            self.tpot_ms_p90 = Some(percentile_sorted(&measurements.tpots_ms, 0.90));
            self.tpot_ms_max = measurements.tpots_ms.last().copied();
        }

        if !measurements.completion_amortized_tpots_ms.is_empty() {
            measurements
                .completion_amortized_tpots_ms
                .sort_by(|left, right| left.total_cmp(right));
            self.completion_amortized_tpot_measured_steps =
                measurements.completion_amortized_tpots_ms.len();
            let sum: f64 = measurements.completion_amortized_tpots_ms.iter().sum();
            self.completion_amortized_tpot_ms_avg =
                Some(sum / self.completion_amortized_tpot_measured_steps as f64);
            self.completion_amortized_tpot_ms_p50 = Some(percentile_sorted(
                &measurements.completion_amortized_tpots_ms,
                0.50,
            ));
            self.completion_amortized_tpot_ms_p90 = Some(percentile_sorted(
                &measurements.completion_amortized_tpots_ms,
                0.90,
            ));
            self.completion_amortized_tpot_ms_max =
                measurements.completion_amortized_tpots_ms.last().copied();
        }

        if let (Some(first_submit), Some(last_complete)) = (
            measurements.first_submit_timestamp,
            measurements.last_complete_timestamp,
        ) {
            let duration_ms = ((last_complete - first_submit) * 1_000.0).max(0.0);
            self.run_duration_ms = Some(duration_ms);
            if duration_ms > 0.0 {
                let duration_s = duration_ms / 1_000.0;
                self.request_throughput_per_s = Some(self.success_steps as f64 / duration_s);
                self.output_token_throughput_per_s =
                    Some(measurements.successful_output_tokens as f64 / duration_s);
            }
        }
    }
}

impl ReplayMeasurements {
    fn add(&mut self, outcome: &GenerationOutcome) {
        self.total_durations_ms.push(outcome.total_duration_ms);
        if let Some(ttft_ms) = outcome.first_token_id_ms {
            self.ttfts_ms.push(ttft_ms);
        } else if let Some(ttft_ms) = outcome.first_token_ms {
            self.ttfts_ms.push(ttft_ms);
        }
        self.first_submit_timestamp = Some(
            self.first_submit_timestamp
                .map_or(outcome.submit_timestamp, |current| {
                    current.min(outcome.submit_timestamp)
                }),
        );
        self.last_complete_timestamp = Some(
            self.last_complete_timestamp
                .map_or(outcome.complete_timestamp, |current| {
                    current.max(outcome.complete_timestamp)
                }),
        );

        if !outcome.is_success() {
            return;
        }
        self.successful_output_tokens += outcome.output_len_actual;
        if let Some(tpot_ms) = outcome.token_delivery_tpot_ms {
            self.tpots_ms.push(tpot_ms);
        }
        if let Some(ttft_ms) = outcome.first_token_ms.or(outcome.first_token_id_ms) {
            if outcome.output_len_actual > 1 && outcome.total_duration_ms >= ttft_ms {
                self.completion_amortized_tpots_ms.push(
                    (outcome.total_duration_ms - ttft_ms) / (outcome.output_len_actual - 1) as f64,
                );
            }
        }
    }
}

impl PrefixCacheSummary {
    fn add(&mut self, source: &SessionRoundSource, outcome: &GenerationOutcome) {
        self.planned_prefix_tokens += source.derived_prefix_len;
        self.planned_prompt_tokens += source.prompt_len;
        if let Some(usage) = &outcome.server_usage {
            if let (Some(cached), Some(prompt)) = (usage.cached_prompt_tokens, usage.prompt_tokens)
            {
                self.measured_cache_steps += 1;
                self.measured_server_cached_prompt_tokens += cached;
                self.measured_server_prompt_tokens += prompt;
                self.planned_prefix_tokens_for_measured_cache_steps += source.derived_prefix_len;
                self.planned_prompt_tokens_for_measured_cache_steps += source.prompt_len;
            }
        }
    }

    fn finalize(&mut self) {
        self.planned_prefix_hit_rate =
            ratio(self.planned_prefix_tokens, self.planned_prompt_tokens);
        self.planned_prefix_hit_rate_for_measured_cache_steps = ratio(
            self.planned_prefix_tokens_for_measured_cache_steps,
            self.planned_prompt_tokens_for_measured_cache_steps,
        );
        self.server_prefix_hit_rate = ratio(
            self.measured_server_cached_prompt_tokens,
            self.measured_server_prompt_tokens,
        );
        self.server_prefix_hit_rate_delta = match (
            self.server_prefix_hit_rate,
            self.planned_prefix_hit_rate_for_measured_cache_steps,
        ) {
            (Some(actual), Some(planned)) => Some(actual - planned),
            _ => None,
        };
    }
}

/// Combined dry-run/full-run JSON summary: workload shape plus replay results.
#[derive(Debug, Serialize)]
pub struct RunSummary {
    pub(crate) workload: WorkloadSummary,
    pub(crate) replay: ReplaySummary,
    pub(crate) client_runtime: ClientRuntimeSummary,
    /// What the per-event timeline recorded, including what it had to drop.
    pub(crate) timeline: TimelineSummary,
    /// Attainment against the objective this run was held to, or `None` when no
    /// objective was declared. Serialized as null rather than omitted, so a
    /// consumer can tell "no SLO" apart from "an older summary format".
    pub slo: Option<SloSummary>,
}

/// What draining the log stream produced.
///
/// Two folds over one pass: the replay statistics every run reports, and the
/// SLO attainment a run reports only when it was given an objective.
pub(crate) struct LogFold {
    pub(crate) replay: ReplaySummary,
    pub(crate) slo: Option<SloSummary>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ClientRuntimeSummary {
    /// Actual Tokio worker population of this process, not a CPU-count guess.
    pub(crate) tokio_worker_threads: usize,
    /// Peak sampled depth of Tokio's global injection queue. Worker-local
    /// queues are not included, so this is one backlog signal rather than a
    /// claim that every runnable task is counted.
    pub(crate) sampled_global_queue_depth_peak: usize,
}

/// Drain logged requests to JSONL on disk and fold them into a typed replay summary.
///
/// SLO attainment folds in the same pass rather than in a second one: every step
/// is already in hand here, and a separate statistics path over the same records
/// is how two numbers that must agree stop agreeing.
pub(crate) async fn write_logs(
    path: String,
    mut rx: mpsc::Receiver<StepLog>,
    mut summary: ReplaySummary,
    mut slo: Option<SloSummary>,
) -> LogFold {
    let file = File::create(&path).expect("failed to create log file");
    let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
    let mut measurements = ReplayMeasurements::default();

    while let Some(record) = rx.recv().await {
        summary.add(&record);
        measurements.add(&record.outcome);
        if let Some(slo) = slo.as_mut() {
            slo.add(&record.slo_measurement());
        }
        log_server_prefix_hit_rate(&record);
        if let Ok(json) = serde_json::to_string(&record) {
            let _ = writeln!(writer, "{json}");
            let _ = writer.flush();
        }
    }
    let _ = writer.flush();
    summary.finalize(&mut measurements);
    log_server_prefix_hit_rate_summary(&summary);
    if let Some(slo) = slo.as_ref() {
        eprintln!("{}", slo.describe());
    }
    LogFold {
        replay: summary,
        slo,
    }
}

fn log_server_prefix_hit_rate(record: &StepLog) {
    if !record.outcome.is_success() {
        return;
    }
    let SourceRecord::SessionRound(source) = &record.source else {
        return;
    };
    let Some(usage) = &record.outcome.server_usage else {
        return;
    };
    if let (Some(planned), Some(actual), Some(cached), Some(prompt)) = (
        source.planned_prefix_hit_rate,
        usage.prefix_hit_rate,
        usage.cached_prompt_tokens,
        usage.prompt_tokens,
    ) {
        eprintln!(
            "prefix hit rate | request_id={} planned={:.4} actual={:.4} delta={:+.4} server_cached_prompt_tokens={} server_prompt_tokens={}",
            record.outcome.request_id,
            planned,
            actual,
            actual - planned,
            cached,
            prompt,
        );
    }
}

fn log_server_prefix_hit_rate_summary(summary: &ReplaySummary) {
    let ReplaySummary::Sessions { prefix_cache, .. } = summary else {
        return;
    };
    if let (Some(actual), Some(planned)) = (
        prefix_cache.server_prefix_hit_rate,
        prefix_cache.planned_prefix_hit_rate_for_measured_cache_steps,
    ) {
        eprintln!(
            "prefix hit rate summary | measured_steps={} planned={:.4} actual={:.4} delta={:+.4} server_cached_prompt_tokens={} server_prompt_tokens={}",
            prefix_cache.measured_cache_steps,
            planned,
            actual,
            actual - planned,
            prefix_cache.measured_server_cached_prompt_tokens,
            prefix_cache.measured_server_prompt_tokens,
        );
    } else {
        eprintln!(
            "prefix hit rate summary | measured_steps=0 actual unavailable; no server cached prompt token details were reported"
        );
    }
}

fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let pos = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = pos - lo as f64;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}

/// Write the combined run summary to `--summary-path` when one was requested.
pub(crate) fn write_summary_if_requested(
    summary_path: Option<&str>,
    summary: &RunSummary,
) -> Result<()> {
    let Some(path) = summary_path else {
        return Ok(());
    };
    let file = File::create(path).with_context(|| format!("failed to create summary: {path}"))?;
    serde_json::to_writer_pretty(file, summary)
        .with_context(|| format!("failed to write summary: {path}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(
        request_id: &str,
        submit_timestamp: f64,
        complete_timestamp: f64,
        total_duration_ms: f64,
        first_token_ms: Option<f64>,
        output_len_actual: usize,
        status: &str,
    ) -> GenerationOutcome {
        let token_delivery_tpot_ms = match first_token_ms {
            Some(first) if output_len_actual > 1 && total_duration_ms >= first => {
                Some((total_duration_ms - first) / (output_len_actual - 1) as f64)
            }
            _ => None,
        };
        GenerationOutcome {
            request_id: request_id.to_string(),
            output_len_actual,
            output_len_text_tokens: output_len_actual,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: None,
            submit_timestamp,
            post_timestamp: Some(submit_timestamp),
            complete_timestamp,
            first_token_ms,
            first_token_id_ms: first_token_ms,
            last_token_id_ms: None,
            first_token_event_tokens: 0,
            token_event_count: 0,
            usage_event_count: 0,
            token_delivery_tpot_ms,
            response_complete_ms: None,
            terminal_tail_ms: None,
            total_duration_ms,
            chunk_count: output_len_actual,
            status: status.to_string(),
            output_preview: String::new(),
            error: None,
        }
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("metric should be measured");
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn overall_metrics_use_wall_clock_and_successful_outputs() {
        let outcomes = [
            outcome("r1", 100.0, 102.0, 110.0, Some(10.0), 4, "SUCCESS"),
            outcome("r2", 101.0, 105.0, 260.0, Some(10.0), 6, "SUCCESS"),
            // Failed partial output contributes to the attempted-run window and
            // existing latency counters, but not successful throughput or TPOT.
            outcome("r3", 102.0, 103.0, 20.0, None, 100, "FAILED"),
        ];
        let mut summary = CommonReplaySummary::default();
        let mut measurements = ReplayMeasurements::default();
        for outcome in &outcomes {
            summary.add(10, outcome);
            measurements.add(outcome);
        }

        summary.finalize(&mut measurements);

        assert_eq!(summary.success_steps, 2);
        assert_close(summary.run_duration_ms, 5_000.0);
        assert_close(summary.request_throughput_per_s, 0.4);
        assert_close(summary.output_token_throughput_per_s, 2.0);
        assert_eq!(summary.tpot_measured_steps, 2);
        assert_close(summary.tpot_ms_avg, 41.66666666666667);
        assert_close(summary.tpot_ms_p50, 41.66666666666667);
        assert_close(summary.tpot_ms_p90, 48.333333333333336);
        assert_close(summary.tpot_ms_max, 50.0);

        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["run_duration_ms"], 5_000.0);
        assert_eq!(json["request_throughput_per_s"], 0.4);
        assert_eq!(json["output_token_throughput_per_s"], 2.0);
        assert_eq!(json["tpot_measured_steps"], 2);
    }

    #[test]
    fn tpot_is_unavailable_without_two_output_tokens_and_ttft() {
        let outcomes = [
            outcome("r1", 10.0, 11.0, 20.0, Some(10.0), 1, "SUCCESS"),
            outcome("r2", 10.0, 12.0, 20.0, None, 4, "SUCCESS"),
        ];
        let mut summary = CommonReplaySummary::default();
        let mut measurements = ReplayMeasurements::default();
        for outcome in &outcomes {
            summary.add(4, outcome);
            measurements.add(outcome);
        }

        summary.finalize(&mut measurements);

        assert_eq!(summary.tpot_measured_steps, 0);
        assert!(summary.tpot_ms_avg.is_none());
        assert!(summary.tpot_ms_p50.is_none());
        assert!(summary.tpot_ms_p90.is_none());
        assert!(summary.tpot_ms_max.is_none());
    }

    #[test]
    fn canonical_metrics_use_token_events_and_retain_completion_audit() {
        let mut measured_outcome = outcome("r1", 100.0, 101.0, 110.0, Some(10.0), 4, "SUCCESS");
        measured_outcome.first_token_id_ms = Some(12.0);
        measured_outcome.token_delivery_tpot_ms = Some(3.0);

        let mut summary = CommonReplaySummary::default();
        let mut measurements = ReplayMeasurements::default();
        summary.add(4, &measured_outcome);
        measurements.add(&measured_outcome);
        summary.finalize(&mut measurements);

        assert_close(summary.ttft_ms_avg, 12.0);
        assert_eq!(summary.ttft_token_id_steps, 1);
        assert_eq!(summary.ttft_text_fallback_steps, 0);
        assert_close(summary.tpot_ms_avg, 3.0);
        assert_close(summary.completion_amortized_tpot_ms_avg, 100.0 / 3.0);
    }

    #[test]
    fn token_only_outcome_keeps_completion_amortized_audit() {
        let mut measured_outcome = outcome("r1", 100.0, 101.0, 110.0, None, 4, "SUCCESS");
        measured_outcome.first_token_id_ms = Some(10.0);
        measured_outcome.token_delivery_tpot_ms = Some(3.0);

        let mut summary = CommonReplaySummary::default();
        let mut measurements = ReplayMeasurements::default();
        summary.add(4, &measured_outcome);
        measurements.add(&measured_outcome);
        summary.finalize(&mut measurements);

        assert_close(summary.completion_amortized_tpot_ms_avg, 100.0 / 3.0);
    }

    #[test]
    fn prefix_summary_uses_the_derived_prefix_not_the_traces_own() {
        let source = SessionRoundSource {
            session_id: "session".to_string(),
            round_idx: 1,
            prefix_len: 10,
            input_len: 90,
            target_prompt_len: 100,
            prompt_len: 100,
            derived_prefix_len: 80,
            derived_append_len: 20,
            prefix_shortfall_tokens: 0,
            planned_prefix_hit_rate: Some(0.8),
            output_len_target: 4,
            tool_wait_after_ms: 0.0,
            arrival_time_ms: 0.0,
            scheduling: Default::default(),
        };
        let mut measured_outcome = outcome("r1", 0.0, 1.0, 10.0, Some(1.0), 4, "SUCCESS");
        measured_outcome.server_usage = Some(crate::record::ServerUsageLog {
            prompt_tokens: Some(100),
            completion_tokens: Some(4),
            total_tokens: Some(104),
            cached_prompt_tokens: Some(80),
            uncached_prompt_tokens: Some(20),
            prefix_hit_rate: Some(0.8),
        });

        let mut summary = PrefixCacheSummary::default();
        summary.add(&source, &measured_outcome);
        summary.finalize();

        assert_close(summary.planned_prefix_hit_rate, 0.8);
        assert_close(
            summary.planned_prefix_hit_rate_for_measured_cache_steps,
            0.8,
        );
        assert_close(summary.server_prefix_hit_rate_delta, 0.0);
    }
}
