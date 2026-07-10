use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use tokio::sync::mpsc;

use crate::record::{GenerationOutcome, SessionRoundSource, SourceRecord, StepLog};
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
    context_overflow_steps: usize,
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
            (Self::IndependentRequests { common }, SourceRecord::VibeSimRequest(source)) => {
                common.add(source.output_len_target, &record.outcome)
            }
            _ => unreachable!("log source must match the selected replay workload"),
        }
    }

    fn finalize(&mut self, total_durations: &mut [f64], ttfts: &mut [f64]) {
        let common = match self {
            Self::Sessions { common, .. } | Self::IndependentRequests { common } => common,
        };
        common.finalize(total_durations, ttfts);

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
    }

    fn finalize(&mut self, total_durations: &mut [f64], ttfts: &mut [f64]) {
        if !total_durations.is_empty() {
            total_durations.sort_by(|a, b| a.total_cmp(b));
            self.total_duration_ms_avg = self.total_duration_ms_sum / total_durations.len() as f64;
            self.total_duration_ms_p50 = percentile_sorted(total_durations, 0.50);
            self.total_duration_ms_p90 = percentile_sorted(total_durations, 0.90);
            self.total_duration_ms_max = *total_durations.last().unwrap_or(&0.0);
        }

        if !ttfts.is_empty() {
            ttfts.sort_by(|a, b| a.total_cmp(b));
            let sum: f64 = ttfts.iter().sum();
            self.ttft_ms_avg = Some(sum / ttfts.len() as f64);
            self.ttft_ms_p50 = Some(percentile_sorted(ttfts, 0.50));
            self.ttft_ms_p90 = Some(percentile_sorted(ttfts, 0.90));
            self.ttft_ms_max = ttfts.last().copied();
        }
    }
}

impl PrefixCacheSummary {
    fn add(&mut self, source: &SessionRoundSource, outcome: &GenerationOutcome) {
        self.planned_prefix_tokens += source.prefix_len;
        self.planned_prompt_tokens += source.prompt_len;
        if let Some(usage) = &outcome.server_usage {
            if let (Some(cached), Some(prompt)) = (usage.cached_prompt_tokens, usage.prompt_tokens)
            {
                self.measured_cache_steps += 1;
                self.measured_server_cached_prompt_tokens += cached;
                self.measured_server_prompt_tokens += prompt;
                self.planned_prefix_tokens_for_measured_cache_steps += source.prefix_len;
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
pub(crate) struct RunSummary {
    pub(crate) workload: WorkloadSummary,
    pub(crate) replay: ReplaySummary,
}

/// Drain logged requests to JSONL on disk and fold them into a typed replay summary.
pub(crate) async fn write_logs(
    path: String,
    mut rx: mpsc::Receiver<StepLog>,
    mut summary: ReplaySummary,
) -> ReplaySummary {
    let file = File::create(&path).expect("failed to create log file");
    let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
    let mut total_durations = Vec::new();
    let mut ttfts = Vec::new();

    while let Some(record) = rx.recv().await {
        summary.add(&record);
        total_durations.push(record.outcome.total_duration_ms);
        if let Some(ttft) = record.outcome.first_token_ms {
            ttfts.push(ttft);
        }
        log_server_prefix_hit_rate(&record);
        if let Ok(json) = serde_json::to_string(&record) {
            let _ = writeln!(writer, "{json}");
            let _ = writer.flush();
        }
    }
    let _ = writer.flush();
    summary.finalize(&mut total_durations, &mut ttfts);
    log_server_prefix_hit_rate_summary(&summary);
    summary
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
    summary: RunSummary,
) -> Result<()> {
    let Some(path) = summary_path else {
        return Ok(());
    };
    let file = File::create(path).with_context(|| format!("failed to create summary: {path}"))?;
    serde_json::to_writer_pretty(file, &summary)
        .with_context(|| format!("failed to write summary: {path}"))?;
    Ok(())
}
