mod admission;
mod independent;
mod multimodal;
mod session;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::backend::GenerationClient;
use crate::cli::Args;
use crate::release::ArrivalMode;
use crate::util::reaches_context_limit;

pub(crate) use admission::drive_bounded;
pub(crate) use independent::run_independent_request;
pub(crate) use multimodal::{
    prepare_multimodal_requests, representative_media, run_multimodal_request, MultimodalState,
};
pub(crate) use session::run_session;

/// The four run-level decisions a workload-unit executor actually makes.
///
/// `Args` has twenty fields and the executors read these four. Handing over the
/// whole struct made every executor look like it might depend on the endpoint
/// URL or the log path, and left no way to tell from a type what a change to
/// `Args` could reach.
#[derive(Clone, Copy)]
pub(crate) struct RunPolicy {
    /// Which axis supplies a unit's start time.
    pub(crate) arrival_mode: ArrivalMode,
    /// Declared model context limit, when there is one.
    max_model_len: Option<usize>,
    /// Whether to skip a request that would reach that limit rather than send it.
    skip_when_reaching_limit: bool,
    /// Whether a session stops after its first failed round.
    pub(crate) stop_session_on_error: bool,
}

impl RunPolicy {
    pub(crate) fn from_args(args: &Args) -> Self {
        Self {
            arrival_mode: args.arrival_mode,
            max_model_len: args.max_model_len,
            skip_when_reaching_limit: args.skip_when_reaching_limit,
            stop_session_on_error: args.stop_session_on_error,
        }
    }

    /// The limit to report on a skip. `None` means none was declared, which is
    /// also the only case in which [`Self::skips_at_context_limit`] is never true.
    pub(crate) fn max_model_len(&self) -> Option<usize> {
        self.max_model_len
    }

    /// Whether this request reaches the declared context limit and must be
    /// skipped rather than sent. Both executors ask the same question, so they
    /// ask it in one place.
    pub(crate) fn skips_at_context_limit(
        &self,
        prompt_len: usize,
        output_len_target: usize,
    ) -> bool {
        self.skip_when_reaching_limit
            && self
                .max_model_len
                .is_some_and(|limit| reaches_context_limit(prompt_len, output_len_target, limit))
    }
}

/// Modality-independent scheduling and measurement state.
///
/// New request families compose this with their own client and input store;
/// they do not inherit text generation's tokenizer or synthetic token pool.
pub(crate) struct CommonState {
    pub(crate) policy: RunPolicy,
    pub(crate) stats: Arc<Stats>,
    pub(crate) run_start: Instant,
}

/// State used only by the existing token-id text replay executors.
pub(crate) struct TextGenerationState {
    pub(crate) common: CommonState,
    pub(crate) client: Arc<GenerationClient>,
    pub(crate) token_pool: Arc<Vec<u32>>,
}

/// Lock-free progress counters shared with the status reporter.
#[derive(Default)]
pub(crate) struct Stats {
    submitted: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    output_tokens: AtomicUsize,
    finished_units: AtomicUsize,
    runtime_global_queue_depth_peak: AtomicUsize,
}

impl Stats {
    pub(crate) fn record_submit(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// `output_tokens` is the round's `output_len_actual`, counted whether or
    /// not the round succeeded, so the progress line reports tokens generated.
    pub(crate) fn record_result(&self, success: bool, output_tokens: usize) {
        self.output_tokens
            .fetch_add(output_tokens, Ordering::Relaxed);
        if success {
            self.completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_unit_done(&self) {
        self.finished_units.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn runtime_global_queue_depth_peak(&self) -> usize {
        self.runtime_global_queue_depth_peak.load(Ordering::Relaxed)
    }
}

/// How often the machine-readable `progress |` line is printed.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);
const STATUS_TICK: std::time::Duration = std::time::Duration::from_millis(500);

/// One reading of the run's progress, rendered as the `progress |` stderr line.
///
/// A harness that kills a run on a deadline sees only what was printed so far,
/// and the request log, timeline and summary are written at exit. This line is
/// how that harness learns how far the run got.
#[derive(Debug, PartialEq)]
struct ProgressLine {
    elapsed_s: f64,
    finished_rounds: usize,
    total_rounds: usize,
    unit_label: &'static str,
    finished_units: usize,
    total_units: usize,
    output_tokens: usize,
}

impl ProgressLine {
    fn render(&self) -> String {
        format!(
            "progress | elapsed_s={:.1} rounds_done={}/{} {}_done={}/{} output_tokens={}",
            self.elapsed_s,
            self.finished_rounds,
            self.total_rounds,
            self.unit_label,
            self.finished_units,
            self.total_units,
            self.output_tokens,
        )
    }
}

/// Periodic stderr progress reporter; exits once all workload units are finished.
///
/// Prints the human status line every tick and the `progress |` line every
/// [`PROGRESS_INTERVAL`] and once more when the run ends.
pub(crate) async fn status_task(
    stats: Arc<Stats>,
    total_units: usize,
    total_steps: usize,
    unit_label: &'static str,
    start: Instant,
) {
    let mut last_progress = Duration::ZERO;
    loop {
        tokio::time::sleep(STATUS_TICK).await;
        let submitted = stats.submitted.load(Ordering::Relaxed);
        let completed = stats.completed.load(Ordering::Relaxed);
        let failed = stats.failed.load(Ordering::Relaxed);
        let finished_units = stats.finished_units.load(Ordering::Relaxed);
        let active = submitted.saturating_sub(completed + failed);
        let finished_steps = completed + failed;
        let runtime_global_queue_depth = tokio::runtime::Handle::current()
            .metrics()
            .global_queue_depth();
        stats
            .runtime_global_queue_depth_peak
            .fetch_max(runtime_global_queue_depth, Ordering::Relaxed);

        let elapsed = start.elapsed();
        eprintln!(
            "{} {}/{} | steps {}/{} completed={} submitted={} active={} failed={} runtime_global_queue_depth={} | elapsed={:.1}s",
            unit_label,
            finished_units,
            total_units,
            finished_steps,
            total_steps,
            completed,
            submitted,
            active,
            failed,
            runtime_global_queue_depth,
            elapsed.as_secs_f64(),
        );

        let done = finished_units >= total_units;
        if done || elapsed.saturating_sub(last_progress) >= PROGRESS_INTERVAL {
            last_progress = elapsed;
            // `eprintln!` writes to unbuffered stderr, so the line is visible
            // to a parent process as soon as it is printed.
            eprintln!(
                "{}",
                ProgressLine {
                    elapsed_s: elapsed.as_secs_f64(),
                    finished_rounds: finished_steps,
                    total_rounds: total_steps,
                    unit_label,
                    finished_units,
                    total_units,
                    output_tokens: stats.output_tokens.load(Ordering::Relaxed),
                }
                .render()
            );
        }

        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_line_names_rounds_units_and_output_tokens() {
        let line = ProgressLine {
            elapsed_s: 10.04,
            finished_rounds: 30,
            total_rounds: 72,
            unit_label: "sessions",
            finished_units: 4,
            total_units: 12,
            output_tokens: 5120,
        };
        assert_eq!(
            line.render(),
            "progress | elapsed_s=10.0 rounds_done=30/72 sessions_done=4/12 output_tokens=5120"
        );
    }

    #[test]
    fn stats_sum_output_tokens_across_successful_and_failed_rounds() {
        let stats = Stats::default();
        stats.record_result(true, 100);
        stats.record_result(false, 7);
        assert_eq!(stats.output_tokens.load(Ordering::Relaxed), 107);
        assert_eq!(stats.completed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.failed.load(Ordering::Relaxed), 1);
    }
}
