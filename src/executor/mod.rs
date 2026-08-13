mod admission;
mod independent;
mod session;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;

use crate::backend::GenerationClient;
use crate::cli::{Args, ArrivalMode};
use crate::util::reaches_context_limit;

pub(crate) use admission::AdmissionOrder;
pub(crate) use independent::run_independent_request;
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

/// Shared, immutable-per-run state handed to every workload-unit executor.
pub(crate) struct AppState {
    pub(crate) policy: RunPolicy,
    pub(crate) client: Arc<GenerationClient>,
    pub(crate) token_pool: Arc<Vec<u32>>,
    pub(crate) stats: Arc<Stats>,
    pub(crate) run_start: Instant,
    pub(crate) concurrency_semaphore: Option<Arc<Semaphore>>,
    /// Present only alongside `concurrency_semaphore`: it exists to make the
    /// contended case deterministic, and there is no contention without a cap.
    pub(crate) admission_order: Option<Arc<AdmissionOrder>>,
}

impl AppState {
    /// Take one capacity slot, in declaration order.
    ///
    /// The returned permit owns the slot for as long as it is alive — for a
    /// session that is every round *and* every tool wait in between, because the
    /// binding lives in `run_session`'s scope. Callers must therefore bind it,
    /// not discard it.
    ///
    /// Returns `None` when the run is uncapped, which is also the only case
    /// where units may enter concurrently with each other.
    pub(crate) async fn acquire_capacity_slot(
        &self,
        ordinal: usize,
    ) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let semaphore = self.concurrency_semaphore.as_ref()?;
        // The turn guard is dropped the moment the permit is in hand, so the
        // next ordinal starts contending only once this one holds a slot. That
        // is exactly VibeSim's cursor rule: row k+1 never enters before row k.
        let _turn = match &self.admission_order {
            Some(order) => Some(order.wait_for_turn(ordinal).await),
            None => None,
        };
        semaphore.clone().acquire_owned().await.ok()
    }
}

/// Lock-free progress counters shared with the status reporter.
#[derive(Default)]
pub(crate) struct Stats {
    submitted: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    finished_units: AtomicUsize,
    runtime_global_queue_depth_peak: AtomicUsize,
}

impl Stats {
    pub(crate) fn record_submit(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_result(&self, success: bool) {
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

/// Periodic stderr progress reporter; exits once all workload units are finished.
pub(crate) async fn status_task(
    stats: Arc<Stats>,
    total_units: usize,
    total_steps: usize,
    unit_label: &'static str,
    start: Instant,
) {
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
            start.elapsed().as_secs_f64(),
        );

        if finished_units >= total_units {
            break;
        }
    }
}
