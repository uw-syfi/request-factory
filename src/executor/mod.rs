mod independent;
mod session;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;

use crate::backend::GenerationClient;
use crate::cli::Args;

pub(crate) use independent::run_independent_request;
pub(crate) use session::run_session;

/// Shared, immutable-per-run state handed to every workload-unit executor.
pub(crate) struct AppState {
    pub(crate) args: Args,
    pub(crate) client: Arc<GenerationClient>,
    pub(crate) token_pool: Arc<Vec<u32>>,
    pub(crate) stats: Arc<Stats>,
    pub(crate) run_start: Instant,
    pub(crate) concurrency_semaphore: Option<Arc<Semaphore>>,
}

/// Lock-free progress counters shared with the status reporter.
#[derive(Default)]
pub(crate) struct Stats {
    submitted: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    finished_units: AtomicUsize,
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

        eprintln!(
            "{} {}/{} | steps {}/{} completed={} submitted={} active={} failed={} | elapsed={:.1}s",
            unit_label,
            finished_units,
            total_units,
            finished_steps,
            total_steps,
            completed,
            submitted,
            active,
            failed,
            start.elapsed().as_secs_f64(),
        );

        if finished_units >= total_units {
            break;
        }
    }
}
