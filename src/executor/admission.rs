//! Deterministic ordering of capacity-slot admission.
//!
//! Every workload unit runs as its own task, so without help the order in which
//! units reach the concurrency semaphore is whatever the tokio scheduler picked
//! that run. That is invisible while the run is uncapped — nobody queues — but
//! under `--max-concurrency` it decides *which* unit gets the freed slot, and
//! two runs of the same trace can then admit different units.
//!
//! VibeSim has no such freedom: its release cursor walks the canonical rows in
//! order and never releases row k+1 before row k. This gate gives the measured
//! runner the same rule, which is what makes an admission sequence comparable
//! between the two systems at all.
//!
//! Ordinals are the unit's canonical position, and canonical order is
//! nondecreasing by arrival, so waiting for your turn never means waiting for a
//! unit that arrives later than you.

use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Notify;

/// Hands out the right to contend for a capacity slot, one ordinal at a time.
///
/// One `Notify` per unit rather than one shared one: a broadcast would wake
/// every queued unit on every advance, which under a saturated arrival mode is
/// quadratic wake churn inside the very tool that measures latency.
#[derive(Debug)]
pub(crate) struct AdmissionOrder {
    next: AtomicUsize,
    turn: Vec<Notify>,
}

impl AdmissionOrder {
    pub(crate) fn new(unit_count: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            turn: (0..unit_count).map(|_| Notify::new()).collect(),
        }
    }

    /// Wait until `ordinal` is the one being admitted.
    ///
    /// Dropping the returned guard passes the turn on. It advances on drop
    /// rather than on an explicit call so that a unit which panics or returns
    /// early cannot strand every later unit behind it.
    pub(crate) async fn wait_for_turn(&self, ordinal: usize) -> TurnGuard<'_> {
        if let Some(turn) = self.turn.get(ordinal) {
            loop {
                // Registered before the load, so an `advance` racing us here is
                // still observed by the `await` below rather than lost.
                let notified = turn.notified();
                if self.next.load(Ordering::Acquire) >= ordinal {
                    break;
                }
                notified.await;
            }
        }
        TurnGuard { order: self }
    }

    fn advance(&self) {
        let next = self.next.fetch_add(1, Ordering::Release) + 1;
        if let Some(turn) = self.turn.get(next) {
            turn.notify_one();
        }
    }
}

/// Holds the admission turn; passes it on when dropped.
pub(crate) struct TurnGuard<'a> {
    order: &'a AdmissionOrder,
}

impl Drop for TurnGuard<'_> {
    fn drop(&mut self) {
        self.order.advance();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Tasks spawned in reverse order still take their turns in ordinal order.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn turns_are_taken_in_ordinal_order_regardless_of_spawn_order() {
        let order = Arc::new(AdmissionOrder::new(16));
        let admitted = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();

        for ordinal in (0..16).rev() {
            let order = order.clone();
            let admitted = admitted.clone();
            tasks.push(tokio::spawn(async move {
                let _turn = order.wait_for_turn(ordinal).await;
                admitted.lock().await.push(ordinal);
                // Hold the turn across an await point, the way a real unit holds
                // it across the semaphore acquisition.
                tokio::task::yield_now().await;
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        assert_eq!(*admitted.lock().await, (0..16).collect::<Vec<_>>());
    }

    /// A unit that dies without ever finishing must not strand the rest.
    #[tokio::test]
    async fn a_panicking_unit_still_passes_its_turn_on() {
        let order = Arc::new(AdmissionOrder::new(2));
        let panicking = tokio::spawn({
            let order = order.clone();
            async move {
                let _turn = order.wait_for_turn(0).await;
                panic!("unit died mid-admission");
            }
        });
        assert!(panicking.await.is_err());

        let follower = tokio::spawn({
            let order = order.clone();
            async move {
                let _turn = order.wait_for_turn(1).await;
            }
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), follower)
            .await
            .expect("ordinal 1 was stranded behind the dead unit")
            .unwrap();
    }
}
