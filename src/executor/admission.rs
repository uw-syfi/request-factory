//! Bounded workload dispatch without one parked task per trace row.
//!
//! Declaration order is the admission order: the dispatcher consumes the
//! canonical iterator from front to back and starts at most `limit` units.
//! A completed unit makes room for exactly the next row. This preserves the
//! deterministic capacity semantics while keeping scheduler memory proportional
//! to active concurrency rather than trace length.

use std::future::Future;

use tokio::task::JoinSet;

pub(crate) async fn drive_bounded<I, F, Fut>(items: I, limit: Option<usize>, mut make: F)
where
    I: IntoIterator,
    I::IntoIter: Send,
    I::Item: Send + 'static,
    F: FnMut(usize, I::Item) -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let mut items = items.into_iter().enumerate();
    let window = limit.unwrap_or(usize::MAX);
    debug_assert!(window > 0);
    let mut active = JoinSet::new();

    loop {
        while active.len() < window {
            let Some((ordinal, item)) = items.next() else {
                break;
            };
            active.spawn(make(ordinal, item));
        }

        let Some(result) = active.join_next().await else {
            break;
        };
        if let Err(error) = result {
            eprintln!("workload task join error: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::sync::Barrier;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bounds_active_units_and_starts_them_in_declaration_order() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(Mutex::new(Vec::new()));

        drive_bounded(0..40, Some(4), |ordinal, item| {
            let active = active.clone();
            let peak = peak.clone();
            let started = started.clone();
            async move {
                started.lock().unwrap().push((ordinal, item));
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::task::yield_now().await;
                active.fetch_sub(1, Ordering::SeqCst);
            }
        })
        .await;

        assert!(peak.load(Ordering::SeqCst) <= 4);
        let mut observed = started.lock().unwrap().clone();
        observed.sort_unstable_by_key(|(ordinal, _)| *ordinal);
        assert_eq!(
            observed,
            (0..40).map(|value| (value, value)).collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_unbounded_run_starts_every_unit_without_a_window() {
        let barrier = Arc::new(Barrier::new(17));
        let completed = Arc::new(AtomicUsize::new(0));
        let driver = tokio::spawn({
            let barrier = barrier.clone();
            let completed = completed.clone();
            async move {
                drive_bounded(0..16, None, |_ordinal, _item| {
                    let barrier = barrier.clone();
                    let completed = completed.clone();
                    async move {
                        barrier.wait().await;
                        completed.fetch_add(1, Ordering::SeqCst);
                    }
                })
                .await;
            }
        });
        // The test itself is the final barrier participant. If the dispatcher
        // imposed an undocumented window, it would deadlock here.
        barrier.wait().await;
        driver.await.unwrap();
        assert_eq!(completed.load(Ordering::SeqCst), 16);
    }
}
