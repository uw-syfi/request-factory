//! Narrow production entry points used only by the component benchmarks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::executor::drive_bounded;

pub async fn dispatch_production(items: usize, concurrency: usize, yield_once: bool) -> usize {
    let completed = Arc::new(AtomicUsize::new(0));
    let dispatch_completed = completed.clone();
    drive_bounded(0..items, Some(concurrency), move |_ordinal, item| {
        let completed = dispatch_completed.clone();
        async move {
            if yield_once {
                tokio::task::yield_now().await;
            }
            completed.fetch_add(item.wrapping_add(1), Ordering::Relaxed);
        }
    })
    .await;
    completed.load(Ordering::Relaxed)
}

pub async fn dispatch_persistent_candidate(
    items: usize,
    concurrency: usize,
    yield_once: bool,
) -> usize {
    let completed = Arc::new(AtomicUsize::new(0));
    let next = Arc::new(Mutex::new((0..items).enumerate()));
    let mut active = tokio::task::JoinSet::new();
    for _ in 0..concurrency.min(items) {
        let next = next.clone();
        let completed = completed.clone();
        active.spawn(async move {
            loop {
                let Some((_ordinal, item)) = next.lock().unwrap().next() else {
                    break;
                };
                if yield_once {
                    tokio::task::yield_now().await;
                }
                // Match the production benchmark's per-item shared-state
                // clone/drop so this comparison isolates dispatcher shape.
                let item_completed = completed.clone();
                item_completed.fetch_add(item.wrapping_add(1), Ordering::Relaxed);
            }
        });
    }
    while let Some(result) = active.join_next().await {
        result.expect("synthetic dispatch task failed");
    }
    completed.load(Ordering::Relaxed)
}

pub fn serialize_vllm_request(prompt_ids: &[u32]) -> Vec<u8> {
    crate::backend::bench_serialize_vllm_request(prompt_ids)
}

pub fn parse_vllm_event(data: &[u8]) -> usize {
    crate::backend::bench_parse_vllm_event(data)
}

pub fn serialize_chat_request(dialect: &str, text: &str, images: usize) -> Vec<u8> {
    crate::backend::bench_serialize_chat_request(dialect, text, images)
}
