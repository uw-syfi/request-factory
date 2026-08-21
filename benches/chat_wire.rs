//! Guards the chat body against drifting back to a JSON DOM.
//!
//! This body's field names come from the dialect table, which is the reason it
//! is hand-serialized rather than derived. The pressure to "just build a
//! `Value`" is therefore permanent, and a `Value` builder measured 4.4x slower
//! on text and 4.8x slower with four images.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use req_frontend::bench_support::serialize_chat_request;

fn chat_wire(c: &mut Criterion) {
    let text = "describe this image in one sentence";
    let mut group = c.benchmark_group("chat_request_serialization");
    for images in [0usize, 1, 4] {
        group.bench_with_input(BenchmarkId::from_parameter(images), &images, |b, &n| {
            b.iter(|| black_box(serialize_chat_request("vllm", black_box(text), n)))
        });
    }
    group.finish();

    // The encoding that moves media off the turn takes a different path through
    // the same serializer, so it gets its own guardrail.
    let mut lists = c.benchmark_group("chat_request_serialization_top_level_lists");
    lists.bench_function("one_image", |b| {
        b.iter(|| black_box(serialize_chat_request("sglang-omni", black_box(text), 1)))
    });
    lists.finish();
}

criterion_group!(benches, chat_wire);
criterion_main!(benches);
