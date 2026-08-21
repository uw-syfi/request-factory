use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use req_frontend::bench_support::{parse_vllm_event, serialize_vllm_request};

fn vllm_wire(c: &mut Criterion) {
    let mut serialization = c.benchmark_group("vllm_request_serialization");
    for &prompt_len in &[1usize, 128, 4_096] {
        let prompt = (0..prompt_len as u32).collect::<Vec<_>>();
        serialization.throughput(Throughput::Elements(prompt_len as u64));
        serialization.bench_with_input(
            BenchmarkId::from_parameter(prompt_len),
            &prompt,
            |b, prompt| b.iter(|| black_box(serialize_vllm_request(black_box(prompt)))),
        );
    }
    serialization.finish();

    let events = [
        (
            "one_token",
            br#"{"choices":[{"index":0,"finish_reason":null,"token_ids":[101]}],"usage":null}"#.as_slice(),
        ),
        (
            "thirty_two_tokens",
            br#"{"choices":[{"index":0,"finish_reason":"length","token_ids":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]}],"usage":null}"#.as_slice(),
        ),
        (
            "usage",
            br#"{"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":32,"total_tokens":160,"prompt_tokens_details":{"cached_tokens":96}}}"#.as_slice(),
        ),
    ];
    let mut parsing = c.benchmark_group("vllm_event_parsing");
    for (name, event) in events {
        parsing.throughput(Throughput::Bytes(event.len() as u64));
        parsing.bench_with_input(BenchmarkId::from_parameter(name), event, |b, event| {
            b.iter(|| black_box(parse_vllm_event(black_box(event))))
        });
    }
    parsing.finish();
}

criterion_group!(benches, vllm_wire);
criterion_main!(benches);
