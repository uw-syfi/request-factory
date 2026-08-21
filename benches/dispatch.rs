use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use req_frontend::bench_support::{dispatch_persistent_candidate, dispatch_production};

fn dispatch(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("dispatch");
    group.sample_size(20);

    for &items in &[10_000usize, 100_000] {
        for &concurrency in &[8usize, 256] {
            for &(shape, yield_once) in &[("ready", false), ("yield_once", true)] {
                group.throughput(Throughput::Elements(items as u64));
                let parameter = format!("{shape}/n={items}/c={concurrency}");
                group.bench_with_input(
                    BenchmarkId::new("production_spawn_per_item", &parameter),
                    &(items, concurrency, yield_once),
                    |b, &(items, concurrency, yield_once)| {
                        b.to_async(&runtime).iter(|| async {
                            black_box(dispatch_production(items, concurrency, yield_once).await)
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("persistent_worker_candidate", &parameter),
                    &(items, concurrency, yield_once),
                    |b, &(items, concurrency, yield_once)| {
                        b.to_async(&runtime).iter(|| async {
                            black_box(
                                dispatch_persistent_candidate(items, concurrency, yield_once).await,
                            )
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, dispatch);
criterion_main!(benches);
