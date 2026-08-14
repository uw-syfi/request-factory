//! Does this client measure what it says it measures?
//!
//! Every number this repo emits is a claim about a server, and every one of them
//! is only as good as the client that took it. A TTFT that silently includes the
//! client's own scheduling lag, a release that went out 40 ms after the trace
//! said, a timeline whose recording slowed the very submissions it was timing —
//! none of those show up as errors. They show up as plausible measurements.
//!
//! So they get checked, against a server whose timing is fixed by construction.
//! `tools/stub_server.py` is told exactly how long to wait before the first
//! chunk and between chunks; those two numbers are the ground truth, and every
//! timing claim here is the client's measurement compared against them.
//!
//! This exists because the alternative was prose. The release-lag and
//! timeline-overhead results used to live in README tables, computed once by an
//! analysis that was not kept, against a trace that no longer exists. A claim
//! nobody can re-run is a claim nobody can lose confidence in.
//!
//!     cargo run --release --bin selfcheck -- --tokenizer <any tokenizer.json>
//!
//! Exits non-zero if any check fails, so it can gate a change rather than only
//! inform one.

mod check;
mod fixture;
mod record;
mod stub;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use req_frontend::{run_once_reusing, Args, CorpusCache};
use serde::Serialize;

use check::{percentile, Bound, Check};
use fixture::{Fixtures, Shape};
use record::{Record, Source};
use stub::{Stub, Timing};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Check this client's own fidelity against a server with known timing"
)]
struct SelfcheckArgs {
    /// Any `tokenizer.json`, or a directory containing one.
    ///
    /// Required because the client tokenizes its synthetic corpus, and not
    /// checked against a particular model: none of these claims depend on which
    /// tokenizer produced the ids, only that the same ids went out.
    #[arg(long)]
    tokenizer: String,

    /// Where fixtures, run logs and `selfcheck.json` go.
    #[arg(long, default_value = "out/selfcheck")]
    out: PathBuf,

    /// Paired timeline-on/timeline-off runs used to measure the timeline's cost
    /// on submission. More pairs, tighter bound on the difference.
    #[arg(long, default_value_t = 2)]
    pairs: usize,

    /// Loopback port owned by the harness while it runs.
    #[arg(long, default_value_t = 8271)]
    port: u16,
}

#[derive(Debug, Serialize)]
struct Report {
    /// What the stub was told to do — the ground truth every timing claim is
    /// stated against.
    server: ServerRecord,
    step_log_schema_version: u32,
    checks: Vec<Check>,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Serialize)]
struct ServerRecord {
    prefill_delay_ms: f64,
    chunk_delay_ms: f64,
    capacity: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = SelfcheckArgs::parse();
    std::fs::create_dir_all(&arguments.out)
        .with_context(|| format!("failed to create {}", arguments.out.display()))?;
    let fixtures = fixture::build(&arguments.out.join("fixtures"))?;

    // One stub for every check, so no claim can be an artifact of a server
    // restart. Prefill and chunk delays are deliberately different numbers: with
    // one knob, a client that confused TTFT with TPOT would still agree.
    let timing = Timing {
        prefill_delay_ms: 50.0,
        chunk_delay_ms: 2.0,
        capacity: 0,
    };
    let server = Stub::start(arguments.port, timing)?;
    eprintln!(
        "selfcheck | stub on {} | prefill={}ms chunk={}ms",
        server.base_url(),
        timing.prefill_delay_ms,
        timing.chunk_delay_ms,
    );

    // One corpus, tokenized once, across every run below — the same reuse a
    // sweep gets, and the reason this harness takes seconds rather than minutes.
    let mut corpus = CorpusCache::default();
    let mut checks = Vec::new();

    checks.extend(release_checks(&arguments, &fixtures, &server, &mut corpus).await?);
    checks.extend(timeline_checks(&arguments, &fixtures, &server, &mut corpus).await?);

    // Prefix fidelity must start from a cache whose contents the timing fixture
    // alone determines. Earlier independent runs intentionally use the same
    // token pool and could otherwise pre-warm a later prompt, turning a wrong
    // prefix assertion into a plausible full-prompt hit.
    drop(server);
    let timing_server = Stub::start(arguments.port, timing)?;
    checks.extend(timing_checks(&arguments, &fixtures, &timing_server, timing, &mut corpus).await?);

    let failed = checks.iter().filter(|check| !check.passed).count();
    let report = Report {
        server: ServerRecord {
            prefill_delay_ms: timing.prefill_delay_ms,
            chunk_delay_ms: timing.chunk_delay_ms,
            capacity: timing.capacity,
        },
        step_log_schema_version: record::EXPECTED_SCHEMA_VERSION,
        passed: checks.len() - failed,
        failed,
        checks,
    };

    eprintln!();
    for line in report.checks.iter().map(Check::report_line) {
        eprintln!("{line}");
    }
    let report_path = arguments.out.join("selfcheck.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    eprintln!(
        "\nselfcheck | {} passed, {} failed | {}",
        report.passed,
        report.failed,
        report_path.display()
    );

    if report.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Did the client release each request when the trace said it should?
///
/// The one check that is about the load generator rather than the server. Every
/// latency in every other report is measured from a submission, so a client that
/// submits late does not report an error — it reports a different workload.
async fn release_checks(
    arguments: &SelfcheckArgs,
    fixtures: &Fixtures,
    server: &Stub,
    corpus: &mut CorpusCache,
) -> Result<Vec<Check>> {
    let shape = fixtures.independent_shape;
    // Offered fast enough that lateness would be visible, slow enough that the
    // server is nowhere near saturation: a queued request is late for a reason
    // that is not the client's.
    let rate = 300.0;
    let directory = arguments.out.join("runs/release");
    let records = replay(
        arguments,
        fixtures,
        server,
        corpus,
        RunSpec {
            trace: fixtures.independent.display().to_string(),
            format: "text-generation-independent",
            rate: Some(rate),
            directory: directory.clone(),
            timeline: true,
        },
    )
    .await?;

    let mut lags: Vec<f64> = records
        .iter()
        .filter_map(|record| match &record.source {
            Source::IndependentRequest(source) => Some(source.arrival_release_lag_ms),
            Source::SessionRound(_) => None,
        })
        .collect();
    let count = lags.len();
    let p50 = percentile(&mut lags, 0.50).unwrap_or_default();
    let p99 = percentile(&mut lags, 0.99).unwrap_or_default();

    // The offered span, from the client's own submission timestamps. `--rate`
    // rescales the file's timeline; this is whether it landed.
    let mut submits: Vec<f64> = records
        .iter()
        .map(|record| record.outcome.submit_timestamp)
        .collect();
    submits.sort_by(|left, right| left.total_cmp(right));
    let observed_span_s = submits.last().copied().unwrap_or_default() - submits[0];
    let expected_span_s = (shape.units - 1) as f64 / rate;

    Ok(vec![
        Check::new(
            "release_is_on_time",
            "The client submits each request at the instant the trace scheduled, so every \
             latency it reports is measured from the arrival the workload declared.",
            "arrival_release_lag p99",
            p99,
            "ms",
            Bound::AtMost { limit: 5.0 },
            "A few milliseconds is Tokio timer granularity plus wakeup. Ten times that would \
             mean the release loop, not the timer, is the thing being measured.",
        )
        .with_detail(format!(
            "{count} requests offered at {rate}/s | p50 {p50:.4} ms"
        )),
        Check::new(
            "rate_scaling_lands",
            "`--rate` rescales the trace's timeline to the rate asked for, rather than \
             approximately toward it.",
            "submission span",
            observed_span_s,
            "s",
            Bound::Within {
                of: expected_span_s,
                by: expected_span_s * 0.02,
            },
            "Two percent over a four-second span is a tail of late releases the p99 above \
             would already have caught; this is the run-level statement of the same thing.",
        )
        .with_detail(format!(
            "{} units at {rate}/s = {expected_span_s:.4} s expected",
            shape.units
        )),
    ])
}

/// Does recording the per-event timeline slow down submission?
///
/// The binding constraint of the whole timeline design, and the one claim that
/// cannot be argued from the code: the write path looks free, and the first
/// version of it was not. Paired runs, alternating, because machine load drifts
/// and two blocks measured back to back would attribute that drift to the
/// feature.
async fn timeline_checks(
    arguments: &SelfcheckArgs,
    fixtures: &Fixtures,
    server: &Stub,
    corpus: &mut CorpusCache,
) -> Result<Vec<Check>> {
    let mut timeline_enabled_lags: Vec<f64> = Vec::new();
    let mut timeline_disabled_lags: Vec<f64> = Vec::new();
    let mut dropped = 0usize;

    for pair in 0..arguments.pairs {
        for timeline in [true, false] {
            let label = if timeline { "on" } else { "off" };
            let records = replay(
                arguments,
                fixtures,
                server,
                corpus,
                RunSpec {
                    trace: fixtures.independent.display().to_string(),
                    format: "text-generation-independent",
                    rate: Some(300.0),
                    directory: arguments.out.join(format!("runs/timeline_{pair}_{label}")),
                    timeline,
                },
            )
            .await?;
            let lags = records.iter().filter_map(|record| match &record.source {
                Source::IndependentRequest(source) => Some(source.arrival_release_lag_ms),
                Source::SessionRound(_) => None,
            });
            if timeline {
                timeline_enabled_lags.extend(lags);
                let path = arguments
                    .out
                    .join(format!("runs/timeline_{pair}_on/summary.json"));
                dropped += dropped_timelines(&path)?;
            } else {
                timeline_disabled_lags.extend(lags);
            }
        }
    }

    let mut checks = Vec::new();
    for (fraction, name) in [(0.50, "p50"), (0.99, "p99")] {
        let on = percentile(&mut timeline_enabled_lags.clone(), fraction).unwrap_or_default();
        let off = percentile(&mut timeline_disabled_lags.clone(), fraction).unwrap_or_default();
        checks.push(
            Check::new(
                if fraction == 0.50 {
                    "timeline_is_free_at_the_median"
                } else {
                    "timeline_is_free_in_the_tail"
                },
                "Recording every token arrival costs the submission path nothing measurable. \
                 If it did, every latency in a timeline-enabled run would be the client's \
                 own bookkeeping.",
                format!("release lag {name}, on minus off"),
                on - off,
                "ms",
                Bound::Within { of: 0.0, by: 0.5 },
                "Half a millisecond is below the timer granularity the release path already \
                 has, so a real cost would have to exceed the noise this check cannot see \
                 through anyway.",
            )
            .with_detail(format!(
                "{} pair(s), {} on / {} off requests | on {on:.4} ms, off {off:.4} ms",
                arguments.pairs,
                timeline_enabled_lags.len(),
                timeline_disabled_lags.len(),
            )),
        );
    }
    checks.push(
        Check::new(
            "timeline_records_every_request",
            "The timeline is a record of the run, not a sample of it: nothing was dropped to \
             keep submission fast.",
            "dropped request timelines",
            dropped as f64,
            "requests",
            Bound::Exactly { value: 0.0 },
            "The channel is bounded and drops on backpressure by design; a nonzero count here \
             means the harness's own runs hit that path, and no other check's numbers can be \
             read as complete.",
        )
        .with_detail(format!("across {} timeline-enabled runs", arguments.pairs)),
    );
    Ok(checks)
}

/// Do the client's timings equal what the server was told to do?
///
/// Prefill and chunk delays are different numbers on purpose. TTFT must equal
/// the first and TPOT the second; a client that conflated them would match a
/// server where both were the same and fail here.
async fn timing_checks(
    arguments: &SelfcheckArgs,
    fixtures: &Fixtures,
    server: &Stub,
    timing: Timing,
    corpus: &mut CorpusCache,
) -> Result<Vec<Check>> {
    let shape: Shape = fixtures.session_shape;
    let records = replay(
        arguments,
        fixtures,
        server,
        corpus,
        RunSpec {
            trace: fixtures.sessions.display().to_string(),
            format: "text-generation-session-execution-v2",
            rate: Some(40.0),
            directory: arguments.out.join("runs/timing"),
            timeline: true,
        },
    )
    .await?;

    let successes: Vec<&Record> = records
        .iter()
        .filter(|record| record.outcome.succeeded())
        .collect();
    let mut ttfts: Vec<f64> = successes
        .iter()
        .filter_map(|record| record.outcome.ttft_ms())
        .collect();
    let mut tpots: Vec<f64> = successes
        .iter()
        .filter_map(|record| record.outcome.token_delivery_tpot_ms)
        .collect();
    let mut totals: Vec<f64> = successes
        .iter()
        .map(|record| record.outcome.total_duration_ms)
        .collect();

    let ttft = percentile(&mut ttfts, 0.50).unwrap_or_default();
    let tpot = percentile(&mut tpots, 0.50).unwrap_or_default();
    let total = percentile(&mut totals, 0.50).unwrap_or_default();
    let expected_total = timing.expected_total_ms(shape.output_len);

    // Accounting, not timing: what went out and what came back.
    let short_outputs = successes
        .iter()
        .filter(|record| record.outcome.output_len_actual != shape.output_len)
        .count();
    let prompt_mismatches = successes
        .iter()
        .filter(|record| match &record.source {
            Source::SessionRound(source) => {
                let declared = source.prefix_len + source.input_len;
                let sent = source.prompt_len;
                let served = record
                    .outcome
                    .server_usage
                    .as_ref()
                    .and_then(|usage| usage.prompt_tokens);
                sent != declared || served != Some(declared) || source.prefix_shortfall_tokens > 0
            }
            Source::IndependentRequest(_) => false,
        })
        .count();
    // Two counts of one thing: ids the client parsed, and completions the server
    // reported. The stub sends one id per chunk, so the event count is the third.
    let delivery_disagreements = successes
        .iter()
        .filter(|record| {
            let served = record
                .outcome
                .server_usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens);
            served != Some(record.outcome.output_len_actual)
                || record.outcome.token_event_count != shape.output_len
        })
        .count();
    // Round 1 of every session reuses round 0's whole conversation. The new
    // suffix was never sent, so a full-prompt hit would be just as wrong as no
    // hit: the stub must report precisely the derived prefix the client planned.
    let prefix_disagreements = successes
        .iter()
        .filter(|record| match &record.source {
            Source::SessionRound(source) if source.derived_prefix_len > 0 => {
                record
                    .outcome
                    .server_usage
                    .as_ref()
                    .and_then(|usage| usage.cached_prompt_tokens)
                    != Some(source.derived_prefix_len)
            }
            _ => false,
        })
        .count();

    Ok(vec![
        Check::new(
            "ttft_is_the_servers_prefill",
            "Time to first token is the server's, not the client's: it measures from the \
             moment the request went on the wire to the moment the first generated id came \
             back, and adds nothing of its own.",
            "median TTFT",
            ttft,
            "ms",
            Bound::Within {
                of: timing.prefill_delay_ms,
                by: 5.0,
            },
            "Five milliseconds over a 50 ms prefill covers loopback and SSE parsing; a client \
             folding its own queueing into TTFT would be out by far more than that under any \
             load worth measuring.",
        )
        .with_detail(format!("{} successful rounds", successes.len())),
        Check::new(
            "tpot_is_the_servers_chunk_gap",
            "Time per output token is the steady-state delivery pace after the first event, \
             with the first event's tokens excluded from the denominator — not the mean of \
             everything divided by everything.",
            "median token-delivery TPOT",
            tpot,
            "ms",
            Bound::Within {
                of: timing.chunk_delay_ms,
                by: 1.0,
            },
            "One millisecond on a 2 ms gap. Anything looser would also accept a TPOT that \
             averaged in the 50 ms prefill, which is the specific mistake this separates.",
        ),
        Check::new(
            "end_to_end_is_prefill_plus_the_stream",
            "End to end is the sum of what the server was told to do and nothing else — no \
             client post-processing leaking into the number.",
            "median total duration",
            total,
            "ms",
            Bound::Within {
                of: expected_total,
                by: 10.0,
            },
            "Ten milliseconds over an 80 ms response. Wider than the TTFT bound because this \
             accumulates every one of the 15 inter-chunk gaps.",
        )
        .with_detail(format!(
            "{} ms prefill + {} gaps × {} ms = {expected_total} ms",
            timing.prefill_delay_ms,
            shape.output_len - 1,
            timing.chunk_delay_ms,
        )),
        Check::new(
            "output_length_is_the_traces",
            "Output length is the trace's, enforced with max_tokens and ignore_eos — never \
             the model's own stopping point.",
            "rounds delivering the wrong token count",
            short_outputs as f64,
            "rounds",
            Bound::Exactly { value: 0.0 },
            "Exact: a single short response would make every per-token average in the run a \
             different quantity than it claims to be.",
        ),
        Check::new(
            "prompt_length_is_the_traces",
            "What went on the wire is the prompt the trace declared, and the server agrees \
             about its length.",
            "rounds whose prompt length disagreed",
            prompt_mismatches as f64,
            "rounds",
            Bound::Exactly { value: 0.0 },
            "Exact, and it checks three numbers against each other: the trace's, the \
             client's, and the server's. Any two agreeing is not enough.",
        ),
        Check::new(
            "the_server_agrees_about_what_it_delivered",
            "The client's own token count and the server's usage report are the same number, \
             and the timeline saw one event per token the stub sent.",
            "rounds where the two counts diverged",
            delivery_disagreements as f64,
            "rounds",
            Bound::Exactly { value: 0.0 },
            "Exact. These are two independent counts of one thing — ids the client parsed and \
             completions the server reported — and a repo that reports both must not be able \
             to report them differently.",
        ),
        Check::new(
            "planned_prefix_is_the_prefix_that_hit",
            "A prefix the client planned to reuse is a prefix the server actually had. \
             Against a stub that caches exactly what it was sent, a gap here is the client \
             sending different ids than it thinks it is.",
            "later rounds whose cache hit disagreed",
            prefix_disagreements as f64,
            "rounds",
            Bound::Exactly { value: 0.0 },
            "Exact against this server. Against a real one the gap is the interesting \
             quantity; against a stub with a perfect cache it is a client defect.",
        ),
    ])
}

struct RunSpec {
    trace: String,
    format: &'static str,
    rate: Option<f64>,
    directory: PathBuf,
    timeline: bool,
}

/// Run one replay and read back what it logged.
///
/// Built through `Args::parse_from` rather than by filling in a struct literal:
/// the harness then exercises the same argument surface a user types, and a new
/// required flag breaks it here instead of drifting silently.
async fn replay(
    arguments: &SelfcheckArgs,
    fixtures: &Fixtures,
    server: &Stub,
    corpus: &mut CorpusCache,
    spec: RunSpec,
) -> Result<Vec<Record>> {
    std::fs::create_dir_all(&spec.directory)
        .with_context(|| format!("failed to create {}", spec.directory.display()))?;
    let log_path = spec.directory.join("requests.jsonl");

    let mut argv: Vec<String> = vec![
        "session_runner".into(),
        "--trace".into(),
        spec.trace,
        "--input-file-format".into(),
        spec.format.into(),
        "--text-file".into(),
        fixtures.corpus.display().to_string(),
        "--tokenizer".into(),
        arguments.tokenizer.clone(),
        "--model".into(),
        "selfcheck".into(),
        "--base-url".into(),
        server.base_url(),
        "--backend".into(),
        "vllm-tokens".into(),
        "--log-path".into(),
        log_path.display().to_string(),
        "--summary-path".into(),
        spec.directory.join("summary.json").display().to_string(),
        "--timeline".into(),
        spec.timeline.to_string(),
        "--timeline-path".into(),
        spec.directory
            .join("timeline.parquet")
            .display()
            .to_string(),
    ];
    if let Some(rate) = spec.rate {
        argv.push("--rate".into());
        argv.push(rate.to_string());
    }

    let args = Args::parse_from(argv);
    run_once_reusing(args, corpus)
        .await
        .with_context(|| format!("the {} replay failed", spec.directory.display()))?;
    record::load(&log_path)
}

fn dropped_timelines(summary_path: &std::path::Path) -> Result<usize> {
    let text = std::fs::read_to_string(summary_path)
        .with_context(|| format!("failed to read {}", summary_path.display()))?;
    let summary: serde_json::Value = serde_json::from_str(&text)?;
    Ok(summary["timeline"]["dropped_requests"]
        .as_u64()
        .unwrap_or_default() as usize)
}
