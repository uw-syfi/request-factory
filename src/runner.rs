//! One run, start to finish: validate the arguments, load the workload, build
//! the token pool, gate on the prefix-cache preflight, release every workload
//! unit, and fold the results into a summary.
//!
//! A run is a library call rather than a `main`. The binary is argument parsing
//! plus one call to [`run_once`], and a sweep is the same call in a loop — which
//! is the point: driving N points in one process is what lets a sweep pay for the
//! tokenizer and the synthetic token pool once instead of per point.

use anyhow::{anyhow, Context, Result};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Semaphore};

use crate::backend::GenerationClient;
use crate::cli::Args;
use crate::executor::{
    prepare_multimodal_requests, run_independent_request, run_multimodal_request, run_session,
    status_task, AdmissionOrder, CommonState, MultimodalState, RunPolicy, Stats,
    TextGenerationState,
};
use crate::record::StepLog;
use crate::release::ArrivalMode;
use crate::schema::{InputFileFormat, InputFileSchema, TraceTag};
use crate::slo::SloSummary;
use crate::slo_source;
use crate::summary::{
    write_logs, write_summary_if_requested, ClientRuntimeSummary, ReplaySummary, RunSummary,
};
use crate::timeline::{self, TimelineSummary};
use crate::tokens::{build_token_pool, load_tokenizer};
use crate::workload::WorkloadSummary;
use crate::workload::{load_workload, ReplayWorkload};

/// Tokenizer and synthetic corpus, kept across runs that can share them.
///
/// This is what makes a sweep of N points cost one tokenization instead of N.
/// The corpus is the expensive part by a wide margin — a hundred million tokens
/// — and it depends only on the tokenizer, the text file, and the pool size, none
/// of which a rate sweep varies.
#[derive(Default)]
pub struct CorpusCache {
    entry: Option<CachedCorpus>,
}

struct CachedCorpus {
    /// Everything the built pool depends on. Compared rather than assumed: a
    /// caller that varies the tokenizer between points must get a rebuilt pool,
    /// not silently reused content in the wrong token space.
    key: (String, String, usize),
    tokenizer: Arc<tokenizers::Tokenizer>,
    token_pool: Arc<Vec<u32>>,
}

impl CorpusCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_or_build(
        &mut self,
        tokenizer_path: &str,
        text_file: &str,
        pool_limit: usize,
    ) -> Result<(Arc<tokenizers::Tokenizer>, Arc<Vec<u32>>)> {
        let key = (
            tokenizer_path.to_string(),
            text_file.to_string(),
            pool_limit,
        );
        if let Some(entry) = self.entry.as_ref() {
            if entry.key == key {
                return Ok((entry.tokenizer.clone(), entry.token_pool.clone()));
            }
        }
        let tokenizer = Arc::new(load_tokenizer(tokenizer_path)?);
        let token_pool = Arc::new(build_token_pool(text_file, tokenizer.as_ref(), pool_limit)?);
        self.entry = Some(CachedCorpus {
            key,
            tokenizer: tokenizer.clone(),
            token_pool: token_pool.clone(),
        });
        Ok((tokenizer, token_pool))
    }
}

/// Execute one workload and return its summary.
///
/// Writes `--log-path` and, when requested, `--summary-path`; the same summary is
/// returned so an in-process caller does not have to read back what it just
/// wrote. A `--dry-run` returns after the static workload summary, without
/// loading tokens or contacting a server.
pub async fn run_once(args: Args) -> Result<RunSummary> {
    run_once_reusing(args, &mut CorpusCache::new()).await
}

/// [`run_once`], reusing a corpus across calls.
///
/// The cache is the caller's, so a sweep decides how long it lives and nothing
/// global outlives a run that did not ask for it.
pub async fn run_once_reusing(args: Args, corpus: &mut CorpusCache) -> Result<RunSummary> {
    // Per-process, and idempotent: a sweep calling this once per point costs
    // nothing, and forgetting it once would cap a large run at the default
    // descriptor limit.
    fdlimit::raise_fd_limit().ok();
    validate(&args)?;

    // What the file says it is, before it is opened. Every reader below checks
    // its header against this rather than trusting whatever columns turn up.
    let input_file_format = InputFileFormat::parse(&args.input_file_format)?;
    let tags = args
        .trace_tags
        .iter()
        .map(|name| TraceTag::parse(name))
        .collect::<Result<Vec<_>>>()?;
    let declaration = InputFileSchema::new(input_file_format, tags)?;
    // Resolved before anything is loaded, so a malformed objective fails in a
    // second rather than after a corpus has been tokenized.
    let objective = slo_source::resolve(&args.trace, args.slo.as_deref())?;
    let mut workload = load_workload(&args.trace, &declaration, args.max_items)?;
    let unit_label = workload.unit_label();
    report_arrival_rate(&args, &mut workload, unit_label)?;

    let replay_summary = ReplaySummary::empty_for(&workload);
    // A run reports attainment when it was given an objective *or* when the
    // trace declares per-request metric bounds -- a trace that sets its own budgets
    // and a run that never mentions them would be a column read and discarded.
    let slo_summary = SloSummary::new(objective, declaration.carries(TraceTag::Slo));
    let workload_summary = WorkloadSummary::from_workload(&workload, args.max_model_len);
    workload_summary.print();
    if args.dry_run {
        let summary = RunSummary {
            workload: workload_summary,
            replay: replay_summary,
            client_runtime: client_runtime_summary(0),
            timeline: TimelineSummary::default(),
            // A dry run sends nothing, so it measures nothing: reporting the
            // declared objective with no steps behind it would read as a result.
            slo: slo_summary,
        };
        write_summary_if_requested(args.summary_path.as_deref(), &summary)?;
        return Ok(summary);
    }

    let workload = match workload {
        ReplayWorkload::MultimodalRequests(requests) => {
            return run_multimodal(
                &args,
                requests,
                workload_summary,
                replay_summary,
                slo_summary,
            )
            .await;
        }
        text_workload => text_workload,
    };

    // Size the synthetic token pool to the workload by default: it must exceed the longest
    // prompt so no single request repeats content, and stay larger than the session count so
    // per-session seed offsets stay distinct (otherwise distant sessions draw identical content
    // and fabricate cross-session prefix-cache hits). The 100M floor gives ~100 sessions of
    // 1M-token context their own non-overlapping content window (~400 MB of u32).
    const MIN_TOKEN_POOL: usize = 100_000_000;
    let pool_limit = args.token_pool_limit.unwrap_or_else(|| {
        workload_summary
            .max_prompt_len()
            .saturating_mul(2)
            .max(workload.unit_count())
            .max(MIN_TOKEN_POOL)
    });
    let tokenizer_path = args
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow!("--tokenizer is required for text-generation formats"))?;
    let text_file = args
        .text_file
        .as_deref()
        .ok_or_else(|| anyhow!("--text-file is required for text-generation formats"))?;
    let (tokenizer, token_pool) = corpus.get_or_build(tokenizer_path, text_file, pool_limit)?;
    if token_pool.len() < workload_summary.max_prompt_len() {
        eprintln!(
            "warning: token pool ({} tokens) is smaller than the longest prompt ({} tokens); \
             synthetic content will repeat within a single request and may distort prefix-cache \
             measurement. Use a larger --text-file corpus.",
            token_pool.len(),
            workload_summary.max_prompt_len(),
        );
    }
    let total_steps = workload_summary.total_steps();

    let client = Arc::new(GenerationClient::new(&args, tokenizer)?);

    // Fail fast if the server won't report prefix-cache hits: otherwise every measured hit
    // rate would silently read as zero. Dry-run returns earlier and never reaches here.
    // Probe the TAIL of the pool: workload unit 0 seeds at offset 0, so a head
    // probe would warm its first prompt and fabricate a cache hit there.
    let probe_len = token_pool.len().min(512);
    client
        .preflight_cache_check(&token_pool[token_pool.len() - probe_len..])
        .await
        .context("prefix-cache preflight failed")?;

    // Bounded in requests, not events: it caps how far the writer may fall
    // behind before timelines are dropped rather than queued. Deep enough that
    // only a genuinely stuck disk reaches it, shallow enough to bound memory.
    const TIMELINE_QUEUE_REQUESTS: usize = 4_096;
    let (timeline_sink, timeline_task) = if args.timeline {
        let (sink, receiver) = timeline::channel(TIMELINE_QUEUE_REQUESTS);
        let dropped = sink.dropped_handle();
        // A thread of its own, not a runtime task: Arrow encoding and zstd are
        // blocking CPU work, and on a worker thread they compete with the
        // requests this file exists to time.
        let path = args.timeline_path.clone();
        let task =
            tokio::task::spawn_blocking(move || timeline::write_timeline(path, receiver, dropped));
        (Some(sink), Some(task))
    } else {
        (None, None)
    };

    let state = Arc::new(TextGenerationState {
        client,
        token_pool,
        common: CommonState {
            policy: RunPolicy::from_args(&args),
            stats: Arc::new(Stats::default()),
            run_start: Instant::now(),
            concurrency_semaphore: args
                .max_concurrency
                .map(|limit| Arc::new(Semaphore::new(limit))),
            // Only under a cap, because it exists to order contention and there is
            // none without one.
            admission_order: args
                .max_concurrency
                .map(|_| Arc::new(AdmissionOrder::new(workload.unit_count()))),
        },
    });

    let (log_tx, log_rx) = mpsc::channel::<StepLog>(100_000);
    let log_task = tokio::spawn(write_logs(
        args.log_path.clone(),
        log_rx,
        replay_summary,
        slo_summary,
    ));
    let status_handle = tokio::spawn(status_task(
        state.common.stats.clone(),
        workload.unit_count(),
        total_steps,
        unit_label,
        state.common.run_start,
    ));

    let mut join_set = tokio::task::JoinSet::new();
    match workload {
        ReplayWorkload::Sessions(sessions) => {
            for (session_ordinal, (session_id, steps)) in sessions.into_iter().enumerate() {
                let state_ref = state.clone();
                let log_tx_ref = log_tx.clone();
                let timeline_ref = timeline_sink.clone();
                join_set.spawn(async move {
                    run_session(
                        state_ref,
                        log_tx_ref,
                        timeline_ref,
                        session_ordinal,
                        session_id,
                        steps,
                    )
                    .await;
                });
            }
        }
        ReplayWorkload::IndependentRequests(requests) => {
            for (request_ordinal, request) in requests.into_iter().enumerate() {
                let state_ref = state.clone();
                let log_tx_ref = log_tx.clone();
                let timeline_ref = timeline_sink.clone();
                join_set.spawn(async move {
                    run_independent_request(
                        state_ref,
                        log_tx_ref,
                        timeline_ref,
                        request_ordinal,
                        request,
                    )
                    .await;
                });
            }
        }
        ReplayWorkload::MultimodalRequests(_) => {
            unreachable!("multimodal workloads return through their own runtime branch")
        }
    }
    // Both writers finish when their last sender is gone.
    drop(log_tx);
    drop(timeline_sink);

    while let Some(result) = join_set.join_next().await {
        if let Err(err) = result {
            eprintln!("workload task join error: {err}");
        }
    }

    let folded = log_task.await?;
    status_handle.await?;
    let runtime_global_queue_depth_peak = state.common.stats.runtime_global_queue_depth_peak();
    let timeline_summary = match timeline_task {
        Some(task) => task.await?,
        None => TimelineSummary::default(),
    };
    let summary = RunSummary {
        workload: workload_summary,
        replay: folded.replay,
        client_runtime: client_runtime_summary(runtime_global_queue_depth_peak),
        timeline: timeline_summary,
        slo: folded.slo,
    };
    write_summary_if_requested(args.summary_path.as_deref(), &summary)?;

    Ok(summary)
}

async fn run_multimodal(
    args: &Args,
    requests: Vec<crate::schema::RequestSpec>,
    workload_summary: WorkloadSummary,
    replay_summary: ReplaySummary,
    slo_summary: Option<SloSummary>,
) -> Result<RunSummary> {
    // All reads, hashes, MIME inference, and base64 encoding happen before the
    // run clock starts, so request timing contains transport/server work rather
    // than benchmark setup.
    let prepared = prepare_multimodal_requests(&args.trace, requests)?;
    let total_steps = prepared.len();
    let common = CommonState {
        policy: RunPolicy::from_args(args),
        stats: Arc::new(Stats::default()),
        run_start: Instant::now(),
        concurrency_semaphore: args
            .max_concurrency
            .map(|limit| Arc::new(Semaphore::new(limit))),
        admission_order: args
            .max_concurrency
            .map(|_| Arc::new(AdmissionOrder::new(total_steps))),
    };
    let state = Arc::new(MultimodalState {
        common,
        client: Arc::new(GenerationClient::new_chat(args)?),
    });

    const TIMELINE_QUEUE_REQUESTS: usize = 4_096;
    let (timeline_sink, timeline_task) = if args.timeline {
        let (sink, receiver) = timeline::channel(TIMELINE_QUEUE_REQUESTS);
        let dropped = sink.dropped_handle();
        let path = args.timeline_path.clone();
        let task =
            tokio::task::spawn_blocking(move || timeline::write_timeline(path, receiver, dropped));
        (Some(sink), Some(task))
    } else {
        (None, None)
    };
    let (log_tx, log_rx) = mpsc::channel::<StepLog>(100_000);
    let log_task = tokio::spawn(write_logs(
        args.log_path.clone(),
        log_rx,
        replay_summary,
        slo_summary,
    ));
    let status_handle = tokio::spawn(status_task(
        state.common.stats.clone(),
        total_steps,
        total_steps,
        "requests",
        state.common.run_start,
    ));

    let mut join_set = tokio::task::JoinSet::new();
    for (request_ordinal, request) in prepared.into_iter().enumerate() {
        let state_ref = state.clone();
        let log_tx_ref = log_tx.clone();
        let timeline_ref = timeline_sink.clone();
        join_set.spawn(async move {
            run_multimodal_request(
                state_ref,
                log_tx_ref,
                timeline_ref,
                request_ordinal,
                request,
            )
            .await;
        });
    }
    drop(log_tx);
    drop(timeline_sink);
    while let Some(result) = join_set.join_next().await {
        if let Err(error) = result {
            eprintln!("workload task join error: {error}");
        }
    }

    let folded = log_task.await?;
    status_handle.await?;
    let timeline_summary = match timeline_task {
        Some(task) => task.await?,
        None => TimelineSummary::default(),
    };
    let summary = RunSummary {
        workload: workload_summary,
        replay: folded.replay,
        client_runtime: client_runtime_summary(
            state.common.stats.runtime_global_queue_depth_peak(),
        ),
        timeline: timeline_summary,
        slo: folded.slo,
    };
    write_summary_if_requested(args.summary_path.as_deref(), &summary)?;
    Ok(summary)
}

/// Reject argument combinations that are contradictory rather than merely
/// unusual, before anything is loaded or any request is sent.
fn validate(args: &Args) -> Result<()> {
    if args.max_concurrency == Some(0) {
        return Err(anyhow!("--max-concurrency must be greater than 0"));
    }
    if args.arrival_mode == ArrivalMode::Saturated && args.rate.is_some() {
        return Err(anyhow!(
            "--rate rescales the trace's arrival timeline, which --arrival-mode saturated \
             discards. Drop one of them; to bound a saturated run, use --max-concurrency."
        ));
    }
    if args.skip_when_reaching_limit && args.max_model_len.is_none() {
        return Err(anyhow!(
            "--skip-when-reaching-limit requires --max-model-len"
        ));
    }
    let format = InputFileFormat::parse(&args.input_file_format)?;
    if format == InputFileFormat::MultimodalIndependentV1 {
        if args.backend != crate::cli::BackendKind::OpenaiChat {
            return Err(anyhow!(
                "multimodal-independent-v1 requires --backend openai-chat"
            ));
        }
        if args.max_model_len.is_some() || args.skip_when_reaching_limit {
            return Err(anyhow!(
                "context-token guards are text-replay-only and cannot be used with multimodal-independent-v1"
            ));
        }
    } else if args.backend == crate::cli::BackendKind::OpenaiChat {
        return Err(anyhow!(
            "--backend openai-chat requires multimodal-independent-v1"
        ));
    }
    Ok(())
}

/// Say on stderr which arrival timeline this run will actually replay, and
/// rescale the workload when `--rate` asked for a different one.
fn report_arrival_rate(
    args: &Args,
    workload: &mut ReplayWorkload,
    unit_label: &'static str,
) -> Result<()> {
    if args.arrival_mode == ArrivalMode::Saturated {
        eprintln!(
            "{} arrival rate | saturated (recorded arrivals ignored{})",
            unit_label,
            match args.max_concurrency {
                Some(cap) => format!(", bounded by --max-concurrency {cap}"),
                None => ", unbounded".to_string(),
            },
        );
    } else if let Some(target_rate) = args.rate {
        let adjustment = workload.apply_arrival_rate(target_rate)?;
        eprintln!(
            "{} arrival rate | trace={:.6}/s target={:.6}/s time_scale={:.6}",
            unit_label, adjustment.trace_rate, adjustment.target_rate, adjustment.time_scale,
        );
    } else if let Some(trace_rate) = workload.arrival_rate() {
        eprintln!(
            "{} arrival rate | trace={:.6}/s (unchanged)",
            unit_label, trace_rate
        );
    } else {
        eprintln!(
            "{} arrival rate | unavailable (need at least two units with distinct arrival times)",
            unit_label,
        );
    }
    Ok(())
}

fn client_runtime_summary(sampled_global_queue_depth_peak: usize) -> ClientRuntimeSummary {
    ClientRuntimeSummary {
        tokio_worker_threads: tokio::runtime::Handle::current().metrics().num_workers(),
        sampled_global_queue_depth_peak,
    }
}
