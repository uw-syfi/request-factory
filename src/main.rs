mod backend;
mod cli;
mod executor;
mod record;
mod summary;
mod tokens;
mod trace;
mod util;
mod workload;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Semaphore};

use backend::GenerationClient;
use cli::{Args, ArrivalMode};
use executor::{
    run_independent_request, run_session, status_task, AdmissionOrder, AppState, Stats,
};
use record::StepLog;
use summary::{
    write_logs, write_summary_if_requested, ClientRuntimeSummary, ReplaySummary, RunSummary,
};
use tokens::{build_token_pool, load_tokenizer};
use trace::{load_workload, ReplayWorkload};
use workload::WorkloadSummary;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    fdlimit::raise_fd_limit().ok();

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
    let mut workload = load_workload(&args.trace, args.trace_format, args.max_items)?;
    let unit_label = workload.unit_label();
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
    let replay_summary = ReplaySummary::empty_for(&workload);
    let workload_summary = WorkloadSummary::from_workload(&workload, args.max_model_len);
    workload_summary.print();
    if args.dry_run {
        write_summary_if_requested(
            args.summary_path.as_deref(),
            RunSummary {
                workload: workload_summary,
                replay: replay_summary,
                client_runtime: client_runtime_summary(0),
            },
        )?;
        return Ok(());
    }

    let tokenizer = Arc::new(load_tokenizer(&args.tokenizer)?);
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
    let token_pool = Arc::new(build_token_pool(
        &args.text_file,
        tokenizer.as_ref(),
        pool_limit,
    )?);
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

    let state = Arc::new(AppState {
        args: args.clone(),
        client,
        token_pool,
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
    });

    let (log_tx, log_rx) = mpsc::channel::<StepLog>(100_000);
    let log_task = tokio::spawn(write_logs(args.log_path.clone(), log_rx, replay_summary));
    let status_handle = tokio::spawn(status_task(
        state.stats.clone(),
        workload.unit_count(),
        total_steps,
        unit_label,
        state.run_start,
    ));

    let mut join_set = tokio::task::JoinSet::new();
    match workload {
        ReplayWorkload::Sessions(sessions) => {
            for (session_ordinal, (session_id, steps)) in sessions.into_iter().enumerate() {
                let state_ref = state.clone();
                let log_tx_ref = log_tx.clone();
                join_set.spawn(async move {
                    run_session(state_ref, log_tx_ref, session_ordinal, session_id, steps).await;
                });
            }
        }
        ReplayWorkload::IndependentRequests(requests) => {
            for (request_ordinal, request) in requests.into_iter().enumerate() {
                let state_ref = state.clone();
                let log_tx_ref = log_tx.clone();
                join_set.spawn(async move {
                    run_independent_request(state_ref, log_tx_ref, request_ordinal, request).await;
                });
            }
        }
    }
    drop(log_tx);

    while let Some(result) = join_set.join_next().await {
        if let Err(err) = result {
            eprintln!("workload task join error: {err}");
        }
    }

    let replay_summary = log_task.await?;
    status_handle.await?;
    write_summary_if_requested(
        args.summary_path.as_deref(),
        RunSummary {
            workload: workload_summary,
            replay: replay_summary,
            client_runtime: client_runtime_summary(state.stats.runtime_global_queue_depth_peak()),
        },
    )?;

    Ok(())
}

fn client_runtime_summary(sampled_global_queue_depth_peak: usize) -> ClientRuntimeSummary {
    ClientRuntimeSummary {
        tokio_worker_threads: tokio::runtime::Handle::current().metrics().num_workers(),
        sampled_global_queue_depth_peak,
    }
}
