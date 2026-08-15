//! The command-line entry point: parse arguments, run one workload.
//!
//! Everything below this is library code, so a sweep can drive the same run
//! without shelling out to this binary. See [`req_frontend::run_once`].

use anyhow::Result;
use clap::Parser;

use req_frontend::{run_once, Args};

fn main() -> Result<()> {
    let args = Args::parse();
    let workers = args
        .runtime_worker_threads
        .unwrap_or_else(default_runtime_worker_threads);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(workers)
        .build()?
        .block_on(run_once(args))?;
    Ok(())
}

fn default_runtime_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
}
