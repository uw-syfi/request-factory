//! The command-line entry point: parse arguments, run one workload.
//!
//! Everything below this is library code, so a sweep can drive the same run
//! without shelling out to this binary. See [`req_frontend::run_once`].

use anyhow::Result;
use clap::Parser;

use req_frontend::{run_once, Args};

#[tokio::main]
async fn main() -> Result<()> {
    run_once(Args::parse()).await?;
    Ok(())
}
