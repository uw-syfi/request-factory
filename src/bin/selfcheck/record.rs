//! Reading the JSONL back, through the public contract rather than the private type.
//!
//! This deliberately re-declares the handful of fields it needs instead of
//! sharing `record.rs`'s struct. The log is a published schema with a version
//! number, and a harness that deserializes it with the same type that wrote it
//! would agree with any rename either of them made. Here, a field that
//! disappears is a compile-time absence in this file and a failing check, which
//! is the point.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// The schema this harness knows how to read. Checked, not assumed: a bumped
/// version means the fields below may mean something else.
pub const EXPECTED_SCHEMA_VERSION: u32 = 15;

#[derive(Debug, Deserialize)]
pub struct Record {
    pub schema_version: u32,
    pub source: Source,
    pub outcome: Outcome,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum Source {
    IndependentRequest(IndependentSource),
    SessionRound(SessionSource),
}

#[derive(Debug, Deserialize)]
pub struct IndependentSource {
    /// Scheduled arrival to the task resuming, measured before the concurrency
    /// semaphore. The evidence that the client released when the trace said.
    pub arrival_release_lag_ms: f64,
}

#[derive(Debug, Deserialize)]
pub struct SessionSource {
    pub prefix_len: usize,
    pub input_len: usize,
    pub prompt_len: usize,
    pub derived_prefix_len: usize,
    pub prefix_shortfall_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct Outcome {
    pub status: String,
    pub output_len_actual: usize,
    pub submit_timestamp: f64,
    pub first_token_id_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub token_delivery_tpot_ms: Option<f64>,
    pub total_duration_ms: f64,
    pub token_event_count: usize,
    pub server_usage: Option<ServerUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ServerUsage {
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub cached_prompt_tokens: Option<usize>,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.status == "SUCCESS"
    }

    /// The same fallback the SLO fold applies, so a check and the run summary
    /// cannot disagree about which requests measured a TTFT.
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_id_ms.or(self.first_token_ms)
    }
}

pub fn load(path: &Path) -> Result<Vec<Record>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut records = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .with_context(|| format!("{}:{} is not a step log", path.display(), number + 1))?;
        if record.schema_version != EXPECTED_SCHEMA_VERSION {
            bail!(
                "{} declares step-log schema v{}, and this harness reads v{}. The fields these \
                 checks read may no longer mean what they meant; update selfcheck rather than \
                 the assertion it fails.",
                path.display(),
                record.schema_version,
                EXPECTED_SCHEMA_VERSION,
            );
        }
        records.push(record);
    }
    if records.is_empty() {
        bail!("{} has no records", path.display());
    }
    Ok(records)
}
