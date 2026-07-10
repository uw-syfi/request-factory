use anyhow::{Context, Result};
use serde::Deserialize;

/// One independent request in VibeSim's L7 trace schema.
///
/// This intentionally does not pretend to be a session round: there is no
/// prefix, round index, or tool wait in this source format.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VibeSimRequest {
    pub(crate) id: String,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) arrival_time: f64,
}

pub(super) fn load(path: &str, max_requests: Option<usize>) -> Result<Vec<VibeSimRequest>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open VibeSim trace: {path}"))?;
    let mut requests = Vec::new();
    for row in reader.deserialize() {
        if max_requests.is_some_and(|max| requests.len() >= max) {
            break;
        }
        requests.push(row.context("failed to parse VibeSim trace row")?);
    }
    Ok(requests)
}
