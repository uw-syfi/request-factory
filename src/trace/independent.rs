use anyhow::{Context, Result};
use serde::Deserialize;

/// One independent request from the generic request CSV frontend.
///
/// This intentionally does not pretend to be a session round: there is no
/// prefix, round index, or tool wait in this source format.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct IndependentRequest {
    pub(crate) id: String,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) arrival_time: f64,
}

pub(super) fn load(path: &str, max_requests: Option<usize>) -> Result<Vec<IndependentRequest>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open independent-request trace: {path}"))?;
    let mut requests = Vec::new();
    for row in reader.deserialize() {
        if max_requests.is_some_and(|max| requests.len() >= max) {
            break;
        }
        requests.push(row.context("failed to parse independent-request trace row")?);
    }
    Ok(requests)
}
