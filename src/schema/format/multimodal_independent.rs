//! Canonical JSON-lines artifact for independent mixed-modality requests.

use std::collections::HashSet;
use std::io::BufRead;

use anyhow::{bail, Context, Result};

use crate::schema::RequestSpec;

pub fn load(path: &str) -> Result<Vec<RequestSpec>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let reader = std::io::BufReader::new(file);
    let mut requests = Vec::new();
    let mut ids = HashSet::new();
    let mut previous_arrival = None;

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("read {path}:{line_number}"))?;
        if line.trim().is_empty() {
            bail!("{path}:{line_number}: blank lines are not allowed");
        }
        let request: RequestSpec = serde_json::from_str(&line)
            .with_context(|| format!("parse {path}:{line_number} as RequestSpec"))?;
        request.validate(&format!("{path}:{line_number}"))?;
        if !ids.insert(request.id.clone()) {
            bail!(
                "{path}:{line_number}: duplicate request id {:?}",
                request.id
            );
        }
        if previous_arrival.is_some_and(|previous| request.arrival_time_ms < previous) {
            bail!(
                "{path}:{line_number}: arrival_time_ms {} is earlier than the previous request's {}",
                request.arrival_time_ms,
                previous_arrival.unwrap_or_default(),
            );
        }
        previous_arrival = Some(request.arrival_time_ms);
        requests.push(request);
    }
    if requests.is_empty() {
        bail!("{path}: request artifact must contain at least one request");
    }
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "req_frontend_multimodal_{name}_{}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn loads_valid_mixed_modality_requests() {
        let path = path("valid");
        std::fs::write(
            &path,
            concat!(
                "{\"id\":\"one\",\"arrival_time_ms\":0,\"inputs\":[{\"type\":\"text\",\"text\":\"draw it\"}],\"outputs\":[{\"type\":\"image\",\"width\":512,\"height\":512,\"steps\":20}]}\n",
                "{\"id\":\"two\",\"arrival_time_ms\":10,\"inputs\":[{\"type\":\"audio\",\"asset\":{\"path\":\"voice.wav\"}},{\"type\":\"text\",\"text\":\"transcribe\"}],\"outputs\":[{\"type\":\"text\",\"max_tokens\":32}]}\n",
            ),
        )
        .unwrap();

        let requests = load(path.to_str().unwrap()).unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].assets().count(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_duplicate_ids_and_out_of_order_arrivals() {
        let path = path("invalid");
        let request = "{\"id\":\"same\",\"arrival_time_ms\":10,\"inputs\":[{\"type\":\"text\",\"text\":\"x\"}],\"outputs\":[{\"type\":\"text\",\"max_tokens\":1}]}";
        let earlier = request.replace("\"arrival_time_ms\":10", "\"arrival_time_ms\":0");
        std::fs::write(&path, format!("{request}\n{earlier}\n")).unwrap();

        let error = load(path.to_str().unwrap()).unwrap_err().to_string();
        assert!(error.contains("duplicate request id"));
        std::fs::remove_file(path).ok();
    }
}
