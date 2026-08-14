use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::schema::{RequestScheduling, RequestSlo, TraceDeclaration, TraceTag};

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
    /// What the `slo` tag declared for this row, empty when it was not declared.
    #[serde(skip)]
    pub(crate) slo: RequestSlo,
    /// What the `priority` tag declared for this row.
    #[serde(skip)]
    pub(crate) scheduling: RequestScheduling,
}

/// Read a native trace, checking it against what it says it is.
///
/// The header is verified before any row is parsed. That is the whole point of a
/// declaration: a column nobody expected is data whose author meant something by
/// it, and silently dropping it is how a trace and the run that replayed it stop
/// describing the same workload.
pub(super) fn load(
    path: &str,
    declaration: &TraceDeclaration,
    max_requests: Option<usize>,
) -> Result<Vec<IndependentRequest>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open independent-request trace: {path}"))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read the header of {path}"))?
        .clone();
    declaration
        .verify_header(headers.iter())
        .map_err(|mismatch| anyhow!("{path}: {mismatch}"))?;

    let reads_slo = declaration.carries(TraceTag::Slo);
    let reads_priority = declaration.carries(TraceTag::Priority);
    let mut requests = Vec::new();
    for (index, record) in reader.records().enumerate() {
        if max_requests.is_some_and(|max| requests.len() >= max) {
            break;
        }
        let line = index + 2; // the header occupies line 1
        let record = record.with_context(|| format!("{path}: failed to read line {line}"))?;
        let mut request: IndependentRequest = record
            .deserialize(Some(&headers))
            .with_context(|| format!("{path}: failed to parse line {line}"))?;
        if reads_slo {
            request.slo = record
                .deserialize(Some(&headers))
                .with_context(|| format!("{path}: failed to parse line {line}"))?;
            request.slo.validate(&format!("{path} line {line}"))?;
        }
        if reads_priority {
            request.scheduling = record
                .deserialize(Some(&headers))
                .with_context(|| format!("{path}: failed to parse line {line}"))?;
            request
                .scheduling
                .validate(&format!("{path} line {line}"))?;
        }
        requests.push(request);
    }
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "req_frontend_native_{}_{name}.csv",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn an_undeclared_column_is_refused_rather_than_dropped() {
        // serde alone would have ignored `ttft_slo_ms` and produced a run that
        // silently held every request to no TTFT bound at all.
        let path = write(
            "undeclared",
            "id,arrival_time,input_len,output_len,ttft_slo_ms\nreq-1,0,16,4,500\n",
        );

        let error = load(path.to_str().unwrap(), &TraceDeclaration::text(), None)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).ok();

        assert!(error.contains("ttft_slo_ms"), "{error}");
    }

    #[test]
    fn a_declared_tag_is_read_per_row_and_may_be_blank_on_any_of_them() {
        let path = write(
            "declared",
            "id,arrival_time,input_len,output_len,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority\n\
             req-1,0,16,4,500,,2000,7\n\
             req-2,10,16,4,,,,\n",
        );
        let declaration = TraceDeclaration::parse(
            "text_generation",
            &["slo".to_string(), "priority".to_string()],
        )
        .unwrap();

        let requests = load(path.to_str().unwrap(), &declaration, None).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(requests[0].slo.ttft_slo_ms, Some(500.0));
        assert_eq!(requests[0].slo.e2e_slo_ms, Some(2000.0));
        assert_eq!(requests[0].scheduling.priority, Some(7));
        assert!(requests[1].slo.is_empty());
        assert!(requests[1].scheduling.is_empty());
    }

    #[test]
    fn a_declared_column_that_is_missing_is_refused_before_any_row_is_read() {
        let path = write(
            "missing",
            "id,arrival_time,input_len,output_len,priority\nreq-1,0,16,4,7\n",
        );
        let declaration = TraceDeclaration::parse("text_generation", &["slo".to_string()]).unwrap();

        let error = load(path.to_str().unwrap(), &declaration, None)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).ok();

        assert!(error.contains("ttft_slo_ms"), "{error}");
    }

    #[test]
    fn an_impossible_metric_bound_names_the_line_it_is_on() {
        let path = write(
            "bad-slo",
            "id,arrival_time,input_len,output_len,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms\n\
             req-1,0,16,4,500,,\n\
             req-2,10,16,4,0,,\n",
        );
        let declaration = TraceDeclaration::parse("text_generation", &["slo".to_string()]).unwrap();

        let error = load(path.to_str().unwrap(), &declaration, None)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).ok();

        assert!(error.contains("line 3"), "{error}");
    }
}
