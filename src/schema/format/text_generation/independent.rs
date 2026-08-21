use anyhow::Result;
use serde::Deserialize;

use crate::schema::format::load_utils::{
    self as load_utils, validate_identity_and_arrival, validate_positive, IndependentRow,
    ParsedIndependentRow,
};
use crate::schema::{InputFileSchema, RequestPriority, RequestSlo};

/// Complete base columns for an independent text-generation input file.
pub const COLUMNS: &[&str] = &["id", "arrival_time", "input_len", "output_len"];

/// One independent text-generation request.
///
/// This intentionally does not pretend to be a session round: there is no
/// prefix, round index, or tool wait in this source format.
#[derive(Debug, Clone, Deserialize)]
pub struct TextGenerationRow {
    pub id: String,
    pub input_len: usize,
    pub output_len: usize,
    pub arrival_time: f64,
}

impl IndependentRow for TextGenerationRow {
    fn validate(&self, at: &str) -> Result<()> {
        validate_identity_and_arrival(&self.id, self.arrival_time, at)?;
        validate_positive(self.input_len, "input_len", at)?;
        validate_positive(self.output_len, "output_len", at)
    }
}

#[derive(Debug, Clone)]
pub struct IndependentRequest {
    pub id: String,
    pub input_len: usize,
    pub output_len: usize,
    pub arrival_time: f64,
    pub slo: RequestSlo,
    pub priority: RequestPriority,
}

/// Load every declared field for consumers that own their own request type.
///
/// The HTTP replay runtime intentionally narrows these rows to
/// [`IndependentRequest`], but simulators also consume the session and
/// speculative tags. Exposing the validated rows keeps those consumers on the
/// same parser instead of forcing them to reimplement this format.
pub fn load(
    path: &str,
    input_file_schema: &InputFileSchema,
) -> Result<Vec<ParsedIndependentRow<TextGenerationRow>>> {
    load_utils::load(path, input_file_schema)
}

/// Project validated rows into the narrower HTTP replay request shape.
///
/// The header is verified before any row is parsed. That is the whole point of a
/// declaration: a column nobody expected is data whose author meant something by
/// it, and silently dropping it is how a trace and the run that replayed it stop
/// describing the same workload.
pub fn load_requests(
    path: &str,
    input_file_schema: &InputFileSchema,
) -> Result<Vec<IndependentRequest>> {
    let rows = load(path, input_file_schema)?;
    Ok(rows
        .into_iter()
        .map(|parsed| IndependentRequest {
            id: parsed.row.id,
            input_len: parsed.row.input_len,
            output_len: parsed.row.output_len,
            arrival_time: parsed.row.arrival_time,
            slo: parsed.slo,
            priority: parsed.priority,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{InputFileFormat, TraceTag};
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

        let error = load(
            path.to_str().unwrap(),
            &InputFileSchema::text_generation_independent(),
        )
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
        let input_file_schema = InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Slo, TraceTag::Priority],
        )
        .unwrap();

        let requests = load(path.to_str().unwrap(), &input_file_schema).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(requests[0].slo.ttft_slo_ms, Some(500.0));
        assert_eq!(requests[0].slo.e2e_slo_ms, Some(2000.0));
        assert_eq!(requests[0].priority.priority, Some(7));
        assert!(requests[1].slo.is_empty());
        assert!(requests[1].priority.is_empty());
    }

    #[test]
    fn load_preserves_tags_the_http_request_shape_does_not_consume() {
        let path = write(
            "all-tags",
            "id,arrival_time,input_len,output_len,session_id,prefix_kv,tool_wait_after_ms,accept_rate\n\
             req-1,0,16,4,session-a,8,25,0.75\n",
        );
        let input_file_schema = InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Session, TraceTag::Speculative],
        )
        .unwrap();

        let rows = load(path.to_str().unwrap(), &input_file_schema).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(rows[0].session.session_id.as_deref(), Some("session-a"));
        assert_eq!(rows[0].session.prefix_kv, Some(8));
        assert_eq!(rows[0].speculative.accept_rate, Some(0.75));
    }

    #[test]
    fn a_declared_column_that_is_missing_is_refused_before_any_row_is_read() {
        let path = write(
            "missing",
            "id,arrival_time,input_len,output_len,priority\nreq-1,0,16,4,7\n",
        );
        let input_file_schema = InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Slo],
        )
        .unwrap();

        let error = load(path.to_str().unwrap(), &input_file_schema)
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
        let input_file_schema = InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Slo],
        )
        .unwrap();

        let error = load(path.to_str().unwrap(), &input_file_schema)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).ok();

        assert!(error.contains("line 3"), "{error}");
    }
}
