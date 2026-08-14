use anyhow::{bail, Result};

use crate::schema::session_execution_v2 as v2;
use crate::schema::{RequestScheduling, TraceDeclaration};

/// One round of one session, as the replay runtime executes it.
///
/// `prefix_len` is what the round will actually reuse, not what the original
/// source reported: the split was resolved when the canonical trace was
/// generated, so nothing here reinterprets it.
#[derive(Debug, Clone)]
pub(crate) struct SessionStep {
    pub(crate) request_id: String,
    pub(crate) session_id: String,
    pub(crate) arrival_time: f64,
    pub(crate) round_idx: usize,
    pub(crate) prefix_len: usize,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) tool_wait_after_ms: f64,
    /// What the `slo` tag declared for this round, empty when it was not
    /// declared. Per round, not per session: within one conversation a tool-call
    /// turn and a long answer owe different things.
    pub(crate) scheduling: RequestScheduling,
}

/// Sessions in replay order, each with its rounds in round order.
///
/// A sequence, not a map keyed by session id: row order carries meaning — it is
/// the release order and the tie-break under equal arrivals — and a map would
/// silently replace it with the lexicographic order of the identifier.
pub(crate) type SessionPlans = Vec<(String, Vec<SessionStep>)>;

/// Load a canonical `session-execution-v2` trace.
///
/// Nothing is re-derived here. The file's own row order is the replay order,
/// its `request_id` is the identifier, and its `prefix_len` is already
/// guaranteed to exist by the time the round runs — which is exactly why a
/// simulator can read the same file and reach the same plan.
pub(super) fn load(
    path: &str,
    declaration: &TraceDeclaration,
    max_sessions: Option<usize>,
) -> Result<SessionPlans> {
    let rows = v2::load(path, declaration)?;

    let mut sessions: SessionPlans = Vec::new();
    for declared in rows {
        let row = declared.row;
        let step = SessionStep {
            request_id: row.request_id,
            session_id: row.session_id,
            arrival_time: row.arrival_time_ms,
            round_idx: row.round_idx,
            prefix_len: row.prefix_len,
            input_len: row.input_len,
            output_len: row.output_len,
            tool_wait_after_ms: row.tool_wait_after_ms,
            scheduling: declared.scheduling,
        };
        match sessions.last_mut() {
            Some((session_id, steps)) if *session_id == step.session_id => steps.push(step),
            _ => sessions.push((step.session_id.clone(), vec![step])),
        }
    }
    if sessions.is_empty() {
        bail!("{path} contains no sessions");
    }
    Ok(apply_session_cap(sessions, max_sessions))
}

/// Keep the first `max` sessions in trace order.
///
/// Trace order is arrival order in a canonical file, so this keeps the earliest
/// arrivals — not the lexicographically smallest identifiers, whose ordering has
/// no relationship to when the work is released.
fn apply_session_cap(mut sessions: SessionPlans, max: Option<usize>) -> SessionPlans {
    if let Some(max) = max {
        sessions.truncate(max);
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// The declaration every canonical file carries, which is what the runner
    /// builds for `--trace-format session`.
    fn canonical() -> TraceDeclaration {
        TraceDeclaration::parse_with_schema("text_generation", &[], "session-execution-v2").unwrap()
    }

    fn write_temp(name: &str, contents: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!("req-frontend-session-test-{name}.csv"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path.display().to_string()
    }

    #[test]
    fn cap_keeps_the_earliest_arrivals_not_the_smallest_identifiers() {
        let path = write_temp(
            "cap",
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session_2_round_000000,2,0,0.000000,0,10,4,0.000000\n\
             session_10_round_000000,10,0,5.000000,0,10,4,0.000000\n",
        );

        let sessions = load(&path, &canonical(), Some(1)).unwrap();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].0, "2");
    }

    /// A raw, unmaterialized CSV must be rejected rather than half-read.
    ///
    /// The legacy loader this replaced accepted such a file: every column it
    /// required was present in a canonical header too, and serde ignored the
    /// rest, so `arrival_time` silently defaulted to 0 and the whole timeline
    /// collapsed with no error. The canonical row type has no such overlap, and
    /// since the declaration was wired in, the rejection happens on the header
    /// before a single row is read — and names every column that disagrees.
    #[test]
    fn raw_unmaterialized_csv_is_rejected_at_parse() {
        let path = write_temp(
            "raw",
            "id,arrival_time,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             abc,0,0,0,10,4,0\n",
        );

        let error = load(&path, &canonical(), None).unwrap_err().to_string();

        assert!(error.contains("header does not match"), "{error}");
        assert!(error.contains("arrival_time_ms"), "{error}");
    }

    #[test]
    fn loader_preserves_file_order_and_ids() {
        let path = write_temp(
            "v2",
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session_b_round_000000,b,0,0.000000,0,512,64,0.000000\n\
             session_b_round_000001,b,1,0.000000,576,128,64,0.000000\n\
             session_a_round_000000,a,0,250.000000,0,400,48,0.000000\n",
        );

        let sessions = load(&path, &canonical(), None).unwrap();

        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].0, "b");
        assert_eq!(sessions[1].0, "a");
        assert_eq!(sessions[0].1[1].request_id, "session_b_round_000001");
        assert_eq!(sessions[0].1[1].prefix_len, 576);
    }

    #[test]
    fn loader_rejects_a_non_canonical_file() {
        let path = write_temp(
            "v2-bad",
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session_a_round_000000,a,0,0.000000,128,512,64,0.000000\n",
        );

        let error = load(&path, &canonical(), None).unwrap_err().to_string();

        assert!(error.contains("not a canonical"), "{error}");
    }
}
