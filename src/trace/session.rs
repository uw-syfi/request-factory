use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

use tracelab_replay::v2;

/// One round of one session, as the replay runtime executes it.
///
/// Both session frontends produce this type. The legacy raw frontend leaves
/// `prefix_len` as the source reported it, so the runtime still has to resolve a
/// context policy; the canonical frontend has already resolved it upstream.
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
}

/// Sessions in replay order, each with its rounds in round order.
///
/// A sequence, not a map keyed by session id: row order carries meaning — it is
/// the release order and the tie-break under equal arrivals — and a map would
/// silently replace it with the lexicographic order of the identifier.
pub(crate) type SessionPlans = Vec<(String, Vec<SessionStep>)>;

/// One row of the legacy raw session schema, where `id` names the session.
#[derive(Debug, Clone, Deserialize)]
struct RawSessionRow {
    #[serde(alias = "id")]
    session_id: String,
    #[serde(default)]
    arrival_time: f64,
    round_idx: usize,
    prefix_len: usize,
    input_len: usize,
    output_len: usize,
    tool_wait_after_ms: f64,
}

/// Load the legacy raw session trace, whose `prefix_len` is what the source
/// reported rather than what the replay can supply.
pub(super) fn load(path: &str, max_sessions: Option<usize>) -> Result<SessionPlans> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open session trace: {path}"))?;

    let mut order: Vec<String> = Vec::new();
    let mut grouped: HashMap<String, Vec<SessionStep>> = HashMap::new();
    for row in reader.deserialize() {
        let raw: RawSessionRow = row.context("failed to parse session trace row")?;
        let step = SessionStep {
            request_id: v2::request_id(&raw.session_id, raw.round_idx),
            session_id: raw.session_id,
            arrival_time: raw.arrival_time,
            round_idx: raw.round_idx,
            prefix_len: raw.prefix_len,
            input_len: raw.input_len,
            output_len: raw.output_len,
            tool_wait_after_ms: raw.tool_wait_after_ms,
        };
        grouped
            .entry(step.session_id.clone())
            .or_insert_with(|| {
                order.push(step.session_id.clone());
                Vec::new()
            })
            .push(step);
    }

    let mut sessions: SessionPlans = Vec::with_capacity(order.len());
    for session_id in order {
        let mut steps = grouped.remove(&session_id).expect("session was recorded");
        steps.sort_by_key(|step| step.round_idx);
        sessions.push((session_id, steps));
    }
    Ok(apply_session_cap(sessions, max_sessions))
}

/// Load a canonical `session-execution-v2` trace.
///
/// Nothing is re-derived here. The file's own row order is the replay order,
/// its `request_id` is the identifier, and its `prefix_len` is already
/// guaranteed to exist by the time the round runs — which is exactly why a
/// simulator can read the same file and reach the same plan.
pub(super) fn load_execution_v2(path: &str, max_sessions: Option<usize>) -> Result<SessionPlans> {
    let rows = v2::load(path)?;

    let mut sessions: SessionPlans = Vec::new();
    for row in rows {
        let step = SessionStep {
            request_id: row.request_id,
            session_id: row.session_id,
            arrival_time: row.arrival_time_ms,
            round_idx: row.round_idx,
            prefix_len: row.prefix_len,
            input_len: row.input_len,
            output_len: row.output_len,
            tool_wait_after_ms: row.tool_wait_after_ms,
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

    fn write_temp(name: &str, contents: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!("tracelab-session-test-{name}.csv"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path.display().to_string()
    }

    #[test]
    fn legacy_cap_keeps_the_first_sessions_in_file_order_not_lexicographic_order() {
        let path = write_temp(
            "cap",
            "id,arrival_time,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session-2,0,0,0,10,4,0\n\
             session-10,5,0,0,10,4,0\n",
        );

        let sessions = load(&path, Some(1)).unwrap();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].0, "session-2");
    }

    #[test]
    fn legacy_loader_mints_the_canonical_request_id() {
        let path = write_temp(
            "request-id",
            "id,arrival_time,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             abc,0,0,0,10,4,0\n",
        );

        let sessions = load(&path, None).unwrap();

        assert_eq!(sessions[0].1[0].request_id, "abc_round_000000");
    }

    #[test]
    fn execution_v2_loader_preserves_file_order_and_ids() {
        let path = write_temp(
            "v2",
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             b_round_000000,b,0,0.000000,0,512,64,0.000000\n\
             b_round_000001,b,1,0.000000,576,128,64,0.000000\n\
             a_round_000000,a,0,250.000000,0,400,48,0.000000\n",
        );

        let sessions = load_execution_v2(&path, None).unwrap();

        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].0, "b");
        assert_eq!(sessions[1].0, "a");
        assert_eq!(sessions[0].1[1].request_id, "b_round_000001");
        assert_eq!(sessions[0].1[1].prefix_len, 576);
    }

    #[test]
    fn execution_v2_loader_rejects_a_non_canonical_file() {
        let path = write_temp(
            "v2-bad",
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             a_round_000000,a,0,0.000000,128,512,64,0.000000\n",
        );

        let error = load_execution_v2(&path, None).unwrap_err().to_string();

        assert!(error.contains("not a canonical"), "{error}");
    }
}
