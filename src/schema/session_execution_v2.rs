//! The `session-execution-v2` canonical trace: the one artifact a measured
//! replay and a simulated run are allowed to disagree about nothing in.
//!
//! A v2 file is *already materialized*. Its `prefix_len` is guaranteed to exist
//! in the replayed conversation by the time the round runs, so a consumer needs
//! no context policy and no reinterpretation — it reads two integers and builds
//! `previous_context[..prefix_len] + fresh(input_len)`. Everything that used to
//! be a runtime knob now happened upstream and is recorded in the manifest.
//!
//! The layout is strict on purpose. Row order carries meaning (it is the
//! release order, and the tie-break under equal arrivals), so it is validated
//! rather than re-derived, and a consumer must never re-sort by identifier.

use std::collections::HashSet;
use std::fmt::Write as _;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Schema name recorded in the manifest and named on the command line. Consumers
/// select it explicitly; no consumer sniffs headers to discover it.
pub const SCHEMA_NAME: &str = "session-execution-v2";

/// Decimal places used for every millisecond field in a canonical file.
///
/// Fixed formatting is part of the canonical form, not a printing preference.
/// Rate scaling multiplies arrivals, and a session's rows must still be
/// byte-identical afterwards; and VibeSim converts milliseconds to its internal
/// clock by truncation, so two spellings of one instant can land on different
/// ticks.
/// The canonical column order, which is also the file's header.
///
/// Declared rather than derived from [`ExecutionRow`]'s field order: the order
/// is part of the format, and a consumer checking a header must be able to state
/// what it expects without constructing a row.
pub const COLUMNS: &[&str] = &[
    "request_id",
    "session_id",
    "round_idx",
    "arrival_time_ms",
    "prefix_len",
    "input_len",
    "output_len",
    "tool_wait_after_ms",
];

pub const MILLISECOND_DECIMALS: usize = 6;

pub fn format_milliseconds(value: f64) -> String {
    let mut formatted = String::new();
    let _ = write!(formatted, "{:.*}", MILLISECOND_DECIMALS, value);
    formatted
}

/// One round of one session, fully materialized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRow {
    /// Globally unique, opaque. Carried by the file rather than derived, so a
    /// consumer's identifiers survive any later change to how they are minted.
    pub request_id: String,
    pub session_id: String,
    pub round_idx: usize,
    /// Session start offset from the beginning of *this* trace. Identical on
    /// every row of one session; only round 0 consumes it, later rounds release
    /// from their predecessor's completion.
    pub arrival_time_ms: f64,
    /// Cache-eligible leading prefix copied from the previous realized context.
    /// Eligible, never a guaranteed hit.
    pub prefix_len: usize,
    /// Fresh tokens appended after that prefix. Zero is valid when `prefix_len`
    /// is positive.
    pub input_len: usize,
    pub output_len: usize,
    pub tool_wait_after_ms: f64,
}

impl ExecutionRow {
    pub fn total_prompt_len(&self) -> usize {
        self.prefix_len + self.input_len
    }
}

/// Mint the canonical request id for a round.
///
/// `session_` namespaces the id by the frontend that produced it, matching what
/// the independent frontend already does with `independent_{id}`. The corpus's
/// own name for the session is left alone in the `session_id` column — this
/// prefix is this client's, not the dataset's, which matters because published
/// sessions are often bare integers that say nothing about what they identify.
pub fn request_id(session_id: &str, round_idx: usize) -> String {
    format!("session_{session_id}_round_{round_idx:06}")
}

/// Read a canonical file, rejecting anything that is not canonical.
///
/// Structural corruption fails here rather than surfacing later as a puzzling
/// replay: the whole value of this format is that two independent consumers can
/// trust it without re-deriving anything.
pub fn load(path: &str) -> Result<Vec<ExecutionRow>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {SCHEMA_NAME} trace: {path}"))?;
    let mut rows: Vec<ExecutionRow> = Vec::new();
    for record in reader.deserialize() {
        rows.push(record.context("failed to parse a session-execution-v2 row")?);
    }
    validate(&rows).with_context(|| format!("{path} is not a canonical {SCHEMA_NAME} trace"))?;
    Ok(rows)
}

/// Validate the canonical layout and the per-row invariants.
pub fn validate(rows: &[ExecutionRow]) -> Result<()> {
    if rows.is_empty() {
        bail!("trace has no rows");
    }

    // A canonical trace is self-contained: its clock starts when it does. A file
    // carved out of a larger source keeps that source's absolute offsets unless
    // it is rebased, and a consumer then idles through a lead-in that belongs to
    // sessions the file does not contain.
    if rows[0].arrival_time_ms != 0.0 {
        bail!(
            "line 2: the earliest arrival_time_ms is {}, expected 0 — a canonical trace \
             is rebased so its own first session arrives at the origin",
            rows[0].arrival_time_ms
        );
    }

    let mut seen_request_ids: HashSet<&str> = HashSet::new();
    let mut finished_sessions: HashSet<&str> = HashSet::new();
    let mut current_session: Option<&str> = None;
    let mut current_arrival = f64::NEG_INFINITY;
    let mut previous_block_arrival = f64::NEG_INFINITY;
    let mut expected_round_idx = 0usize;

    for (index, row) in rows.iter().enumerate() {
        let line = index + 2; // header occupies line 1
        if row.request_id.is_empty() {
            bail!("line {line}: request_id is empty");
        }
        if row.session_id.is_empty() {
            bail!("line {line}: session_id is empty");
        }
        if !seen_request_ids.insert(row.request_id.as_str()) {
            bail!("line {line}: duplicate request_id {:?}", row.request_id);
        }
        if !row.arrival_time_ms.is_finite() || row.arrival_time_ms < 0.0 {
            bail!(
                "line {line}: arrival_time_ms must be finite and non-negative, got {}",
                row.arrival_time_ms
            );
        }
        if !row.tool_wait_after_ms.is_finite() || row.tool_wait_after_ms < 0.0 {
            bail!(
                "line {line}: tool_wait_after_ms must be finite and non-negative, got {}",
                row.tool_wait_after_ms
            );
        }
        if row.output_len == 0 {
            bail!("line {line}: output_len must be positive");
        }
        if row.prefix_len == 0 && row.input_len == 0 {
            bail!("line {line}: a round with no prefix and no fresh input is an empty prompt");
        }

        let starts_new_session = current_session != Some(row.session_id.as_str());
        if starts_new_session {
            if let Some(previous) = current_session {
                finished_sessions.insert(previous);
                previous_block_arrival = current_arrival;
            }
            if finished_sessions.contains(row.session_id.as_str()) {
                bail!(
                    "line {line}: session {:?} resumes after another session; \
                     all rows of one session must be contiguous",
                    row.session_id
                );
            }
            if row.arrival_time_ms < previous_block_arrival {
                bail!(
                    "line {line}: session blocks must be ordered by nondecreasing \
                     arrival_time_ms ({} follows {})",
                    row.arrival_time_ms,
                    previous_block_arrival
                );
            }
            if row.round_idx != 0 {
                bail!(
                    "line {line}: session {:?} starts at round_idx {}, expected 0",
                    row.session_id,
                    row.round_idx
                );
            }
            if row.prefix_len != 0 {
                bail!(
                    "line {line}: round 0 of session {:?} declares prefix_len {}, but a \
                     session's first round has no previous context to copy from",
                    row.session_id,
                    row.prefix_len
                );
            }
            current_session = Some(row.session_id.as_str());
            current_arrival = row.arrival_time_ms;
            expected_round_idx = 0;
        } else {
            expected_round_idx += 1;
            if row.round_idx != expected_round_idx {
                bail!(
                    "line {line}: session {:?} jumps to round_idx {}, expected {}",
                    row.session_id,
                    row.round_idx,
                    expected_round_idx
                );
            }
            if row.arrival_time_ms != current_arrival {
                bail!(
                    "line {line}: session {:?} declares arrival_time_ms {} but its first row \
                     declared {}; the value must be identical on every row of a session",
                    row.session_id,
                    row.arrival_time_ms,
                    current_arrival
                );
            }
        }

        if row.request_id != request_id(&row.session_id, row.round_idx) {
            bail!(
                "line {line}: request_id {:?} does not match the canonical form {:?}",
                row.request_id,
                request_id(&row.session_id, row.round_idx)
            );
        }
    }

    Ok(())
}

/// One row of the normalized plan both consumers export for differential
/// comparison. Deliberately not the CSV row: it adds the resolved causal link,
/// which is what a scheduler acts on and where the two systems could silently
/// disagree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRow {
    pub request_id: String,
    pub session_id: String,
    pub round_idx: usize,
    pub session_arrival_time_ms: String,
    pub predecessor_request_id: Option<String>,
    pub prefix_len: usize,
    pub input_len: usize,
    pub output_len: usize,
    pub tool_wait_after_ms: String,
}

/// Project a canonical trace into its normalized plan, preserving row order.
pub fn plan(rows: &[ExecutionRow]) -> Vec<PlanRow> {
    rows.iter()
        .map(|row| PlanRow {
            request_id: row.request_id.clone(),
            session_id: row.session_id.clone(),
            round_idx: row.round_idx,
            session_arrival_time_ms: format_milliseconds(row.arrival_time_ms),
            predecessor_request_id: row
                .round_idx
                .checked_sub(1)
                .map(|previous| request_id(&row.session_id, previous)),
            prefix_len: row.prefix_len,
            input_len: row.input_len,
            output_len: row.output_len,
            tool_wait_after_ms: format_milliseconds(row.tool_wait_after_ms),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        session: &str,
        round_idx: usize,
        arrival: f64,
        prefix: usize,
        input: usize,
    ) -> ExecutionRow {
        ExecutionRow {
            request_id: request_id(session, round_idx),
            session_id: session.to_string(),
            round_idx,
            arrival_time_ms: arrival,
            prefix_len: prefix,
            input_len: input,
            output_len: 4,
            tool_wait_after_ms: 0.0,
        }
    }

    #[test]
    fn accepts_a_canonical_two_session_trace() {
        let rows = vec![
            row("a", 0, 0.0, 0, 512),
            row("a", 1, 0.0, 516, 128),
            row("b", 0, 250.0, 0, 400),
        ];
        validate(&rows).unwrap();
    }

    #[test]
    fn rejects_a_trace_that_does_not_start_at_the_origin() {
        let rows = vec![row("a", 0, 1_173_758.075312, 0, 512)];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("rebased"), "{error}");
    }

    #[test]
    fn rejects_a_nonzero_prefix_on_a_first_round() {
        let rows = vec![row("a", 0, 0.0, 128, 512)];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("no previous context"), "{error}");
    }

    #[test]
    fn rejects_interleaved_sessions() {
        let rows = vec![
            row("a", 0, 0.0, 0, 512),
            row("b", 0, 10.0, 0, 400),
            row("a", 1, 0.0, 516, 128),
        ];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("contiguous"), "{error}");
    }

    #[test]
    fn rejects_a_gap_in_round_indices() {
        let rows = vec![row("a", 0, 0.0, 0, 512), row("a", 2, 0.0, 516, 128)];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("expected 1"), "{error}");
    }

    #[test]
    fn rejects_disagreeing_arrivals_inside_one_session() {
        let rows = vec![row("a", 0, 0.0, 0, 512), row("a", 1, 5.0, 516, 128)];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("identical on every row"), "{error}");
    }

    #[test]
    fn rejects_session_blocks_out_of_arrival_order() {
        let rows = vec![
            row("a", 0, 0.0, 0, 512),
            row("b", 0, -0.0, 0, 400),
            row("c", 0, 100.0, 0, 1),
            row("d", 0, 10.0, 0, 1),
        ];
        let error = validate(&rows).unwrap_err().to_string();
        assert!(error.contains("nondecreasing"), "{error}");
    }

    #[test]
    fn accepts_zero_fresh_input_when_a_prefix_exists() {
        let rows = vec![
            row("a", 0, 0.0, 0, 512),
            ExecutionRow {
                request_id: request_id("a", 1),
                session_id: "a".to_string(),
                round_idx: 1,
                arrival_time_ms: 0.0,
                prefix_len: 516,
                input_len: 0,
                output_len: 4,
                tool_wait_after_ms: 0.0,
            },
        ];
        validate(&rows).unwrap();
    }

    #[test]
    fn plan_links_each_round_to_its_predecessor() {
        let rows = vec![
            row("a", 0, 0.0, 0, 512),
            row("a", 1, 0.0, 516, 128),
            row("b", 0, 250.0, 0, 400),
        ];

        let plan = plan(&rows);

        assert_eq!(plan[0].predecessor_request_id, None);
        assert_eq!(
            plan[1].predecessor_request_id.as_deref(),
            Some("session_a_round_000000")
        );
        assert_eq!(plan[2].predecessor_request_id, None);
        assert_eq!(plan[2].session_arrival_time_ms, "250.000000");
    }
}
