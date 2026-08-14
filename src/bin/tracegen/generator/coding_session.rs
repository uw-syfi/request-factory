//! Materialize a recorded coding-agent trace into canonical form.
//!
//! This is the seam the whole alignment story rests on. Upstream, a raw trace
//! reports what a coding agent actually did — including prefixes that only
//! existed in a conversation the published data does not contain. Downstream,
//! two very different consumers (a live replay against a real server, and a
//! discrete-event simulation) must agree on every integer. Resolving the first
//! into the second is a decision, so it happens once, here, and is recorded.
//!
//! The source has no timeline at all, so this generator invents one. Every knob
//! that shapes it is recorded, because a trace whose arrivals cannot be
//! reproduced from its manifest is not reproducible.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use req_frontend::schema::format::text_generation::session::{request_id, ExecutionRow};
use serde::{Deserialize, Serialize};

use super::{Generated, Generator};
use crate::arrivals::{self, ArrivalPattern, Rng, SessionOrder};
use crate::policy::{
    ContextChain, RawRound, SessionContextPolicy, MAJOR_COMPACTION_MIN_DROP_RATIO,
    MAJOR_COMPACTION_MIN_DROP_TOKENS,
};

/// Raw schema name accepted by `--source-schema`. Declared rather than sniffed:
/// a new raw format is a new declared schema, not a header guess.
const RAW_SCHEMA_SESSION_ROUNDS_V2: &str = "session-rounds-v2";

/// One row of the raw trace TraceLab's
/// `artifacts/trace_facts/csv_export/convert.py` exports.
///
/// It carries no arrival time: the corpus has none, so this tool invents the
/// timeline. Every field here is something the source actually observed.
///
/// Unknown columns are rejected rather than ignored. A `session-rounds-v1` file
/// still has every column v2 needs, so without this it would parse cleanly and
/// its recorded `arrival_time` would be silently replaced by a synthetic one —
/// the exact failure mode of quietly accepting a file from the wrong schema.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRow {
    session_id: String,
    round_idx: usize,
    input_len: usize,
    output_len: usize,
    prefix_len: usize,
    tool_wait_after_ms: f64,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Raw session trace CSV.
    #[arg(long)]
    source: PathBuf,

    /// Declared raw schema. Never inferred from the header.
    #[arg(long, default_value = RAW_SCHEMA_SESSION_ROUNDS_V2)]
    source_schema: String,

    /// Context policy to resolve into the canonical trace. The generated file
    /// carries no policy switch; this choice is recorded in the manifest.
    #[arg(long, value_enum, default_value = "trace-reported")]
    policy: SessionContextPolicy,

    /// Output canonical CSV. The manifest and plan are written beside it.
    #[arg(long)]
    out: PathBuf,

    /// Keep only the first N sessions, in the emitted session order.
    #[arg(long)]
    max_sessions: Option<usize>,

    /// Session order before arrivals are assigned.
    #[arg(long, value_enum, default_value = "source")]
    session_order: SessionOrder,

    /// Synthetic session arrival rate, in sessions per second. The source has
    /// no arrival times, so this timeline is invented here and recorded.
    #[arg(long, default_value_t = 1.0)]
    arrival_rate: f64,

    /// Synthetic session arrival process.
    #[arg(long, value_enum, default_value = "poisson")]
    arrival_pattern: ArrivalPattern,

    /// Seed for session shuffling and Poisson arrivals. Same seed, same trace.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// Everything needed to explain, and reproduce, this trace.
#[derive(Debug, Serialize)]
struct Record {
    source_path: String,
    source_schema: String,
    source_sha256: String,
    source_bytes: u64,
    context_policy: &'static str,
    major_compaction_min_drop_tokens: usize,
    major_compaction_min_drop_ratio: f64,
    selection_rule: &'static str,
    max_sessions: Option<usize>,
    /// How the timeline was invented. The source has no arrival times, so
    /// without these four fields a trace cannot be reproduced from its source.
    session_order: &'static str,
    arrival_rate_per_second: f64,
    arrival_pattern: &'static str,
    seed: u64,
    /// How much of the raw trace was not replayable as reported. These are the
    /// headline numbers, not a footnote: on a real coding-agent trace most
    /// sessions report a first-round prefix that the published data never
    /// contained, and every one of those tokens becomes prefill work here.
    folded_prefix_tokens: u64,
    folded_rounds: usize,
    folded_first_rounds: usize,
    major_compaction_rounds: usize,
}

impl Generator for Args {
    fn name(&self) -> &'static str {
        "coding-session"
    }

    fn out(&self) -> &Path {
        &self.out
    }

    fn generate(&self) -> Result<Generated> {
        if self.source_schema != RAW_SCHEMA_SESSION_ROUNDS_V2 {
            bail!(
                "unknown --source-schema {:?}; this build reads {RAW_SCHEMA_SESSION_ROUNDS_V2:?}",
                self.source_schema
            );
        }
        if !(self.arrival_rate.is_finite() && self.arrival_rate > 0.0) {
            bail!(
                "--arrival-rate must be finite and positive, got {}",
                self.arrival_rate
            );
        }

        let raw_rows = read_raw(&self.source)?;
        let sessions = group_sessions(raw_rows)?;

        // One RNG drives both the permutation and the gaps, so a seed names the
        // whole timeline rather than half of it.
        let mut rng = Rng::new(self.seed);
        let selected = select_sessions(sessions, self.max_sessions, self.session_order, &mut rng);
        // Arrivals are drawn for the sessions that survived selection, so capping
        // shortens the trace without changing the offered rate.
        let arrivals = arrivals::synthesize(
            &mut rng,
            selected.len(),
            self.arrival_rate,
            self.arrival_pattern,
        );

        let mut rows: Vec<ExecutionRow> = Vec::new();
        let mut folded_prefix_tokens = 0u64;
        let mut folded_rounds = 0usize;
        let mut folded_first_rounds = 0usize;
        let mut major_compaction_rounds = 0usize;

        for ((session_id, rounds), &arrival_time_ms) in selected.iter().zip(&arrivals) {
            let mut chain = ContextChain::new();
            for (round_idx, raw) in rounds.iter().enumerate() {
                let materialized = chain.materialize(
                    RawRound {
                        prefix_len: raw.prefix_len,
                        input_len: raw.input_len,
                        output_len: raw.output_len,
                    },
                    self.policy,
                );

                if materialized.folded_tokens > 0 {
                    folded_prefix_tokens += materialized.folded_tokens as u64;
                    folded_rounds += 1;
                    if round_idx == 0 {
                        folded_first_rounds += 1;
                    }
                }
                if materialized.major_compaction {
                    major_compaction_rounds += 1;
                }

                rows.push(ExecutionRow {
                    request_id: request_id(session_id, round_idx),
                    session_id: session_id.clone(),
                    round_idx,
                    arrival_time_ms,
                    prefix_len: materialized.prefix_len,
                    input_len: materialized.input_len,
                    output_len: materialized.output_len,
                    tool_wait_after_ms: raw.tool_wait_after_ms,
                });
            }
        }

        let source_bytes = std::fs::metadata(&self.source)
            .with_context(|| format!("failed to stat {}", self.source.display()))?
            .len();
        let record = Record {
            source_path: self.source.display().to_string(),
            source_schema: self.source_schema.clone(),
            source_sha256: crate::sha256_file(&self.source)?,
            source_bytes,
            context_policy: self.policy.label(),
            major_compaction_min_drop_tokens: MAJOR_COMPACTION_MIN_DROP_TOKENS,
            major_compaction_min_drop_ratio: MAJOR_COMPACTION_MIN_DROP_RATIO,
            selection_rule: "first_n_in_emitted_session_order",
            max_sessions: self.max_sessions,
            session_order: self.session_order.label(),
            arrival_rate_per_second: self.arrival_rate,
            arrival_pattern: self.arrival_pattern.label(),
            seed: self.seed,
            folded_prefix_tokens,
            folded_rounds,
            folded_first_rounds,
            major_compaction_rounds,
        };
        eprintln!(
            "folded to fresh input | tokens={folded_prefix_tokens} rounds={folded_rounds} \
             of which first rounds={folded_first_rounds} \
             major_compactions={major_compaction_rounds}",
        );
        Ok(Generated {
            rows,
            record: serde_json::to_value(record)?,
        })
    }
}

fn read_raw(path: &Path) -> Result<Vec<RawRow>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open raw trace: {}", path.display()))?;
    let mut rows = Vec::new();
    for record in reader.deserialize() {
        rows.push(record.context("failed to parse a raw session row")?);
    }
    if rows.is_empty() {
        bail!("raw trace {} has no rows", path.display());
    }
    Ok(rows)
}

/// Group raw rows into sessions, preserving first-appearance order and sorting
/// each session's rounds by the raw `round_idx`.
///
/// First-appearance order is the source's own order, which is the default
/// emission order, so it is preserved rather than replaced by a sort on the
/// identifier — that is lexicographic and means nothing here.
fn group_sessions(rows: Vec<RawRow>) -> Result<Vec<(String, Vec<RawRow>)>> {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: BTreeMap<String, Vec<RawRow>> = BTreeMap::new();
    for row in rows {
        if row.session_id.is_empty() {
            bail!("raw trace has a row with an empty session id");
        }
        let entry = grouped.entry(row.session_id.clone()).or_insert_with(|| {
            order.push(row.session_id.clone());
            Vec::new()
        });
        entry.push(row);
    }

    let mut sessions = Vec::with_capacity(order.len());
    for session_id in order {
        let mut rounds = grouped.remove(&session_id).expect("session was recorded");
        rounds.sort_by_key(|round| round.round_idx);
        sessions.push((session_id, rounds));
    }
    Ok(sessions)
}

/// Order the sessions, then keep the first `max`.
///
/// Selection happens before arrivals are drawn, so a cap takes a prefix of the
/// session order rather than the earliest slice of a timeline. That keeps the
/// offered rate of a capped trace equal to the rate of the full one instead of
/// silently compressing it.
fn select_sessions(
    mut sessions: Vec<(String, Vec<RawRow>)>,
    max: Option<usize>,
    order: SessionOrder,
    rng: &mut Rng,
) -> Vec<(String, Vec<RawRow>)> {
    if order == SessionOrder::Shuffle {
        rng.shuffle(&mut sessions);
    }
    if let Some(max) = max {
        sessions.truncate(max);
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_row(session_id: &str, round_idx: usize) -> RawRow {
        RawRow {
            session_id: session_id.to_string(),
            round_idx,
            input_len: 10,
            output_len: 4,
            prefix_len: 0,
            tool_wait_after_ms: 0.0,
        }
    }

    fn ids(sessions: &[(String, Vec<RawRow>)]) -> Vec<&str> {
        sessions.iter().map(|(id, _)| id.as_str()).collect()
    }

    #[test]
    fn grouping_preserves_first_appearance_and_orders_rounds() {
        let sessions =
            group_sessions(vec![raw_row("b", 1), raw_row("a", 0), raw_row("b", 0)]).unwrap();

        assert_eq!(ids(&sessions), ["b", "a"]);
        let round_indices: Vec<usize> = sessions[0].1.iter().map(|row| row.round_idx).collect();
        assert_eq!(round_indices, [0, 1]);
    }

    #[test]
    fn grouping_rejects_an_empty_session_id() {
        let error = group_sessions(vec![raw_row("", 0)])
            .unwrap_err()
            .to_string();

        assert!(error.contains("empty session id"), "{error}");
    }

    #[test]
    fn selection_in_source_order_takes_a_prefix_of_the_file() {
        let sessions =
            group_sessions(vec![raw_row("c", 0), raw_row("a", 0), raw_row("b", 0)]).unwrap();

        let selected = select_sessions(sessions, Some(2), SessionOrder::Source, &mut Rng::new(0));

        assert_eq!(ids(&selected), ["c", "a"]);
    }

    /// Shuffling happens before truncation, so a cap samples across the whole
    /// file rather than re-cutting the same prefix under a different name.
    #[test]
    fn shuffled_selection_samples_the_whole_file_and_is_seed_reproducible() {
        let rows: Vec<RawRow> = (0..32)
            .map(|index| raw_row(&format!("s{index:02}"), 0))
            .collect();

        let take = |seed: u64| {
            let sessions = group_sessions(rows.clone()).unwrap();
            let selected = select_sessions(
                sessions,
                Some(4),
                SessionOrder::Shuffle,
                &mut Rng::new(seed),
            );
            ids(&selected)
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
        };

        let shuffled = take(1);
        assert_eq!(shuffled.len(), 4);
        assert_eq!(shuffled, take(1), "the same seed must select the same four");
        assert_ne!(
            shuffled,
            ["s00", "s01", "s02", "s03"],
            "a shuffle that returns the source prefix is not shuffling"
        );
    }

    /// The whole point of drawing arrivals after selection: a capped trace has
    /// the same offered rate as the full one, not a compressed slice of it.
    #[test]
    fn capping_does_not_change_the_offered_rate() {
        let mut rng = Rng::new(0);
        let capped = arrivals::synthesize(&mut rng, 10, 2.0, ArrivalPattern::Constant);

        assert_eq!(capped.last().copied(), Some(4500.0));
        assert!(capped.windows(2).all(|pair| pair[1] - pair[0] == 500.0));
    }
}
