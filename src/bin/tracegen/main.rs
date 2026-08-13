//! Turn a raw session trace into a canonical `session-execution-v2` trace.
//!
//! This is the seam the whole alignment story rests on. Upstream, a raw trace
//! reports what a coding agent actually did — including prefixes that only
//! existed in a conversation the published data does not contain. Downstream,
//! two very different consumers (a live replay against a real server, and a
//! discrete-event simulation) must agree on every integer. Resolving the first
//! into the second is a decision, so it happens once, here, and is recorded.
//!
//! What comes out is three files:
//!
//! - the canonical CSV, which both consumers read verbatim;
//! - a manifest, which records the source, the policy, and how much of the raw
//!   trace had to be folded to make it replayable;
//! - a normalized plan, which is what a differential test compares.

mod policy;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

use policy::{
    ContextChain, RawRound, SessionContextPolicy, MAJOR_COMPACTION_MIN_DROP_RATIO,
    MAJOR_COMPACTION_MIN_DROP_TOKENS,
};
use req_frontend::v2::{
    self, format_milliseconds, request_id, ExecutionRow, MILLISECOND_DECIMALS, SCHEMA_NAME,
};

/// Raw schema name accepted by `--source-schema`. Declared rather than sniffed:
/// a new raw format is a new declared schema, not a header guess.
const RAW_SCHEMA_SESSION_ROUNDS_V1: &str = "session-rounds-v1";

/// One row of the raw trace `artifacts/trace_facts/csv_export/convert.py` emits.
///
/// `id` names the *session* here, which is exactly the ambiguity v2 removes —
/// the same column name means a request in other tools. It is read under its
/// raw name and never propagated.
#[derive(Debug, Clone, Deserialize)]
struct RawRow {
    #[serde(alias = "session_id")]
    id: String,
    #[serde(default)]
    arrival_time: f64,
    round_idx: usize,
    prefix_len: usize,
    input_len: usize,
    output_len: usize,
    #[serde(default)]
    tool_wait_after_ms: f64,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Materialize a raw session trace into a canonical session-execution-v2 trace"
)]
struct Args {
    /// Raw session trace CSV.
    #[arg(long)]
    source: PathBuf,

    /// Declared raw schema. Never inferred from the header.
    #[arg(long, default_value = RAW_SCHEMA_SESSION_ROUNDS_V1)]
    source_schema: String,

    /// Context policy to resolve into the canonical trace. The generated file
    /// carries no policy switch; this choice is recorded in the manifest.
    #[arg(long, value_enum, default_value = "trace-reported")]
    policy: SessionContextPolicy,

    /// Output canonical CSV. The manifest and plan are written beside it.
    #[arg(long)]
    out: PathBuf,

    /// Keep only the first N sessions, in canonical arrival order.
    #[arg(long)]
    max_sessions: Option<usize>,
}

/// Everything needed to explain, and reproduce, one canonical trace.
#[derive(Debug, Serialize)]
struct Manifest {
    schema: &'static str,
    source_path: String,
    source_schema: String,
    source_sha256: String,
    source_bytes: u64,
    context_policy: &'static str,
    major_compaction_min_drop_tokens: usize,
    major_compaction_min_drop_ratio: f64,
    millisecond_decimals: usize,
    selection_rule: &'static str,
    max_sessions: Option<usize>,
    /// Source arrival subtracted from every session so this trace starts at its
    /// own origin. Recorded so a row can be traced back to the source timeline.
    arrival_origin_ms: f64,
    sessions: usize,
    rounds: usize,
    /// How much of the raw trace was not replayable as reported. These are the
    /// headline numbers, not a footnote: on a real coding-agent trace most
    /// sessions report a first-round prefix that the published data never
    /// contained, and every one of those tokens becomes prefill work here.
    folded_prefix_tokens: u64,
    folded_rounds: usize,
    folded_first_rounds: usize,
    major_compaction_rounds: usize,
    total_prompt_tokens: u64,
    total_prefix_tokens: u64,
    total_output_tokens: u64,
    planned_prefix_hit_rate: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.source_schema != RAW_SCHEMA_SESSION_ROUNDS_V1 {
        bail!(
            "unknown --source-schema {:?}; this build reads {RAW_SCHEMA_SESSION_ROUNDS_V1:?}",
            args.source_schema
        );
    }

    let raw_rows = read_raw(&args.source)?;
    let sessions = group_sessions(raw_rows)?;
    let selected = select_sessions(sessions, args.max_sessions);

    let mut rows: Vec<ExecutionRow> = Vec::new();
    let mut manifest = new_manifest(&args)?;

    // Rebase onto this trace's own origin. Source arrivals are offsets within
    // the whole dataset, so any subset would otherwise open with a lead-in that
    // belongs to sessions it does not contain — a consumer would idle through it
    // for no reason, and two subsets of one source would not be comparable.
    let arrival_origin_ms = selected
        .first()
        .map(|(_, rounds)| rounds[0].arrival_time)
        .unwrap_or(0.0);
    manifest.arrival_origin_ms = arrival_origin_ms;

    for (session_id, rounds) in &selected {
        let mut chain = ContextChain::new();
        let arrival_time_ms = rounds[0].arrival_time - arrival_origin_ms;
        for (round_idx, raw) in rounds.iter().enumerate() {
            let materialized = chain.materialize(
                RawRound {
                    prefix_len: raw.prefix_len,
                    input_len: raw.input_len,
                    output_len: raw.output_len,
                },
                args.policy,
            );

            if materialized.folded_tokens > 0 {
                manifest.folded_prefix_tokens += materialized.folded_tokens as u64;
                manifest.folded_rounds += 1;
                if round_idx == 0 {
                    manifest.folded_first_rounds += 1;
                }
            }
            if materialized.major_compaction {
                manifest.major_compaction_rounds += 1;
            }
            manifest.total_prompt_tokens += materialized.total_prompt_len() as u64;
            manifest.total_prefix_tokens += materialized.prefix_len as u64;
            manifest.total_output_tokens += materialized.output_len as u64;

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

    manifest.sessions = selected.len();
    manifest.rounds = rows.len();
    manifest.planned_prefix_hit_rate = if manifest.total_prompt_tokens == 0 {
        0.0
    } else {
        manifest.total_prefix_tokens as f64 / manifest.total_prompt_tokens as f64
    };

    // Validate what we just produced, with the same code a consumer will run.
    // A generator that trusts itself is how a "canonical" format stops being one.
    v2::validate(&rows).context("generated trace failed canonical validation")?;

    write_trace(&args.out, &rows)?;
    let manifest_path = sibling(&args.out, "manifest.json");
    let plan_path = sibling(&args.out, "plan.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    fs::write(
        &plan_path,
        serde_json::to_string_pretty(&v2::plan(&rows))? + "\n",
    )
    .with_context(|| format!("failed to write {}", plan_path.display()))?;

    eprintln!(
        "{SCHEMA_NAME} | policy={} sessions={} rounds={} prompt_tokens={} planned_prefix_hit_rate={:.4}",
        manifest.context_policy,
        manifest.sessions,
        manifest.rounds,
        manifest.total_prompt_tokens,
        manifest.planned_prefix_hit_rate,
    );
    eprintln!(
        "folded to fresh input | tokens={} rounds={} of which first rounds={} major_compactions={}",
        manifest.folded_prefix_tokens,
        manifest.folded_rounds,
        manifest.folded_first_rounds,
        manifest.major_compaction_rounds,
    );
    eprintln!(
        "wrote | {} {} {}",
        args.out.display(),
        manifest_path.display(),
        plan_path.display()
    );
    Ok(())
}

fn new_manifest(args: &Args) -> Result<Manifest> {
    let source_bytes = fs::metadata(&args.source)
        .with_context(|| format!("failed to stat {}", args.source.display()))?
        .len();
    Ok(Manifest {
        schema: SCHEMA_NAME,
        source_path: args.source.display().to_string(),
        source_schema: args.source_schema.clone(),
        source_sha256: sha256_file(&args.source)?,
        source_bytes,
        context_policy: args.policy.label(),
        major_compaction_min_drop_tokens: MAJOR_COMPACTION_MIN_DROP_TOKENS,
        major_compaction_min_drop_ratio: MAJOR_COMPACTION_MIN_DROP_RATIO,
        millisecond_decimals: MILLISECOND_DECIMALS,
        selection_rule: "first_n_by_arrival_then_source_order",
        max_sessions: args.max_sessions,
        arrival_origin_ms: 0.0,
        sessions: 0,
        rounds: 0,
        folded_prefix_tokens: 0,
        folded_rounds: 0,
        folded_first_rounds: 0,
        major_compaction_rounds: 0,
        total_prompt_tokens: 0,
        total_prefix_tokens: 0,
        total_output_tokens: 0,
        planned_prefix_hit_rate: 0.0,
    })
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
/// First-appearance order is the tie-break under equal arrivals (canonical rule
/// O2), so it is preserved rather than replaced by a sort on the identifier.
fn group_sessions(rows: Vec<RawRow>) -> Result<Vec<(String, Vec<RawRow>)>> {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: BTreeMap<String, Vec<RawRow>> = BTreeMap::new();
    for row in rows {
        if row.id.is_empty() {
            bail!("raw trace has a row with an empty session id");
        }
        let entry = grouped.entry(row.id.clone()).or_insert_with(|| {
            order.push(row.id.clone());
            Vec::new()
        });
        entry.push(row);
    }

    let mut sessions = Vec::with_capacity(order.len());
    for session_id in order {
        let mut rounds = grouped.remove(&session_id).expect("session was recorded");
        rounds.sort_by_key(|round| round.round_idx);
        let arrival = rounds[0].arrival_time;
        if !arrival.is_finite() || arrival < 0.0 {
            bail!("session {session_id:?} has arrival_time {arrival}, expected finite and non-negative");
        }
        if let Some(mismatch) = rounds.iter().find(|round| round.arrival_time != arrival) {
            bail!(
                "session {session_id:?} declares two arrival times ({arrival} and {}); \
                 a session arrives once",
                mismatch.arrival_time
            );
        }
        sessions.push((session_id, rounds));
    }
    Ok(sessions)
}

/// Keep the first `max` sessions in canonical order.
///
/// Canonical order is arrival first, then source appearance for equal arrivals,
/// which is also the order the rows are emitted in. Selecting a prefix of a
/// fixed schedule must follow that schedule — not the session identifier, whose
/// lexicographic order is unrelated to when the work arrives.
fn select_sessions(
    mut sessions: Vec<(String, Vec<RawRow>)>,
    max: Option<usize>,
) -> Vec<(String, Vec<RawRow>)> {
    sessions.sort_by(|left, right| {
        left.1[0]
            .arrival_time
            .partial_cmp(&right.1[0].arrival_time)
            .expect("arrival times were validated as finite")
    });
    if let Some(max) = max {
        sessions.truncate(max);
    }
    sessions
}

fn write_trace(path: &Path, rows: &[ExecutionRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    writer.write_record([
        "request_id",
        "session_id",
        "round_idx",
        "arrival_time_ms",
        "prefix_len",
        "input_len",
        "output_len",
        "tool_wait_after_ms",
    ])?;
    for row in rows {
        writer.write_record([
            row.request_id.as_str(),
            row.session_id.as_str(),
            &row.round_idx.to_string(),
            &format_milliseconds(row.arrival_time_ms),
            &row.prefix_len.to_string(),
            &row.input_len.to_string(),
            &row.output_len.to_string(),
            &format_milliseconds(row.tool_wait_after_ms),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn sibling(path: &Path, name: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "trace".to_string());
    path.with_file_name(format!("{stem}.{name}"))
}

/// Minimal SHA-256 so the manifest can pin its source without adding a
/// dependency to a crate that a public release ships.
fn sha256_file(path: &Path) -> Result<String> {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let bit_len = (bytes.len() as u64) * 8;
    let mut padded = bytes;
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in schedule.iter_mut().take(16).enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    Ok(state.iter().map(|word| format!("{word:08x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_row(id: &str, arrival_time: f64, round_idx: usize) -> RawRow {
        RawRow {
            id: id.to_string(),
            arrival_time,
            round_idx,
            prefix_len: 0,
            input_len: 10,
            output_len: 4,
            tool_wait_after_ms: 0.0,
        }
    }

    fn ids(sessions: &[(String, Vec<RawRow>)]) -> Vec<&str> {
        sessions.iter().map(|(id, _)| id.as_str()).collect()
    }

    #[test]
    fn grouping_preserves_first_appearance_and_orders_rounds() {
        let sessions = group_sessions(vec![
            raw_row("b", 0.0, 1),
            raw_row("a", 5.0, 0),
            raw_row("b", 0.0, 0),
        ])
        .unwrap();

        assert_eq!(ids(&sessions), ["b", "a"]);
        let round_indices: Vec<usize> = sessions[0].1.iter().map(|row| row.round_idx).collect();
        assert_eq!(round_indices, [0, 1]);
    }

    /// A session arrives once. The loader this replaced kept a per-round arrival
    /// column and silently used only the first round's value, so a trace that
    /// disagreed with itself replayed as though it did not.
    #[test]
    fn grouping_rejects_a_session_that_declares_two_arrival_times() {
        let error = group_sessions(vec![raw_row("a", 0.0, 0), raw_row("a", 250.0, 1)])
            .unwrap_err()
            .to_string();

        assert!(error.contains("declares two arrival times"), "{error}");
    }

    #[test]
    fn grouping_rejects_an_empty_session_id() {
        let error = group_sessions(vec![raw_row("", 0.0, 0)])
            .unwrap_err()
            .to_string();

        assert!(error.contains("empty session id"), "{error}");
    }

    #[test]
    fn selection_keeps_the_earliest_arrivals_not_the_first_seen() {
        let sessions =
            group_sessions(vec![raw_row("late", 900.0, 0), raw_row("early", 5.0, 0)]).unwrap();

        let selected = select_sessions(sessions, Some(1));

        assert_eq!(ids(&selected), ["early"]);
    }

    /// Canonical rule O2: equal arrivals fall back to source appearance order,
    /// which is what makes a measured replay and a simulated run agree on which
    /// session entered first.
    #[test]
    fn selection_breaks_equal_arrivals_by_source_order() {
        let sessions = group_sessions(vec![
            raw_row("second", 100.0, 0),
            raw_row("first", 100.0, 0),
            raw_row("third", 100.0, 0),
        ])
        .unwrap();

        let selected = select_sessions(sessions, Some(2));

        assert_eq!(ids(&selected), ["second", "first"]);
    }
}
