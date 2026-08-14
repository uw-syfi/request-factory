//! Produce a canonical `session-execution-v2` trace, whichever way it was made.
//!
//! Two consumers — a live replay against a real server, and a discrete-event
//! simulation — must agree on every integer in the file they read. So the file
//! is produced once, here, and everything true of *every* canonical trace lives
//! in this one place: validating what was produced, writing the CSV, deriving
//! the plan, and computing the manifest's totals.
//!
//! Where the rows came from is the generator's business. `generator/` holds the
//! registry; a generator returns rows plus a record of how it made them, and
//! this file does the rest. That is what lets a new trace category be a new file
//! in one directory rather than a new binary.
//!
//! What comes out is three files:
//!
//! - the canonical CSV, which both consumers read verbatim;
//! - a manifest, which records the generator, its parameters, and totals derived
//!   from the rows actually emitted;
//! - a normalized plan, which is what a differential test compares.

mod arrivals;
mod generator;
mod policy;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

use generator::Registry;
use req_frontend::schema::session_execution_v2 as v2;
use req_frontend::schema::session_execution_v2::{
    format_milliseconds, ExecutionRow, MILLISECOND_DECIMALS, SCHEMA_NAME,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate a canonical session-execution-v2 trace",
    subcommand_help_heading = "Generators"
)]
struct Args {
    #[command(subcommand)]
    generator: Registry,
}

/// Everything needed to explain, and reproduce, one canonical trace.
///
/// The totals are derived from the emitted rows rather than accumulated by the
/// generator, so the manifest describes the file rather than the generator's own
/// bookkeeping. The two used to be able to drift.
#[derive(Debug, Serialize)]
struct Manifest {
    schema: &'static str,
    /// Which entry in the registry produced this. Reading `parameters` without
    /// knowing this is reading a blob whose keys mean whatever they meant to
    /// some generator.
    generator: &'static str,
    millisecond_decimals: usize,
    sessions: usize,
    rounds: usize,
    total_prompt_tokens: u64,
    total_prefix_tokens: u64,
    total_output_tokens: u64,
    planned_prefix_hit_rate: f64,
    /// The generator's own parameters and statistics, verbatim. This is the half
    /// of reproducibility this file cannot know anything about.
    parameters: serde_json::Value,
}

impl Manifest {
    fn derive(
        generator: &'static str,
        rows: &[ExecutionRow],
        parameters: serde_json::Value,
    ) -> Self {
        let mut total_prompt_tokens = 0u64;
        let mut total_prefix_tokens = 0u64;
        let mut total_output_tokens = 0u64;
        let mut sessions: BTreeSet<&str> = BTreeSet::new();
        for row in rows {
            sessions.insert(row.session_id.as_str());
            total_prompt_tokens += (row.prefix_len + row.input_len) as u64;
            total_prefix_tokens += row.prefix_len as u64;
            total_output_tokens += row.output_len as u64;
        }
        Self {
            schema: SCHEMA_NAME,
            generator,
            millisecond_decimals: MILLISECOND_DECIMALS,
            sessions: sessions.len(),
            rounds: rows.len(),
            total_prompt_tokens,
            total_prefix_tokens,
            total_output_tokens,
            planned_prefix_hit_rate: if total_prompt_tokens == 0 {
                0.0
            } else {
                total_prefix_tokens as f64 / total_prompt_tokens as f64
            },
            parameters,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let generator = args.generator.selected();

    let generated = generator.generate()?;
    let manifest = Manifest::derive(generator.name(), &generated.rows, generated.record);

    // Validate what we just produced, with the same code a consumer will run.
    // A generator that trusts itself is how a "canonical" format stops being one.
    v2::validate(&generated.rows).context("generated trace failed canonical validation")?;

    let out = generator.out();
    write_trace(out, &generated.rows)?;
    let manifest_path = sibling(out, "manifest.json");
    let plan_path = sibling(out, "plan.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    fs::write(
        &plan_path,
        serde_json::to_string_pretty(&v2::plan(&generated.rows))? + "\n",
    )
    .with_context(|| format!("failed to write {}", plan_path.display()))?;

    eprintln!(
        "{SCHEMA_NAME} | generator={} sessions={} rounds={} prompt_tokens={} planned_prefix_hit_rate={:.4}",
        manifest.generator,
        manifest.sessions,
        manifest.rounds,
        manifest.total_prompt_tokens,
        manifest.planned_prefix_hit_rate,
    );
    eprintln!(
        "wrote | {} {} {}",
        out.display(),
        manifest_path.display(),
        plan_path.display()
    );
    Ok(())
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

/// Minimal SHA-256 so a manifest can pin its source without adding a
/// dependency to a crate that a public release ships.
pub(crate) fn sha256_file(path: &Path) -> Result<String> {
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

    fn row(session_id: &str, round_idx: usize, prefix_len: usize) -> ExecutionRow {
        ExecutionRow {
            request_id: v2::request_id(session_id, round_idx),
            session_id: session_id.to_string(),
            round_idx,
            arrival_time_ms: 0.0,
            prefix_len,
            input_len: 10,
            output_len: 4,
            tool_wait_after_ms: 0.0,
        }
    }

    /// The reason the totals moved out of the generators: they are a property of
    /// the file, so they are counted from the file.
    #[test]
    fn the_manifests_totals_describe_the_rows_that_were_emitted() {
        let rows = vec![row("a", 0, 0), row("a", 1, 14), row("b", 0, 0)];

        let manifest = Manifest::derive("test", &rows, serde_json::json!({}));

        assert_eq!(manifest.sessions, 2);
        assert_eq!(manifest.rounds, 3);
        assert_eq!(manifest.total_prompt_tokens, 10 + 24 + 10);
        assert_eq!(manifest.total_prefix_tokens, 14);
        assert_eq!(manifest.total_output_tokens, 12);
        assert!((manifest.planned_prefix_hit_rate - 14.0 / 44.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_trace_reports_a_zero_hit_rate_rather_than_dividing_by_zero() {
        let manifest = Manifest::derive("test", &[], serde_json::json!({}));

        assert_eq!(manifest.planned_prefix_hit_rate, 0.0);
    }

    /// The generator's record goes into the manifest untouched. Nothing here
    /// interprets it, so nothing here can quietly drop a knob out of it.
    #[test]
    fn the_generators_record_is_carried_verbatim() {
        let record = serde_json::json!({"seed": 7, "nested": {"kept": true}});

        let manifest = Manifest::derive("test", &[], record.clone());

        assert_eq!(manifest.parameters, record);
    }

    #[test]
    fn siblings_are_named_after_the_trace_they_describe() {
        assert_eq!(
            sibling(Path::new("out/execution.csv"), "manifest.json"),
            PathBuf::from("out/execution.manifest.json")
        );
    }
}
