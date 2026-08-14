//! The traces and corpus the harness measures against, built rather than shipped.
//!
//! Built, because every check's expected value is arithmetic over the trace's
//! own numbers, and a checked-in file drifts away from the arithmetic that was
//! written for it. A generated fixture cannot: the same constants that shape the
//! rows are the ones the claims are stated in.
//!
//! Deliberately boring. Fixed lengths, fixed spacing, no distributions — the
//! point is that the *client* is the only thing that varies.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use req_frontend::schema::session_execution_v2::{self as v2, request_id, ExecutionRow};

/// The shape of one fixture, so a check can state its expectation in these terms
/// rather than in a magic number.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub units: usize,
    pub rounds_per_unit: usize,
    pub input_len: usize,
    pub output_len: usize,
    /// Milliseconds between unit arrivals in the file itself. `--rate` rescales
    /// it, so this is only the file's own timeline.
    pub arrival_spacing_ms: f64,
}

impl Shape {
    pub fn steps(&self) -> usize {
        self.units * self.rounds_per_unit
    }
}

pub struct Fixtures {
    pub corpus: PathBuf,
    pub independent: PathBuf,
    pub sessions: PathBuf,
    pub independent_shape: Shape,
    pub session_shape: Shape,
}

/// Enough distinct text that the token pool is not a handful of repeated ids.
///
/// Generated from a fixed seed: the corpus must be the same on every machine, or
/// two runs of this harness are not comparable.
fn corpus_text(words: usize) -> String {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut text = String::with_capacity(words * 8);
    for index in 0..words {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Plain ASCII words: what the tokenizer does with them is its business,
        // and the checks never depend on which tokenizer is pointed at.
        let length = 3 + (state % 9) as usize;
        for offset in 0..length {
            text.push((b'a' + ((state >> (offset * 5)) % 26) as u8) as char);
        }
        text.push(if index % 12 == 11 { '\n' } else { ' ' });
    }
    text
}

pub fn build(directory: &Path) -> Result<Fixtures> {
    fs::create_dir_all(directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;

    let corpus = directory.join("corpus.txt");
    fs::write(&corpus, corpus_text(200_000))
        .with_context(|| format!("failed to write {}", corpus.display()))?;

    // One-shot requests, because `arrival_release_lag_ms` — the evidence that the
    // client released on time — is recorded only for them. A session's later
    // rounds are released by the session itself when the previous one finishes,
    // so there is no scheduled instant for them to be late against.
    let independent_shape = Shape {
        units: 1_200,
        rounds_per_unit: 1,
        input_len: 128,
        output_len: 8,
        arrival_spacing_ms: 10.0,
    };
    let independent = directory.join("independent.csv");
    write_independent(&independent, independent_shape)?;

    // Multi-round sessions, for the timing checks: they exercise the prefix path
    // as well, so prompt fidelity is checked on the rows where it can go wrong.
    let session_shape = Shape {
        units: 100,
        rounds_per_unit: 2,
        input_len: 96,
        output_len: 16,
        arrival_spacing_ms: 25.0,
    };
    let sessions = directory.join("sessions.csv");
    write_sessions(&sessions, session_shape)?;

    Ok(Fixtures {
        corpus,
        independent,
        sessions,
        independent_shape,
        session_shape,
    })
}

fn write_independent(path: &Path, shape: Shape) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    writer.write_record(["id", "input_len", "output_len", "arrival_time"])?;
    for index in 0..shape.units {
        writer.write_record([
            format!("request-{index:06}"),
            shape.input_len.to_string(),
            shape.output_len.to_string(),
            format!("{:.6}", index as f64 * shape.arrival_spacing_ms),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_sessions(path: &Path, shape: Shape) -> Result<()> {
    let mut rows = Vec::with_capacity(shape.steps());
    for unit in 0..shape.units {
        let session_id = format!("session-{unit:06}");
        let mut carried = 0usize;
        for round_idx in 0..shape.rounds_per_unit {
            rows.push(ExecutionRow {
                request_id: request_id(&session_id, round_idx),
                session_id: session_id.clone(),
                round_idx,
                arrival_time_ms: unit as f64 * shape.arrival_spacing_ms,
                prefix_len: carried,
                input_len: shape.input_len,
                output_len: shape.output_len,
                // No tool wait: it would put the client's own sleep inside every
                // end-to-end number and make the server's arithmetic
                // uncheckable.
                tool_wait_after_ms: 0.0,
            });
            carried += shape.input_len + shape.output_len;
        }
    }
    // The same validator a consumer runs. A harness that measures an invalid
    // trace is measuring the wrong thing.
    v2::validate(&rows).context("the generated session fixture is not canonical")?;
    v2::write_csv(path, &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_session_fixture_is_canonical_and_chains_its_prefixes() {
        let shape = Shape {
            units: 3,
            rounds_per_unit: 3,
            input_len: 10,
            output_len: 4,
            arrival_spacing_ms: 5.0,
        };
        let directory = std::env::temp_dir().join("req_frontend_selfcheck_fixture_test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("sessions.csv");

        write_sessions(&path, shape).expect("the fixture must pass the canonical validator");

        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), shape.steps() + 1, "header plus rows");
        // Round 2 reuses everything rounds 0 and 1 produced: 2 * (10 + 4).
        assert!(text.contains(",2,0.000000,28,10,4,"), "{text}");
    }

    #[test]
    fn the_corpus_is_the_same_bytes_on_every_machine() {
        assert_eq!(corpus_text(64), corpus_text(64));
        assert!(corpus_text(1_000).len() > 4_000);
    }
}
