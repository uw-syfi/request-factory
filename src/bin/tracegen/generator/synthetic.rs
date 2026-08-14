//! A session workload drawn from distributions, with no corpus at all.
//!
//! The second entry in the registry, and the one that proves the seam: it shares
//! nothing with the coding-session materializer except the arrival synthesis and
//! the write path. There is no source file, no context policy, and nothing to
//! fold — the numbers were never observed, so there is no gap between what was
//! reported and what is replayable.
//!
//! What it is for is the studies a recorded corpus cannot answer. A capacity
//! sweep wants to vary *one* thing — prompt length, output length, reuse — while
//! holding the rest fixed, and a real trace has whatever mix it happened to have.
//!
//! It is not a model of anything. A trace from here tells you how a deployment
//! responds to the shape you asked for, which is a different claim from telling
//! you how it responds to real traffic. Say which one you measured.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use req_frontend::schema::format::text_generation::session::{request_id, ExecutionRow};
use serde::Serialize;

use super::distribution::Distribution;
use super::{Generated, Generator};
use crate::arrivals::{self, ArrivalPattern, Rng};

#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Output canonical CSV. The manifest and plan are written beside it.
    #[arg(long)]
    out: PathBuf,

    /// How many sessions to draw.
    #[arg(long, default_value_t = 100)]
    sessions: usize,

    /// Rounds per session. `4`, `uniform:1..12`, `lognormal:3,0.9`.
    #[arg(long, default_value = "uniform:1..8")]
    rounds: Distribution,

    /// Fresh input tokens per round — the part not carried from the previous
    /// round. `512`, `uniform:256..2048`, `lognormal:1024,0.8`.
    #[arg(long, default_value = "lognormal:1024,0.8")]
    input_len: Distribution,

    /// Output tokens per round.
    #[arg(long, default_value = "lognormal:256,0.7")]
    output_len: Distribution,

    /// Milliseconds a session waits between one round completing and the next
    /// being submitted.
    #[arg(long, default_value = "lognormal:500,1.0")]
    tool_wait_ms: Distribution,

    /// Chance that a round drops its carried context instead of reusing it,
    /// standing in for a compaction. Round 0 never carries anything, so this
    /// applies from round 1 onward.
    #[arg(long, default_value_t = 0.0)]
    compaction_probability: f64,

    /// Synthetic session arrival rate, in sessions per second.
    #[arg(long, default_value_t = 1.0)]
    arrival_rate: f64,

    /// Synthetic session arrival process.
    #[arg(long, value_enum, default_value = "poisson")]
    arrival_pattern: ArrivalPattern,

    /// Seed for every draw here. Same seed, same trace, byte for byte.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// Exactly what it takes to draw this file again.
#[derive(Debug, Serialize)]
struct Record {
    sessions: usize,
    rounds: Distribution,
    input_len: Distribution,
    output_len: Distribution,
    tool_wait_ms: Distribution,
    compaction_probability: f64,
    arrival_rate_per_second: f64,
    arrival_pattern: &'static str,
    seed: u64,
    /// Rounds that dropped their carried context. Reported because the drawn
    /// count of a probabilistic event is a property of this file, not of the
    /// probability that produced it.
    compaction_rounds: usize,
}

impl Generator for Args {
    fn name(&self) -> &'static str {
        "synthetic"
    }

    fn out(&self) -> &Path {
        &self.out
    }

    fn generate(&self) -> Result<Generated> {
        self.validate()?;
        // One stream for everything, so the seed names the whole file rather
        // than one part of it.
        let mut rng = Rng::new(self.seed);
        let arrivals = arrivals::synthesize(
            &mut rng,
            self.sessions,
            self.arrival_rate,
            self.arrival_pattern,
        );

        let mut rows = Vec::new();
        let mut compaction_rounds = 0usize;
        for (session_ordinal, &arrival_time_ms) in arrivals.iter().enumerate() {
            let session_id = format!("synthetic_{session_ordinal:06}");
            let rounds = self.rounds.draw(&mut rng);
            // What the conversation holds after the round just emitted. This is
            // what makes the output a valid canonical trace: a round may only
            // reuse a prefix that an earlier round actually produced.
            let mut carried_context = 0usize;
            for round_idx in 0..rounds {
                let compacted = round_idx > 0
                    && self.compaction_probability > 0.0
                    && rng.chance(self.compaction_probability);
                if compacted {
                    compaction_rounds += 1;
                    carried_context = 0;
                }
                let prefix_len = carried_context;
                let input_len = self.input_len.draw(&mut rng);
                let output_len = self.output_len.draw(&mut rng);
                let tool_wait_after_ms = self.tool_wait_ms.draw(&mut rng) as f64;

                rows.push(ExecutionRow {
                    request_id: request_id(&session_id, round_idx),
                    session_id: session_id.clone(),
                    round_idx,
                    arrival_time_ms,
                    prefix_len,
                    input_len,
                    output_len,
                    tool_wait_after_ms,
                });
                carried_context = prefix_len + input_len + output_len;
            }
        }

        let record = Record {
            sessions: self.sessions,
            rounds: self.rounds,
            input_len: self.input_len,
            output_len: self.output_len,
            tool_wait_ms: self.tool_wait_ms,
            compaction_probability: self.compaction_probability,
            arrival_rate_per_second: self.arrival_rate,
            arrival_pattern: self.arrival_pattern.label(),
            seed: self.seed,
            compaction_rounds,
        };
        Ok(Generated {
            rows,
            record: serde_json::to_value(record)?,
        })
    }
}

impl Args {
    fn validate(&self) -> Result<()> {
        if self.sessions == 0 {
            bail!("--sessions must be greater than 0");
        }
        if !(self.arrival_rate.is_finite() && self.arrival_rate > 0.0) {
            bail!(
                "--arrival-rate must be finite and positive, got {}",
                self.arrival_rate
            );
        }
        if !(0.0..=1.0).contains(&self.compaction_probability) {
            bail!(
                "--compaction-probability is a fraction between 0 and 1, got {}",
                self.compaction_probability
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use req_frontend::schema::format::text_generation::session as v2;

    fn args() -> Args {
        Args {
            out: PathBuf::from("unused.csv"),
            sessions: 40,
            rounds: "uniform:1..6".parse().unwrap(),
            input_len: "lognormal:512,0.6".parse().unwrap(),
            output_len: "lognormal:128,0.5".parse().unwrap(),
            tool_wait_ms: "fixed:100".parse().unwrap(),
            compaction_probability: 0.0,
            arrival_rate: 2.0,
            arrival_pattern: ArrivalPattern::Poisson,
            seed: 5,
        }
    }

    #[test]
    fn what_it_draws_passes_the_canonical_validator_the_consumers_run() {
        // The point of validating here as well as in the shared write path: a
        // generator that emits an invalid chain should fail in its own tests,
        // not in someone's replay.
        let generated = args().generate().unwrap();

        v2::validate(&generated.rows).expect("a synthetic trace must be canonical");
    }

    #[test]
    fn a_round_only_reuses_a_prefix_an_earlier_round_actually_produced() {
        // The invariant that makes the output replayable at all. A prefix longer
        // than the conversation so far would be a cache hit on content that was
        // never sent.
        let generated = args().generate().unwrap();

        let mut context = 0usize;
        let mut current_session: Option<&str> = None;
        for row in &generated.rows {
            if current_session != Some(row.session_id.as_str()) {
                current_session = Some(&row.session_id);
                context = 0;
                assert_eq!(row.prefix_len, 0, "a first round carries nothing");
            }
            assert!(
                row.prefix_len <= context,
                "{} reuses {} of a {context}-token conversation",
                row.request_id,
                row.prefix_len,
            );
            context = row.prefix_len + row.input_len + row.output_len;
        }
    }

    #[test]
    fn the_same_seed_produces_the_same_file_and_a_different_one_does_not() {
        let first = args().generate().unwrap();
        let again = args().generate().unwrap();
        let other = Args { seed: 6, ..args() }.generate().unwrap();

        assert_eq!(first.rows, again.rows);
        assert_ne!(first.rows, other.rows);
    }

    #[test]
    fn compaction_drops_the_carried_context_and_is_counted_not_assumed() {
        // The manifest records the count drawn, not the probability asked for:
        // the two differ on any finite file, and the count is the property of
        // this one.
        let generated = Args {
            compaction_probability: 0.9,
            sessions: 60,
            ..args()
        }
        .generate()
        .unwrap();

        let drawn = generated.record["compaction_rounds"].as_u64().unwrap();
        let later_rounds = generated
            .rows
            .iter()
            .filter(|row| row.round_idx > 0)
            .count() as u64;
        assert!(drawn > 0 && drawn <= later_rounds);
        let resets = generated
            .rows
            .iter()
            .filter(|row| row.round_idx > 0 && row.prefix_len == 0)
            .count() as u64;
        assert_eq!(drawn, resets);
    }

    #[test]
    fn no_compaction_means_every_later_round_carries_the_whole_conversation() {
        let generated = args().generate().unwrap();

        assert!(generated
            .rows
            .iter()
            .filter(|row| row.round_idx > 0)
            .all(|row| row.prefix_len > 0));
        assert_eq!(generated.record["compaction_rounds"], 0);
    }

    #[test]
    fn the_record_carries_every_knob_needed_to_draw_the_file_again() {
        let generated = args().generate().unwrap();
        let record = &generated.record;

        for key in [
            "sessions",
            "rounds",
            "input_len",
            "output_len",
            "tool_wait_ms",
            "compaction_probability",
            "arrival_rate_per_second",
            "arrival_pattern",
            "seed",
        ] {
            assert!(
                !record[key].is_null(),
                "the manifest would not record {key}"
            );
        }
        // Distributions are recorded as the strings they were typed as.
        assert_eq!(record["input_len"], "lognormal:512,0.6");
    }

    #[test]
    fn arguments_that_cannot_produce_a_trace_are_refused() {
        assert!(Args {
            sessions: 0,
            ..args()
        }
        .generate()
        .is_err());
        assert!(Args {
            compaction_probability: 1.5,
            ..args()
        }
        .generate()
        .is_err());
        assert!(Args {
            arrival_rate: 0.0,
            ..args()
        }
        .generate()
        .is_err());
    }
}
