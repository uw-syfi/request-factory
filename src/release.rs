//! Where a workload unit gets its eligibility time.
//!
//! This vocabulary is shared because a measured replay and a simulation must
//! not be able to select different release disciplines with the same config.
//! The rate is deliberately not stored here: clients rescale an absolute trace
//! rate, while VibeSim rescales a rate-1-normalized trace. That arithmetic stays
//! with each consumer; the choice of timeline versus immediate eligibility does
//! not.

use anyhow::{bail, Result};

/// Which source supplies a top-level workload unit's eligibility time.
#[cfg_attr(feature = "runtime", derive(clap::ValueEnum))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrivalMode {
    /// Replay the trace's arrival timeline, using the consumer's declared rate.
    #[cfg_attr(feature = "runtime", value(name = "trace-timed"))]
    TraceTimed,
    /// Ignore recorded arrivals: every unit is eligible immediately.
    Saturated,
}

impl ArrivalMode {
    /// Stable names used by VibeSim preset schemas.
    pub const CONFIG_CHOICES: &'static [&'static str] = &["trace_timed", "saturated"];

    /// Parse a persisted config value.
    ///
    /// The CLI spelling is accepted as well so tools translating a command line
    /// into a preset do not need a second vocabulary table.
    pub fn parse_config(name: &str) -> Result<Self> {
        Ok(match name {
            "trace_timed" | "trace-timed" => Self::TraceTimed,
            "saturated" => Self::Saturated,
            "open_loop" | "closed_loop" => bail!(
                "arrival_mode {name:?} has been replaced: arrival and capacity are separate \
                 axes. Use arrival_mode: trace_timed (was open_loop), or arrival_mode: \
                 saturated with max_concurrency (was closed_loop)."
            ),
            other => bail!(
                "unknown arrival_mode {other:?} (expected one of {:?})",
                Self::CONFIG_CHOICES
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_and_cli_spellings_select_the_same_mode() {
        assert_eq!(
            ArrivalMode::parse_config("trace_timed").unwrap(),
            ArrivalMode::TraceTimed
        );
        assert_eq!(
            ArrivalMode::parse_config("trace-timed").unwrap(),
            ArrivalMode::TraceTimed
        );
        assert_eq!(
            ArrivalMode::parse_config("saturated").unwrap(),
            ArrivalMode::Saturated
        );
    }

    #[test]
    fn retired_cross_product_names_fail_with_migration_guidance() {
        let error = ArrivalMode::parse_config("closed_loop")
            .unwrap_err()
            .to_string();
        assert!(error.contains("arrival and capacity are separate axes"));
    }
}
