//! One assertion, stated so that a reader can disagree with it.
//!
//! A check is not a boolean. Every one carries the claim it is making in words,
//! the quantity it measured, the value it expected, and the tolerance it allowed
//! — because a fidelity harness whose output is "12 passed" tells you nothing
//! about what was verified, and a tolerance nobody can see is a tolerance nobody
//! reviews. When one of these fails, the failure line should be enough to argue
//! with.

use serde::Serialize;

/// What a measured value has to satisfy.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Bound {
    /// The measurement must not exceed this. For quantities whose ideal is zero
    /// — lag, drift, drop counts — where only one direction is a defect.
    AtMost { limit: f64 },
    /// The measurement must land within `by` of `of`. For quantities the stub
    /// server fixes by construction, where either direction is a disagreement.
    Within { of: f64, by: f64 },
    /// Exactly this, no tolerance. For counts.
    Exactly { value: f64 },
}

impl Bound {
    fn satisfied_by(self, measured: f64) -> bool {
        match self {
            Self::AtMost { limit } => measured <= limit,
            Self::Within { of, by } => (measured - of).abs() <= by,
            Self::Exactly { value } => measured == value,
        }
    }

    fn describe(self) -> String {
        match self {
            Self::AtMost { limit } => format!("<= {limit}"),
            Self::Within { of, by } => format!("{of} ± {by}"),
            Self::Exactly { value } => format!("== {value}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    /// What this check asserts, in a sentence. Read aloud, it should be a claim
    /// about the client that someone could reasonably dispute.
    pub claim: &'static str,
    /// The name of the number, so `measured` is never a bare float.
    pub quantity: String,
    pub measured: f64,
    pub unit: &'static str,
    pub bound: Bound,
    /// Why the tolerance is what it is. Required, because the alternative is a
    /// number that gets widened whenever it fails.
    pub tolerance_rationale: &'static str,
    /// Anything else worth reading beside the verdict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub passed: bool,
}

impl Check {
    pub fn new(
        name: &'static str,
        claim: &'static str,
        quantity: impl Into<String>,
        measured: f64,
        unit: &'static str,
        bound: Bound,
        tolerance_rationale: &'static str,
    ) -> Self {
        Self {
            name,
            claim,
            quantity: quantity.into(),
            measured,
            unit,
            bound,
            tolerance_rationale,
            detail: None,
            passed: bound.satisfied_by(measured),
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn report_line(&self) -> String {
        let mark = if self.passed { "pass" } else { "FAIL" };
        let mut line = format!(
            "{mark} | {:<38} {} = {:.4} {} (needs {})",
            self.name,
            self.quantity,
            self.measured,
            self.unit,
            self.bound.describe(),
        );
        if let Some(detail) = &self.detail {
            line.push_str(&format!("\n     | {detail}"));
        }
        if !self.passed {
            line.push_str(&format!("\n     | claim: {}", self.claim));
            line.push_str(&format!("\n     | tolerance: {}", self.tolerance_rationale));
        }
        line
    }
}

/// Nearest-rank-free percentile, matching `summary.rs::percentile_sorted`.
///
/// The same definition the run summaries use, so a check and the summary it is
/// checking cannot disagree about what p99 means.
pub fn percentile(values: &mut [f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    if values.len() == 1 {
        return Some(values[0]);
    }
    let position = fraction.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        return Some(values[low]);
    }
    let weight = position - low as f64;
    Some(values[low] * (1.0 - weight) + values[high] * weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bound_is_what_decides_the_verdict_not_the_caller() {
        let check = Check::new(
            "example",
            "claim",
            "value",
            5.0,
            "ms",
            Bound::AtMost { limit: 4.0 },
            "why",
        );

        assert!(!check.passed);
        assert!(check.report_line().starts_with("FAIL"));
    }

    #[test]
    fn within_is_two_sided_because_either_direction_is_a_disagreement() {
        let too_fast = Bound::Within { of: 50.0, by: 5.0 };

        assert!(too_fast.satisfied_by(46.0));
        assert!(!too_fast.satisfied_by(44.0));
        assert!(!too_fast.satisfied_by(56.0));
    }

    #[test]
    fn percentiles_agree_with_the_summarys_definition() {
        let mut values: Vec<f64> = (1..=10).map(|value| value as f64).collect();

        assert_eq!(percentile(&mut values, 0.5), Some(5.5));
        assert_eq!(percentile(&mut values.clone(), 0.9), Some(9.1));
        assert_eq!(percentile(&mut Vec::new(), 0.5), None);
    }

    #[test]
    fn a_failing_check_prints_its_claim_and_a_passing_one_does_not() {
        // The failure line has to be enough to argue with; the passing line has
        // to be short enough that twelve of them are readable.
        let failing = Check::new(
            "example",
            "the client releases on time",
            "p99",
            9.0,
            "ms",
            Bound::AtMost { limit: 1.0 },
            "because",
        );
        let passing = Check::new(
            "example",
            "the client releases on time",
            "p99",
            0.5,
            "ms",
            Bound::AtMost { limit: 1.0 },
            "because",
        );

        assert!(failing.report_line().contains("releases on time"));
        assert!(!passing.report_line().contains("releases on time"));
    }
}
