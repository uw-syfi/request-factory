//! Per-request metric bounds added by the `slo` trace tag.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// The `_slo_ms` suffix distinguishes declared thresholds from measurements in
/// logs that use names such as `ttft_ms`. Every field is independently optional:
/// two rows in one trace may owe different metrics and different thresholds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestSlo {
    #[serde(default)]
    pub ttft_slo_ms: Option<f64>,
    #[serde(default)]
    pub tpot_slo_ms: Option<f64>,
    #[serde(default)]
    pub e2e_slo_ms: Option<f64>,
}

impl RequestSlo {
    pub fn is_empty(&self) -> bool {
        self.bounds().all(|(_, bound)| bound.is_none())
    }

    pub fn bounds(&self) -> impl Iterator<Item = (&'static str, Option<f64>)> {
        [
            ("ttft_slo_ms", self.ttft_slo_ms),
            ("tpot_slo_ms", self.tpot_slo_ms),
            ("e2e_slo_ms", self.e2e_slo_ms),
        ]
        .into_iter()
    }

    pub fn validate(&self, at: &str) -> Result<()> {
        for (name, bound) in self.bounds() {
            if let Some(bound) = bound {
                if !bound.is_finite() || bound <= 0.0 {
                    bail!("{at}: {name} must be finite and greater than zero, got {bound}");
                }
            }
        }
        Ok(())
    }
}
