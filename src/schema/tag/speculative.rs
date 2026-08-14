//! Decode behaviour added by the `speculative` trace tag.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestSpeculative {
    #[serde(default)]
    pub accept_rate: Option<f32>,
}

impl RequestSpeculative {
    pub fn validate(&self, at: &str) -> Result<()> {
        if self
            .accept_rate
            .is_some_and(|rate| !rate.is_finite() || !(0.0..=1.0).contains(&rate))
        {
            bail!("{at}: accept_rate must be finite and between 0 and 1");
        }
        Ok(())
    }

    pub fn strategy(self) -> DecodingStrategy {
        self.accept_rate
            .map_or(DecodingStrategy::Standard, |accept_rate| {
                DecodingStrategy::Speculative { accept_rate }
            })
    }
}

/// A replay client cannot honour this — the server decides how it decodes — but
/// the trace still declares it, and a simulator reading the same file must get
/// the same value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DecodingStrategy {
    #[default]
    Standard,
    Speculative {
        accept_rate: f32,
    },
}
