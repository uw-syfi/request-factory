mod priority;
mod session;
mod slo;
mod speculative;

use anyhow::{bail, Result};

use super::RequestFamily;

/// An orthogonal column bundle added to a complete input-file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceTag {
    Session,
    Slo,
    Priority,
    Speculative,
}

impl TraceTag {
    pub const CHOICES: &'static [&'static str] = &["session", "slo", "priority", "speculative"];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "session" => Self::Session,
            "slo" => Self::Slo,
            "priority" => Self::Priority,
            "speculative" => Self::Speculative,
            other => bail!(
                "unknown trace tag {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    pub fn name(self) -> &'static str {
        Self::CHOICES[self as usize]
    }

    pub fn columns(self) -> &'static [&'static str] {
        match self {
            Self::Session => &["session_id", "prefix_kv", "tool_wait_after_ms"],
            Self::Slo => &["ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms"],
            Self::Priority => &["priority"],
            Self::Speculative => &["accept_rate"],
        }
    }

    pub fn applies_to(self, request_family: RequestFamily) -> bool {
        match self {
            Self::Session | Self::Slo | Self::Priority => true,
            Self::Speculative => matches!(
                request_family,
                RequestFamily::TextGeneration
                    | RequestFamily::ImageToText
                    | RequestFamily::VideoToText
                    | RequestFamily::AudioToText
                    | RequestFamily::OmniGeneration
            ),
        }
    }
}

pub use priority::{RequestPriority, DEFAULT_PRIORITY};
pub use session::RequestSession;
pub use slo::RequestSlo;
pub use speculative::{DecodingStrategy, RequestSpeculative};
