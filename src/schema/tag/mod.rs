mod placement;
mod priority;
mod session;
mod slo;
mod speculative;

use anyhow::{bail, Result};

use super::RequestFamily;

/// An orthogonal column bundle added to a complete input-file format.
///
/// `name()` indexes [`Self::CHOICES`] by `self as usize`, so a new variant must
/// be **appended** here and its name appended to `CHOICES` in the same position.
/// Inserting in the middle silently shifts every later tag's name; the
/// `every_tag_name_round_trips` test below is what catches that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceTag {
    Session,
    Slo,
    Priority,
    Speculative,
    Placement,
}

impl TraceTag {
    pub const CHOICES: &'static [&'static str] =
        &["session", "slo", "priority", "speculative", "placement"];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "session" => Self::Session,
            "slo" => Self::Slo,
            "priority" => Self::Priority,
            "speculative" => Self::Speculative,
            "placement" => Self::Placement,
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
            Self::Placement => &["target_worker"],
        }
    }

    pub fn applies_to(self, request_family: RequestFamily) -> bool {
        match self {
            // Every family is served by some set of replicas, so naming one is
            // meaningful regardless of what the request generates.
            Self::Session | Self::Slo | Self::Priority | Self::Placement => true,
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

pub use placement::RequestPlacement;
pub use priority::{RequestPriority, DEFAULT_PRIORITY};
pub use session::RequestSession;
pub use slo::RequestSlo;
pub use speculative::{AcceptanceProfile, DecodingStrategy, RequestSpeculative};

#[cfg(test)]
mod tests {
    use super::*;

    /// `name()` is `CHOICES[self as usize]`, which the compiler cannot check.
    /// Round-tripping every name pins the enum order to the array order.
    #[test]
    fn every_tag_name_round_trips() {
        for (index, name) in TraceTag::CHOICES.iter().enumerate() {
            let tag = TraceTag::parse(name).unwrap();
            assert_eq!(tag.name(), *name);
            assert_eq!(tag as usize, index, "{name} is out of order");
            assert!(!tag.columns().is_empty(), "{name} declares no column");
        }
    }
}
