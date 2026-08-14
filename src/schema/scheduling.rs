//! The columns the `slo` trace tag adds to a row.
//!
//! Orthogonal to the kind, because a deadline is a statement about *this
//! request* rather than about what the request is: a coding agent's tool-call
//! round and a long generation live in one trace and owe different things.
//!
//! Kept out of [`super::session_execution_v2::ExecutionRow`] on purpose. The
//! canonical column set *is* the format; a tag is something a file carries in
//! addition to its format, and folding tag columns into the format's own row
//! type would make every canonical file grow two columns it does not want.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Default when a row declares the tag but leaves `priority` blank.
///
/// Zero rather than "unset": a scheduler comparing priorities needs a number for
/// every request, and picking the middle of a range nobody has agreed on would
/// be a guess dressed as a default.
pub const DEFAULT_PRIORITY: i64 = 0;

/// What the `slo` tag declares about one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestScheduling {
    /// Completion budget for this request, measured from its release. A
    /// *relative* budget, not an absolute instant: the same trace replayed at
    /// two different rates must mean the same thing both times.
    #[serde(default)]
    pub deadline_ms: Option<f64>,
    /// Server-side scheduling priority.
    ///
    /// A replay client carries this and does not act on it: it has no queue of
    /// its own to reorder, and the server is the thing being measured. Where a
    /// backend accepts a priority on the wire, forwarding it is that backend's
    /// decision to make explicitly.
    #[serde(default)]
    pub priority: Option<i64>,
}

impl RequestScheduling {
    pub fn is_empty(&self) -> bool {
        self.deadline_ms.is_none() && self.priority.is_none()
    }

    pub fn priority_or_default(&self) -> i64 {
        self.priority.unwrap_or(DEFAULT_PRIORITY)
    }

    /// Reject values that cannot mean what the column says.
    ///
    /// `at` is the caller's description of where the row came from, so one rule
    /// produces a message that names the file and line whichever consumer read
    /// it.
    pub fn validate(&self, at: &str) -> Result<()> {
        if let Some(deadline_ms) = self.deadline_ms {
            if !deadline_ms.is_finite() || deadline_ms <= 0.0 {
                bail!("{at}: deadline_ms must be finite and greater than zero, got {deadline_ms}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_row_declares_nothing_rather_than_a_deadline_of_zero() {
        let blank = RequestScheduling::default();

        assert!(blank.is_empty());
        assert_eq!(blank.priority_or_default(), DEFAULT_PRIORITY);
        blank.validate("row 2").unwrap();
    }

    #[test]
    fn a_deadline_of_zero_is_refused_rather_than_read_as_absent() {
        // Zero would be an unmeetable deadline, and an empty cell already says
        // "no deadline". A file spelling it 0 meant something else.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let scheduling = RequestScheduling {
                deadline_ms: Some(bad),
                priority: None,
            };
            assert!(scheduling.validate("row 2").is_err(), "{bad} was accepted");
        }
    }
}
