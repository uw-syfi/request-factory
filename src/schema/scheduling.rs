//! The column the `priority` trace tag adds to a row.
//!
//! Priority is scheduling policy, not an SLO. Keeping it under its own tag lets
//! a trace declare service bounds without also claiming that it carries a
//! scheduling priority, and vice versa.
//!
//! Kept out of [`super::session_execution_v2::ExecutionRow`] on purpose. The
//! canonical column set *is* the format; a tag is something a file carries in
//! addition to its format, and folding tag columns into the format's own row
//! type would make every canonical file grow a column it does not want.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Default when a row declares the tag but leaves `priority` blank.
///
/// Zero rather than "unset": a scheduler comparing priorities needs a number for
/// every request, and picking the middle of a range nobody has agreed on would
/// be a guess dressed as a default.
pub const DEFAULT_PRIORITY: i64 = 0;

/// What the `priority` tag declares about one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestScheduling {
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
        self.priority.is_none()
    }

    pub fn priority_or_default(&self) -> i64 {
        self.priority.unwrap_or(DEFAULT_PRIORITY)
    }

    pub fn validate(&self, _at: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_row_uses_the_default_priority() {
        let blank = RequestScheduling::default();

        assert!(blank.is_empty());
        assert_eq!(blank.priority_or_default(), DEFAULT_PRIORITY);
        blank.validate("row 2").unwrap();
    }

    #[test]
    fn an_explicit_priority_is_preserved() {
        let scheduling = RequestScheduling { priority: Some(7) };

        assert_eq!(scheduling.priority_or_default(), 7);
        scheduling.validate("row 2").unwrap();
    }
}
