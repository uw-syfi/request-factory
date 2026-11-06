//! The column the `placement` trace tag adds to a row.
//!
//! Placement is routing, not ranking. A priority says how far forward a request
//! goes in some queue; a placement names the machine it must run on. Keeping
//! them under separate tags lets a trace declare a reproducible placement
//! sequence without also claiming to carry a scheduling priority, and vice
//! versa.
//!
//! There is deliberately no default. "Blank" means the trace declines to place
//! this request and the consumer's own placement policy decides — a real
//! third state, not worker 0. A consumer that requires a placement should
//! refuse a blank cell rather than invent one.
//!
//! Kept out of [`crate::schema::format::text_generation::independent::TextGenerationRow`]
//! on purpose: the canonical column set *is* the format, and a tag is something
//! a file carries in addition to its format.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// What the `placement` tag declares about one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestPlacement {
    /// Which server this request must run on, as an index within the consumer's
    /// own set of replicas.
    ///
    /// This crate does not know the topology it will be replayed against, so it
    /// cannot check the index against a replica count. It validates only that
    /// the cell parses as a non-negative index; the consumer that owns the
    /// replica list is the one that can say whether `7` exists.
    #[serde(default)]
    pub target_worker: Option<u16>,
}

impl RequestPlacement {
    pub fn is_empty(&self) -> bool {
        self.target_worker.is_none()
    }

    pub fn validate(&self, _at: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_row_declares_no_placement() {
        let blank = RequestPlacement::default();

        assert!(blank.is_empty());
        assert_eq!(blank.target_worker, None);
        blank.validate("row 2").unwrap();
    }

    #[test]
    fn an_explicit_target_worker_is_preserved() {
        let placement = RequestPlacement {
            target_worker: Some(3),
        };

        assert!(!placement.is_empty());
        assert_eq!(placement.target_worker, Some(3));
        placement.validate("row 2").unwrap();
    }
}
