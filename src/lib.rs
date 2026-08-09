//! Contract shared by the trace generator and the replay runtime.
//!
//! Everything here is a rule that both binaries must apply identically:
//! [`policy`] resolves a raw trace's prefix/append split, and [`v2`] defines the
//! canonical execution trace they exchange. Keeping them in a library rather
//! than in each binary is the point — a second implementation of these rules is
//! exactly the drift the canonical trace exists to eliminate.

pub mod policy;
pub mod v2;
