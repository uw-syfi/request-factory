//! The canonical execution trace, shared by everything that reads one.
//!
//! [`v2`] defines the `session-execution-v2` schema and its validator. It lives
//! in a library rather than in a binary because the generator that writes a file
//! and the runtime that replays it must agree on every rule; a second
//! implementation of those rules is exactly the drift the canonical trace exists
//! to eliminate.
//!
//! Resolving a *raw* trace into a canonical one is not here. That is a
//! generation-time decision owned solely by the `tracegen` binary, and its
//! outcome is recorded in the manifest beside the file it produced.

pub mod v2;
