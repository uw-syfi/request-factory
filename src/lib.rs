//! Trace-driven load generation for inference servers.
//!
//! Two things live here, and they are here for the same reason.
//!
//! [`v2`] defines the `session-execution-v2` schema and its validator. It is in a
//! library rather than a binary because the generator that writes a file and the
//! runtime that replays it must agree on every rule; a second implementation of
//! those rules is exactly the drift the canonical trace exists to eliminate.
//!
//! [`run_once`] is the runtime itself. It is in the library rather than in
//! `main.rs` because a run is not only something a person launches from a shell —
//! a sweep drives dozens of them, and driving them in one process is what lets it
//! load the tokenizer and the synthetic token pool once instead of re-tokenizing a
//! hundred-million-token corpus per point.
//!
//! Resolving a *raw* trace into a canonical one is not here. That is a
//! generation-time decision owned solely by the `tracegen` binary, and its
//! outcome is recorded in the manifest beside the file it produced.

mod backend;
mod cli;
mod executor;
mod record;
mod runner;
mod summary;
mod tokens;
mod trace;
mod util;
mod workload;

pub mod v2;

pub use cli::{Args, ArrivalMode, BackendKind};
pub use runner::run_once;
pub use summary::RunSummary;
pub use trace::TraceFormat;
