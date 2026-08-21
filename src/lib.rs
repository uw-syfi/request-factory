//! Trace-driven load generation for inference servers.
//!
//! Two things live here, and they are here for the same reason.
//!
//! [`schema::format::text_generation::session`] defines the canonical schema and its
//! validator. It is in a
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

pub mod release;
pub mod schema;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_support;

// The runtime is feature-gated so a consumer that only reads trace files -- a
// simulator replaying the same bytes -- can take the schemas without building an
// HTTP client, a tokenizer and an async runtime to get at them.
#[cfg(feature = "runtime")]
pub mod assets;
#[cfg(feature = "runtime")]
mod backend;
#[cfg(feature = "runtime")]
mod cli;
#[cfg(feature = "runtime")]
mod executor;
#[cfg(feature = "runtime")]
mod record;
#[cfg(feature = "runtime")]
mod runner;
#[cfg(feature = "runtime")]
mod slo;
#[cfg(feature = "runtime")]
mod slo_source;
#[cfg(feature = "runtime")]
mod summary;
#[cfg(feature = "runtime")]
mod synthetic;
#[cfg(feature = "runtime")]
mod timeline;
#[cfg(feature = "runtime")]
mod tokens;
#[cfg(feature = "runtime")]
mod util;
#[cfg(feature = "runtime")]
mod workload;

#[cfg(feature = "runtime")]
pub use cli::{Args, BackendKind};
pub use release::ArrivalMode;
#[cfg(feature = "runtime")]
pub use runner::{run_once, run_once_reusing, CorpusCache};
#[cfg(feature = "runtime")]
pub use summary::{RunMetrics, RunSummary};
