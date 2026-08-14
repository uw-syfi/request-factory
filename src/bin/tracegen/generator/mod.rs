//! The registry: what a generator owes, and what it gets for free.
//!
//! One category of trace will have several ways to produce it — a materializer
//! that resolves a real recorded corpus, a synthesizer that draws from
//! distributions, later a replayer of some other lab's export — and a new
//! category must not mean a new binary. So everything that is true of *every*
//! canonical trace lives outside the generators: validating what was produced,
//! writing the CSV, deriving the plan, and computing the manifest's totals.
//!
//! A generator supplies two things. **Rows**, and a **record** of how it made
//! them. The record goes into the manifest verbatim and is the generator's whole
//! contribution to reproducibility, so a generator that forgets to record a knob
//! produces a trace nobody can regenerate — which is why it is a required part
//! of the return value rather than an optional method.
//!
//! Note what is *not* in the trait: token totals, session counts, the prefix hit
//! rate. Those are derived from the emitted rows by the shared path, so they
//! describe the file rather than the generator's own bookkeeping. The two used
//! to be able to drift.

pub(crate) mod coding_session;
pub(crate) mod distribution;
pub(crate) mod synthetic;

use std::path::Path;

use anyhow::Result;
use req_frontend::schema::session_execution_v2::ExecutionRow;

/// What one generator produced.
pub(crate) struct Generated {
    pub(crate) rows: Vec<ExecutionRow>,
    /// This generator's parameters and its own statistics, recorded under
    /// `parameters` in the manifest. Anything needed to produce this file again.
    pub(crate) record: serde_json::Value,
}

/// The contract every entry in the registry meets.
pub(crate) trait Generator {
    /// Name recorded in the manifest, and the subcommand that selects it.
    fn name(&self) -> &'static str;

    /// Where the canonical trace goes. The manifest and plan are written beside
    /// it, by the shared path.
    fn out(&self) -> &Path;

    fn generate(&self) -> Result<Generated>;
}

/// The registry itself.
///
/// A subcommand rather than a `--generator` string plus a union of every
/// generator's flags: the flags of one generator mean nothing to another, and
/// `tracegen synthetic --help` should list what `synthetic` takes and nothing
/// else.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum Registry {
    /// Materialize a recorded coding-agent trace into canonical form.
    CodingSession(coding_session::Args),
    /// Draw a session workload from distributions, with no corpus at all.
    Synthetic(synthetic::Args),
}

impl Registry {
    pub(crate) fn selected(&self) -> &dyn Generator {
        match self {
            Self::CodingSession(args) => args,
            Self::Synthetic(args) => args,
        }
    }
}
