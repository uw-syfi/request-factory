//! Where a run's service-level objective comes from.
//!
//! Two scopes, because the two questions are different. A **global** objective
//! is a property of the experiment: "under this deployment, is 500 ms TTFT
//! achievable?" A **trace** objective is a property of the workload: a coding
//! agent's tool-calling round has a different obligation than a chat turn, and
//! that obligation belongs with the trace rather than being retyped on every
//! command line that replays it.
//!
//! A trace declares its objective in a sidecar beside it — the same convention
//! `.manifest.json` and `.plan.json` already use, so a trace and everything true
//! about it travel together. `--slo` overrides it, and says so.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::slo::{SloSource, SloSpec};

/// Suffix appended to a trace path to find its objective.
const SIDECAR_SUFFIX: &str = ".slo.json";

/// The objective in force for one run, and where it came from.
///
/// `None` means no objective was declared anywhere, which is not the same as an
/// objective that everything meets: a run with no SLO reports no attainment
/// rather than reporting 100%.
pub(crate) fn resolve(
    trace_path: &str,
    flag: Option<&str>,
) -> Result<Option<(SloSpec, SloSource)>> {
    if let Some(spec) = flag {
        let spec = SloSpec::parse(spec).context("invalid --slo")?;
        if let Some(path) = sidecar_path(trace_path) {
            if path.exists() {
                // Said out loud rather than resolved silently: a trace that
                // declares an objective and a run that ignores it is exactly the
                // situation where a quiet winner produces a number nobody can
                // account for later.
                eprintln!(
                    "slo | --slo {spec} overrides the objective declared in {}",
                    path.display(),
                );
            }
        }
        return Ok(Some((spec, SloSource::Global)));
    }
    let Some(path) = sidecar_path(trace_path) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read the SLO sidecar {}", path.display()))?;
    let spec: SloSpec = serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid SLO document", path.display()))?;
    if spec.is_empty() {
        // An empty document is almost certainly a mistake, and the mistake it
        // makes is invisible: every step would attain an objective of nothing.
        anyhow::bail!(
            "{} declares no bounds; remove the file or give it at least one of {:?}",
            path.display(),
            SloSpec::METRICS,
        );
    }
    eprintln!("slo | {spec} declared by {}", path.display());
    Ok(Some((spec, SloSource::Trace)))
}

/// `trace/execution.csv` -> `trace/execution.slo.json`.
///
/// Replaces the extension rather than appending to it, so the sidecar sits
/// beside the trace under one name whatever the trace is called.
fn sidecar_path(trace_path: &str) -> Option<PathBuf> {
    let path = Path::new(trace_path);
    let stem = path.file_stem()?.to_str()?;
    Some(path.with_file_name(format!("{stem}{SIDECAR_SUFFIX}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("req_frontend_slo_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_sidecar_sits_beside_the_trace_under_the_traces_own_name() {
        assert_eq!(
            sidecar_path("trace/execution.csv").unwrap(),
            PathBuf::from("trace/execution.slo.json"),
        );
        assert_eq!(
            sidecar_path("bare").unwrap(),
            PathBuf::from("bare.slo.json"),
        );
    }

    #[test]
    fn no_flag_and_no_sidecar_is_no_objective_rather_than_an_empty_one() {
        let dir = scratch("absent");
        let trace = dir.join("workload.csv");
        std::fs::write(&trace, "id\n").unwrap();

        assert!(resolve(trace.to_str().unwrap(), None).unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_trace_can_declare_its_own_objective() {
        let dir = scratch("declared");
        let trace = dir.join("workload.csv");
        std::fs::write(&trace, "id\n").unwrap();
        std::fs::write(dir.join("workload.slo.json"), r#"{"ttft_ms": 250}"#).unwrap();

        let (spec, source) = resolve(trace.to_str().unwrap(), None).unwrap().unwrap();
        assert_eq!(spec.ttft_ms, Some(250.0));
        assert_eq!(source, SloSource::Trace);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_flag_wins_over_the_sidecar_and_the_summary_records_which() {
        let dir = scratch("override");
        let trace = dir.join("workload.csv");
        std::fs::write(&trace, "id\n").unwrap();
        std::fs::write(dir.join("workload.slo.json"), r#"{"ttft_ms": 250}"#).unwrap();

        let (spec, source) = resolve(trace.to_str().unwrap(), Some("ttft_ms=900"))
            .unwrap()
            .unwrap();
        assert_eq!(spec.ttft_ms, Some(900.0));
        assert_eq!(source, SloSource::Global);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_sidecar_declaring_nothing_is_refused_rather_than_trivially_attained() {
        let dir = scratch("empty");
        let trace = dir.join("workload.csv");
        std::fs::write(&trace, "id\n").unwrap();
        std::fs::write(dir.join("workload.slo.json"), "{}").unwrap();

        let err = resolve(trace.to_str().unwrap(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("declares no bounds"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
