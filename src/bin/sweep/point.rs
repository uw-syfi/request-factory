//! Running one point of a sweep, and everything that must be true before it runs.
//!
//! Two things here are not bookkeeping. **Resetting the server's prefix cache**
//! between points, because otherwise point *k+1* starts warm on point *k*'s
//! content and its measured hit rate is not comparable to anything. And
//! **resumability**, because a sweep is long enough that losing it to a
//! disconnected shell should not mean re-measuring the points that already
//! succeeded.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use req_frontend::{run_once_reusing, Args, CorpusCache, RunMetrics};
use serde::{Deserialize, Serialize};

/// What happened to the server's carried-over state before this point ran.
///
/// Recorded per point rather than assumed once, because a sweep that could not
/// reset is still a sweep worth reading — as long as it says so. A contaminated
/// curve reported as a clean one is the failure worth preventing here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CacheReset {
    /// The endpoint accepted the reset.
    Done,
    /// This backend exposes no reset, so every point after the first starts warm
    /// on its predecessor's content. Prefix-cache numbers across points are not
    /// comparable; latency and throughput are affected to the extent the trace
    /// reuses content.
    Unsupported { backend: String },
    /// The endpoint has a reset and it failed. Louder than `Unsupported`,
    /// because here something was expected to work.
    Failed { error: String },
}

/// One measured point, as it is written to disk and read back on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointRecord {
    pub rate: f64,
    /// Directory holding this point's own log, summary and timeline.
    pub directory: String,
    pub cache_reset: CacheReset,
    pub metrics: RunMetrics,
}

/// Where one point's outputs live, derived from its rate.
///
/// Fixed decimals so a resumed sweep finds the directory it wrote last time.
/// Bisection produces rates that do not round to anything tidy, and formatting
/// them shortest-first would make `2.5` and `2.500001` share a directory.
pub fn directory_for(root: &Path, rate: f64) -> PathBuf {
    root.join("points").join(format!("rate_{rate:012.6}"))
}

/// Read a completed point back, or `None` if it was never finished.
///
/// A missing or unparsable record means "not done", never "done with defaults":
/// a half-written summary from a killed process must be re-measured, not
/// silently accepted as a data point.
pub fn completed(directory: &Path) -> Option<PointRecord> {
    let text = std::fs::read_to_string(directory.join("point.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// Run one point: reset the server, replay the trace, record what came back.
pub async fn run(
    base: &Args,
    rate: f64,
    directory: &Path,
    corpus: &mut CorpusCache,
) -> Result<PointRecord> {
    std::fs::create_dir_all(directory).with_context(|| {
        format!(
            "failed to create the point directory {}",
            directory.display()
        )
    })?;

    let cache_reset = reset_prefix_cache(base).await;
    if let CacheReset::Failed { error } = &cache_reset {
        eprintln!("sweep | rate {rate:.6}/s: prefix-cache reset failed: {error}");
    }

    let mut args = base.clone();
    args.rate = Some(rate);
    args.log_path = directory.join("requests.jsonl").display().to_string();
    args.summary_path = Some(directory.join("summary.json").display().to_string());
    args.timeline_path = directory.join("timeline.parquet").display().to_string();

    let summary = run_once_reusing(args, corpus)
        .await
        .with_context(|| format!("the run at rate {rate:.6}/s failed"))?;

    let record = PointRecord {
        rate,
        directory: directory.display().to_string(),
        cache_reset,
        metrics: summary.metrics(),
    };
    // Written last, and only on success: its presence is what a resumed sweep
    // reads as "this point is done".
    let file = std::fs::File::create(directory.join("point.json"))
        .with_context(|| format!("failed to record {}", directory.display()))?;
    serde_json::to_writer_pretty(file, &record).context("failed to write the point record")?;
    Ok(record)
}

/// Ask the server to forget what earlier points taught it.
async fn reset_prefix_cache(args: &Args) -> CacheReset {
    let Some(url) = reset_endpoint(args) else {
        return CacheReset::Unsupported {
            backend: format!("{:?}", args.backend).to_lowercase(),
        };
    };
    let client = reqwest::Client::new();
    match client.post(&url).send().await {
        Ok(response) if response.status().is_success() => CacheReset::Done,
        Ok(response) => CacheReset::Failed {
            error: format!("{url} returned {}", response.status()),
        },
        Err(error) => CacheReset::Failed {
            error: format!("{url}: {error}"),
        },
    }
}

/// The reset endpoint for this backend, if it has one.
///
/// vLLM's `/reset_prefix_cache` lives on the server root, beside the API rather
/// than inside it, so the `/v1` an OpenAI-compatible base URL carries is
/// stripped.
fn reset_endpoint(args: &Args) -> Option<String> {
    let base = args.base_url.trim_end_matches('/');
    let root = base.strip_suffix("/v1").unwrap_or(base);
    match args.backend {
        req_frontend::BackendKind::Openai | req_frontend::BackendKind::VllmTokens => {
            Some(format!("{root}/reset_prefix_cache"))
        }
        // SGLang has `/flush_cache`, but it is not the same operation on every
        // version and a wrong guess would report a reset that did not happen.
        // Saying "unsupported" is the honest state until it is verified.
        req_frontend::BackendKind::SglangTokens | req_frontend::BackendKind::OpenaiChat => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_points_directory_is_stable_across_runs_of_the_same_rate() {
        // Resume depends on this: bisection produces rates like 12.03125, and a
        // shortest-representation name would not round-trip.
        let root = Path::new("/sweep");
        assert_eq!(
            directory_for(root, 12.031_25),
            PathBuf::from("/sweep/points/rate_00012.031250")
        );
        assert_ne!(
            directory_for(root, 2.5),
            directory_for(root, 2.500_001),
            "two distinct rates must not share a directory"
        );
    }

    #[test]
    fn the_reset_endpoint_sits_beside_the_api_rather_than_inside_it() {
        let mut args = base_args();
        args.base_url = "http://host:8000/v1".to_string();
        args.backend = req_frontend::BackendKind::Openai;
        assert_eq!(
            reset_endpoint(&args).as_deref(),
            Some("http://host:8000/reset_prefix_cache")
        );

        args.base_url = "http://host:8000".to_string();
        args.backend = req_frontend::BackendKind::VllmTokens;
        assert_eq!(
            reset_endpoint(&args).as_deref(),
            Some("http://host:8000/reset_prefix_cache")
        );
    }

    #[test]
    fn a_backend_with_no_verified_reset_says_so_instead_of_guessing_one() {
        let mut args = base_args();
        args.backend = req_frontend::BackendKind::SglangTokens;

        assert_eq!(reset_endpoint(&args), None);
    }

    #[test]
    fn an_unfinished_point_is_not_mistaken_for_a_completed_one() {
        let directory =
            std::env::temp_dir().join(format!("req_frontend_sweep_point_{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();

        // No record at all.
        assert!(completed(&directory).is_none());
        // A record a killed process left half-written.
        std::fs::write(directory.join("point.json"), "{\"rate\": 1.0").unwrap();
        assert!(completed(&directory).is_none());

        std::fs::remove_dir_all(&directory).ok();
    }

    fn base_args() -> Args {
        use clap::Parser;
        Args::parse_from([
            "sweep",
            "--trace",
            "trace.csv",
            "--text-file",
            "corpus.txt",
            "--tokenizer",
            "gpt2",
            "--model",
            "model",
        ])
    }
}
