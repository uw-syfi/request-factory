//! Find the rate at which a server stops doing what you wanted, without
//! guessing where to look.
//!
//! A grid sweep asks you to know the answer before you start: too coarse and the
//! knee sits in a gap, too fine and most of the points are spent far away from
//! it. The adaptive modes here ramp by doubling until the boundary flips, bisect
//! back to locate it, then spend their remaining points *at* the knee, which is
//! the only part of the curve anyone reads.
//!
//! Runs happen in this process, which is what lets a sweep of twenty points pay
//! for the tokenizer and the hundred-million-token synthetic corpus once.

mod boundary;
mod point;
mod search;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use req_frontend::{Args, ArrivalMode, CorpusCache};
use serde::Serialize;

use boundary::{AttainmentMetric, Boundary, Measured};
use search::{Knee, Phase, Search, SearchConfig};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Adaptive rate sweep: ramp, bisect the knee, then draw it"
)]
struct SweepArgs {
    /// What this sweep is looking for.
    #[arg(long, value_enum, default_value = "max-throughput")]
    mode: Mode,

    /// Directory for `sweep.json` and one subdirectory per point.
    #[arg(long)]
    out: PathBuf,

    /// First rate to offer, in workload units per second.
    #[arg(long, default_value_t = 1.0)]
    start_rate: f64,

    /// Never offer more than this. A sweep that reaches it without crossing
    /// reports that the knee is above the range searched.
    #[arg(long, default_value_t = 4096.0)]
    max_rate: f64,

    /// Never offer less than this while ramping downward.
    #[arg(long, default_value_t = 0.001)]
    min_rate: f64,

    /// Stop bisecting once the bracket is this narrow relative to its upper end.
    #[arg(long, default_value_t = 0.05)]
    tolerance: f64,

    /// Extra points drawn across the located knee.
    #[arg(long, default_value_t = 3)]
    densify_points: usize,

    /// `max-throughput`: how far delivered throughput may fall behind the
    /// offered rate and still count as keeping up.
    #[arg(long, default_value_t = 0.10)]
    max_shortfall: f64,

    /// `max-rate-under-slo`: the attainment the run must keep.
    #[arg(long, default_value_t = 0.99)]
    target_attainment: f64,

    /// `max-rate-under-slo`: which attainment to watch.
    #[arg(long, value_enum, default_value = "overall")]
    attainment_metric: AttainmentChoice,

    /// `grid`: the rates to run, comma-separated. No search happens.
    #[arg(long, value_delimiter = ',')]
    rates: Vec<f64>,

    /// Re-measure every point, including ones a previous sweep completed.
    #[arg(long)]
    no_resume: bool,

    /// Everything a single run takes. `--rate` is the knob this sweep turns, so
    /// it is rejected here; the output paths are set per point.
    #[command(flatten)]
    run: Args,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Ramp until offering more stops buying more served throughput.
    MaxThroughput,
    /// Ramp until SLO attainment falls below the target.
    MaxRateUnderSlo,
    /// Run the declared rates and nothing else.
    Grid,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum AttainmentChoice {
    Overall,
    DeclaredDeadline,
}

/// The whole sweep, as one readable document.
#[derive(Debug, Serialize)]
struct SweepReport {
    /// The knob this sweep turned. Named rather than implied, because the search
    /// machinery is knob-agnostic and a later sweep may turn a different one.
    knob: &'static str,
    objective: &'static str,
    boundary: Option<Boundary>,
    config: ReportConfig,
    /// Every point, in the order it was measured — including ones the search
    /// discarded. A curve drawn only from the surviving points would hide how
    /// the search got there.
    points: Vec<ReportPoint>,
    /// The same points sorted by rate: the curve.
    curve: Vec<CurveEntry>,
    knee: Option<Knee>,
    /// Set when any point ran against a server whose carried-over state could
    /// not be cleared, so a reader is not left to infer it from the points.
    contamination_warning: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReportConfig {
    mode: String,
    start_rate: f64,
    max_rate: f64,
    min_rate: f64,
    tolerance: f64,
    densify_points: usize,
    resumed_points: usize,
}

#[derive(Debug, Serialize)]
struct ReportPoint {
    #[serde(flatten)]
    record: point::PointRecord,
    phase: Option<Phase>,
    crossed: Option<bool>,
    verdict: Option<String>,
    /// True when this point was read back from a previous sweep rather than
    /// measured now.
    reused: bool,
}

#[derive(Debug, Serialize)]
struct CurveEntry {
    rate: f64,
    output_token_throughput_per_s: Option<f64>,
    request_throughput_per_s: Option<f64>,
    ttft_ms_p50: Option<f64>,
    ttft_ms_p90: Option<f64>,
    tpot_ms_p50: Option<f64>,
    tpot_ms_p90: Option<f64>,
    slo_attainment: Option<f64>,
    declared_deadline_attainment: Option<f64>,
    failed_steps: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = SweepArgs::parse();
    validate(&args)?;
    // Said once, at the top, rather than left for a reader to discover: the
    // sweep owns these three paths and whatever was passed for them is not used.
    eprintln!(
        "sweep | point outputs go to {}/points/rate_*/; --log-path, --summary-path and \
         --timeline-path are set per point",
        args.out.display()
    );
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("failed to create {}", args.out.display()))?;
    // Rate scaling replays the trace's own arrival timeline at a new speed. A
    // saturated run has no timeline to scale, so there would be no knob at all.
    args.run.arrival_mode = ArrivalMode::TraceTimed;

    let report = match args.mode {
        Mode::Grid => run_grid(&args).await?,
        _ => run_adaptive(&args).await?,
    };

    let path = args.out.join("sweep.json");
    let file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(file, &report)
        .with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("sweep | wrote {}", path.display());
    if let Some(warning) = &report.contamination_warning {
        eprintln!("sweep | warning: {warning}");
    }
    if let Some(knee) = &report.knee {
        eprintln!("sweep | knee: {}", describe(knee));
    }
    Ok(())
}

fn validate(args: &SweepArgs) -> Result<()> {
    if args.run.rate.is_some() {
        bail!(
            "--rate is the knob this sweep turns; drop it and set the range with --start-rate / \
             --max-rate"
        );
    }
    if args.run.dry_run {
        bail!("--dry-run never contacts a server, so a sweep of dry runs measures nothing");
    }
    if args.mode == Mode::Grid && args.rates.is_empty() {
        bail!("--mode grid needs the rates to run: --rates 1,2,4,8");
    }
    if args.mode != Mode::Grid && !args.rates.is_empty() {
        bail!(
            "--rates only applies to --mode grid; the adaptive modes choose their own points from \
             --start-rate and --max-rate"
        );
    }
    for rate in &args.rates {
        if !rate.is_finite() || *rate <= 0.0 {
            bail!("--rates must all be finite and greater than zero, got {rate}");
        }
    }
    if !(0.0..1.0).contains(&args.max_shortfall) {
        bail!(
            "--max-shortfall is a fraction below 1, got {}",
            args.max_shortfall
        );
    }
    if !(0.0..=1.0).contains(&args.target_attainment) {
        bail!(
            "--target-attainment is a fraction between 0 and 1, got {}",
            args.target_attainment
        );
    }
    if args.min_rate >= args.max_rate {
        bail!(
            "--min-rate {} must be below --max-rate {}",
            args.min_rate,
            args.max_rate
        );
    }
    Ok(())
}

/// Run the declared rates. No search, and no pretence of one.
async fn run_grid(args: &SweepArgs) -> Result<SweepReport> {
    let mut corpus = CorpusCache::new();
    let mut points = Vec::new();
    let mut rates = args.rates.clone();
    rates.sort_by(f64::total_cmp);

    for rate in rates {
        let (record, reused) = measure(args, rate, &mut corpus).await?;
        points.push(ReportPoint {
            record,
            phase: None,
            crossed: None,
            verdict: None,
            reused,
        });
    }
    Ok(assemble(args, None, points, None))
}

/// Ramp, bisect, densify.
async fn run_adaptive(args: &SweepArgs) -> Result<SweepReport> {
    let boundary = match args.mode {
        Mode::MaxThroughput => Boundary::MaxThroughput {
            max_shortfall: args.max_shortfall,
        },
        Mode::MaxRateUnderSlo => Boundary::MaxRateUnderSlo {
            target_attainment: args.target_attainment,
            metric: match args.attainment_metric {
                AttainmentChoice::Overall => AttainmentMetric::Overall,
                AttainmentChoice::DeclaredDeadline => AttainmentMetric::DeclaredDeadline,
            },
        },
        Mode::Grid => unreachable!("grid does not search"),
    };

    let mut search = Search::new(SearchConfig {
        start_rate: args.start_rate,
        max_rate: args.max_rate,
        min_rate: args.min_rate,
        tolerance: args.tolerance,
        densify_points: args.densify_points,
    });
    let mut corpus = CorpusCache::new();
    let mut points: Vec<ReportPoint> = Vec::new();

    while let Some((rate, phase)) = search.next_rate() {
        let (record, reused) = measure(args, rate, &mut corpus).await?;
        let verdict = boundary.judge(&Measured {
            rate,
            metrics: record.metrics,
        })?;
        eprintln!(
            "sweep | {phase:?} rate={rate:.6}/s crossed={} | {}",
            verdict.crossed, verdict.reason
        );
        points.push(ReportPoint {
            record,
            phase: Some(phase),
            crossed: Some(verdict.crossed),
            verdict: Some(verdict.reason),
            reused,
        });
        search.record(rate, verdict.crossed);
    }

    Ok(assemble(args, Some(boundary), points, Some(search.knee())))
}

/// Measure one rate, or read back the point a previous sweep completed.
async fn measure(
    args: &SweepArgs,
    rate: f64,
    corpus: &mut CorpusCache,
) -> Result<(point::PointRecord, bool)> {
    let directory = point::directory_for(&args.out, rate);
    if !args.no_resume {
        if let Some(record) = point::completed(&directory) {
            eprintln!("sweep | rate={rate:.6}/s reusing {}", directory.display());
            return Ok((record, true));
        }
    }
    let record = point::run(&args.run, rate, &directory, corpus).await?;
    Ok((record, false))
}

fn assemble(
    args: &SweepArgs,
    boundary: Option<Boundary>,
    points: Vec<ReportPoint>,
    knee: Option<Knee>,
) -> SweepReport {
    let mut curve: Vec<CurveEntry> = points
        .iter()
        .map(|point| CurveEntry {
            rate: point.record.rate,
            output_token_throughput_per_s: point.record.metrics.output_token_throughput_per_s,
            request_throughput_per_s: point.record.metrics.request_throughput_per_s,
            ttft_ms_p50: point.record.metrics.ttft_ms_p50,
            ttft_ms_p90: point.record.metrics.ttft_ms_p90,
            tpot_ms_p50: point.record.metrics.tpot_ms_p50,
            tpot_ms_p90: point.record.metrics.tpot_ms_p90,
            slo_attainment: point.record.metrics.slo_attainment,
            declared_deadline_attainment: point.record.metrics.declared_deadline_attainment,
            failed_steps: point.record.metrics.failed_steps,
        })
        .collect();
    curve.sort_by(|left, right| left.rate.total_cmp(&right.rate));

    SweepReport {
        knob: "rate",
        objective: boundary
            .map(|boundary| boundary.objective())
            .unwrap_or("none"),
        boundary,
        config: ReportConfig {
            mode: format!("{:?}", args.mode).to_lowercase(),
            start_rate: args.start_rate,
            max_rate: args.max_rate,
            min_rate: args.min_rate,
            tolerance: args.tolerance,
            densify_points: args.densify_points,
            resumed_points: points.iter().filter(|point| point.reused).count(),
        },
        contamination_warning: contamination_warning(&points),
        points,
        curve,
        knee,
    }
}

/// Say plainly when the curve is not one clean server's answer.
fn contamination_warning(points: &[ReportPoint]) -> Option<String> {
    let unreset = points
        .iter()
        .filter(|point| point.record.cache_reset != point::CacheReset::Done)
        .count();
    if unreset == 0 {
        return None;
    }
    Some(format!(
        "{unreset} of {} points ran without clearing the server's prefix cache, so each started \
         warm on the previous point's content. Prefix-cache rates are not comparable across \
         points, and throughput is affected to the extent this trace reuses content. See each \
         point's `cache_reset`.",
        points.len(),
    ))
}

fn describe(knee: &Knee) -> String {
    match (knee.last_good_rate, knee.first_bad_rate) {
        (Some(last_good), Some(first_bad)) => format!(
            "{:?} between {last_good:.6}/s and {first_bad:.6}/s (bracket {:.1}%)",
            knee.outcome,
            knee.bracket_width.unwrap_or(0.0) * 100.0,
        ),
        (Some(last_good), None) => format!(
            "{:?}: nothing up to {last_good:.6}/s crossed the boundary",
            knee.outcome
        ),
        (None, Some(first_bad)) => format!(
            "{:?}: everything down to {first_bad:.6}/s had already crossed",
            knee.outcome
        ),
        (None, None) => format!("{:?}: nothing was measured", knee.outcome),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> SweepArgs {
        let mut argv = vec![
            "sweep",
            "--out",
            "/tmp/sweep",
            "--trace",
            "trace.csv",
            "--text-file",
            "corpus.txt",
            "--tokenizer",
            "gpt2",
            "--model",
            "model",
        ];
        argv.extend_from_slice(extra);
        SweepArgs::parse_from(argv)
    }

    #[test]
    fn a_hand_set_rate_is_refused_because_the_sweep_owns_that_knob() {
        let error = validate(&parse(&["--rate", "4"])).unwrap_err().to_string();
        assert!(error.contains("--start-rate"), "{error}");
    }

    #[test]
    fn a_grid_with_no_rates_and_a_search_with_rates_are_both_refused() {
        assert!(validate(&parse(&["--mode", "grid"])).is_err());
        let error = validate(&parse(&["--mode", "max-throughput", "--rates", "1,2"]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("grid"), "{error}");
    }

    #[test]
    fn a_sweep_of_dry_runs_is_refused_rather_than_reporting_an_empty_curve() {
        let error = validate(&parse(&["--dry-run"])).unwrap_err().to_string();
        assert!(error.contains("measures nothing"), "{error}");
    }

    #[test]
    fn a_point_that_could_not_reset_the_server_produces_a_warning_naming_the_count() {
        let points = vec![
            report_point(1.0, point::CacheReset::Done),
            report_point(
                2.0,
                point::CacheReset::Unsupported {
                    backend: "sglang-tokens".to_string(),
                },
            ),
        ];

        let warning = contamination_warning(&points).expect("an unreset point must be reported");

        assert!(warning.contains("1 of 2"), "{warning}");
        assert!(contamination_warning(&points[..1]).is_none());
    }

    fn report_point(rate: f64, cache_reset: point::CacheReset) -> ReportPoint {
        ReportPoint {
            record: point::PointRecord {
                rate,
                directory: String::new(),
                cache_reset,
                metrics: Default::default(),
            },
            phase: None,
            crossed: None,
            verdict: None,
            reused: false,
        }
    }
}
