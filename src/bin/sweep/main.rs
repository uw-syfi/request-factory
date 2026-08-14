//! Find the rate at which a server stops doing what you wanted, without
//! guessing where to look.
//!
//! A grid sweep asks you to know the answer before you start: too coarse and the
//! knee sits in a gap, too fine and most of the points are spent far away from
//! it. Every adaptive mode here ramps by doubling and then spends its remaining
//! points on the part of the curve anyone actually reads.
//!
//! **Two of the questions are not the same question.** "What is the highest rate
//! this deployment keeps up with?" is a boundary, and bisection finds it.  "What
//! is the most work it will ever do per second?" is a maximum, it lives *past*
//! that boundary, and it is usually a plateau rather than a point. They get
//! separate modes because they are separate numbers, and on a server whose batch
//! grows with load they can be far apart.
//!
//! Runs happen in this process, which is what lets a sweep of twenty points pay
//! for the tokenizer and the hundred-million-token synthetic corpus once.

mod boundary;
mod peak;
mod point;
mod search;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use req_frontend::{Args, ArrivalMode, CorpusCache};
use serde::Serialize;

use boundary::{AttainmentMetric, Boundary, Measured};
use peak::{Peak, PeakConfig, PeakSearch};
use search::{Knee, Phase, Search, SearchConfig};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Adaptive rate sweep: ramp, bisect the knee, then draw it"
)]
struct SweepArgs {
    /// What this sweep is looking for.
    #[arg(long, value_enum, default_value = "max-sustainable-rate")]
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

    /// `max-sustainable-rate`: how far delivered throughput may fall behind the
    /// offered rate and still count as keeping up.
    #[arg(long, default_value_t = 0.10)]
    max_shortfall: f64,

    /// `peak-throughput`: relative improvement that still counts as rising.
    #[arg(long, default_value_t = 0.03)]
    min_gain: f64,

    /// `peak-throughput`: consecutive non-improving points before the ramp
    /// stops. More than one, so a single noisy run does not end the search.
    #[arg(long, default_value_t = 2)]
    patience: usize,

    /// `peak-throughput`: throughput within this fraction of the peak counts as
    /// being on the plateau.
    #[arg(long, default_value_t = 0.02)]
    plateau_tolerance: f64,

    /// `peak-throughput`: which throughput to maximize.
    #[arg(long, value_enum, default_value = "output-tokens")]
    peak_metric: PeakMetric,

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
    /// The highest rate the server keeps up with: ramp, bisect, densify.
    ///
    /// Answers "how much load can this deployment take?". The boundary is
    /// delivered throughput falling behind the offered rate.
    MaxSustainableRate,
    /// The most work the server will ever do per second, wherever that happens.
    ///
    /// Answers "how much can this deployment produce?". Deliberately keeps
    /// ramping past the point where it stopped keeping up, because that is where
    /// peak throughput lives — and because a server that gets *worse* under more
    /// load is worth catching.
    PeakThroughput,
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

/// Which throughput `peak-throughput` maximizes.
///
/// They peak in different places once output lengths vary: a workload of short
/// answers maximizes requests per second well before it maximizes tokens.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum PeakMetric {
    OutputTokens,
    Requests,
}

impl PeakMetric {
    fn of(self, metrics: &req_frontend::RunMetrics) -> Option<f64> {
        match self {
            Self::OutputTokens => metrics.output_token_throughput_per_s,
            Self::Requests => metrics.request_throughput_per_s,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OutputTokens => "output_token_throughput_per_s",
            Self::Requests => "request_throughput_per_s",
        }
    }
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
    /// Where the server stopped keeping up. Set by the boundary-searching modes.
    knee: Option<Knee>,
    /// The most work the server produced, and over which rates. Set by
    /// `peak-throughput`. Distinct from `knee` on purpose: the peak lives past
    /// the boundary, so the two are different rates and answer different
    /// questions.
    peak: Option<Peak>,
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
    /// Offered load, in workload units per second — sessions for a session
    /// trace. Not the same currency as the throughputs below.
    rate: f64,
    /// The conversion between the two. Multiply `rate` by this to get the
    /// steps per second a server that kept up perfectly would have delivered;
    /// that, not `rate` itself, is what `request_throughput_per_s` is measured
    /// against.
    steps_per_workload_unit: f64,
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
        Mode::PeakThroughput => run_peak(&args).await?,
        Mode::MaxSustainableRate | Mode::MaxRateUnderSlo => run_adaptive(&args).await?,
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
    if let Some(peak) = &report.peak {
        eprintln!("sweep | peak: {}", describe_peak(peak));
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
    Ok(assemble(args, None, points, None, None))
}

/// Ramp, bisect, densify.
async fn run_adaptive(args: &SweepArgs) -> Result<SweepReport> {
    let boundary = match args.mode {
        Mode::MaxSustainableRate => Boundary::MaxSustainableRate {
            max_shortfall: args.max_shortfall,
        },
        Mode::MaxRateUnderSlo => Boundary::MaxRateUnderSlo {
            target_attainment: args.target_attainment,
            metric: match args.attainment_metric {
                AttainmentChoice::Overall => AttainmentMetric::Overall,
                AttainmentChoice::DeclaredDeadline => AttainmentMetric::DeclaredDeadline,
            },
        },
        Mode::Grid | Mode::PeakThroughput => {
            unreachable!("neither runs a boundary search")
        }
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

    Ok(assemble(
        args,
        Some(boundary),
        points,
        Some(search.knee()),
        None,
    ))
}

/// Ramp until throughput stops improving, then draw the top of the curve.
async fn run_peak(args: &SweepArgs) -> Result<SweepReport> {
    let mut search = PeakSearch::new(PeakConfig {
        start_rate: args.start_rate,
        max_rate: args.max_rate,
        min_gain: args.min_gain,
        patience: args.patience,
        densify_points: args.densify_points,
        plateau_tolerance: args.plateau_tolerance,
    });
    let mut corpus = CorpusCache::new();
    let mut points: Vec<ReportPoint> = Vec::new();

    while let Some((rate, phase)) = search.next_rate() {
        let (record, reused) = measure(args, rate, &mut corpus).await?;
        let Some(throughput) = args.peak_metric.of(&record.metrics) else {
            bail!(
                "the run at rate {rate:.6}/s produced no {}, so there is nothing to maximize; \
                 check {}",
                args.peak_metric.name(),
                record.directory,
            );
        };
        eprintln!(
            "sweep | {phase:?} rate={rate:.6}/s {}={throughput:.4}",
            args.peak_metric.name()
        );
        points.push(ReportPoint {
            record,
            phase: Some(phase),
            crossed: None,
            verdict: Some(format!("{}={throughput:.4}", args.peak_metric.name())),
            reused,
        });
        search.record(rate, throughput);
    }

    Ok(assemble(args, None, points, None, Some(search.peak())))
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
    peak: Option<Peak>,
) -> SweepReport {
    let mut curve: Vec<CurveEntry> = points
        .iter()
        .map(|point| CurveEntry {
            rate: point.record.rate,
            steps_per_workload_unit: point.record.metrics.steps_per_workload_unit,
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
        objective: match (boundary, args.mode) {
            (Some(boundary), _) => boundary.objective(),
            (None, Mode::PeakThroughput) => args.peak_metric.name(),
            _ => "none",
        },
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
        peak,
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

fn describe_peak(peak: &Peak) -> String {
    let Some(throughput) = peak.peak_throughput else {
        return format!("{:?}: nothing was measured", peak.outcome);
    };
    let decline = match peak.decline_from_peak {
        // Only worth saying when it is real: a server that got worse under more
        // load is a finding, and noise around zero is not.
        Some(decline) if decline > 0.05 => format!(
            ", falling {:.1}% below it at the highest rate measured",
            decline * 100.0
        ),
        _ => String::new(),
    };
    format!(
        "{:?} {throughput:.4} at {:.6}/s, flat from {:.6}/s to {:.6}/s{decline}",
        peak.outcome,
        peak.peak_rate.unwrap_or(f64::NAN),
        peak.plateau_low_rate.unwrap_or(f64::NAN),
        peak.plateau_high_rate.unwrap_or(f64::NAN),
    )
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
        let error = validate(&parse(&[
            "--mode",
            "max-sustainable-rate",
            "--rates",
            "1,2",
        ]))
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
