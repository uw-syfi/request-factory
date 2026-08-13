//! Per-metric service-level objectives, and what fraction of a run met them.
//!
//! An SLO here is a set of **upper bounds**, one per metric, and the number it
//! produces is an **attainment rate**: the fraction of steps that met every
//! bound declared. Not a percentile target ("p99 TTFT under 500 ms") — that
//! phrasing hides how many requests were bad, and a run of two hundred thousand
//! rounds has a lot of room to hide things in.
//!
//! This lives beside the schemas rather than in the runtime because a measured
//! replay and a simulated run must report the same number for the same trace.
//! An objective that means one thing to the client and another to the simulator
//! would make the comparison they exist for meaningless.

use std::fmt;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Upper bounds a step must meet to count as attained.
///
/// Every field is optional and every present field is a ceiling. An empty spec
/// is not the same as no spec: `Option<SloSpec>` is how a run says it declared
/// no objective at all and reports nothing, whereas an empty `SloSpec` would
/// declare an objective that everything trivially meets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloSpec {
    /// Time to first token, from the moment the request was sent.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ttft_ms: Option<f64>,
    /// Client-observed delivery time per output token after the first timed
    /// event. The steady-state pace a reader of the response experiences.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tpot_ms: Option<f64>,
    /// End to end, submission to completion.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub e2e_ms: Option<f64>,
}

impl SloSpec {
    pub const METRICS: &'static [&'static str] = &["ttft_ms", "tpot_ms", "e2e_ms"];

    /// Parse `ttft_ms=500,tpot_ms=50`.
    ///
    /// Strict about unknown names rather than ignoring them: a misspelled metric
    /// silently declares no objective, and a run that reports 100% attainment
    /// because it was asked for nothing is the worst possible failure here.
    pub fn parse(spec: &str) -> Result<Self> {
        let mut parsed = Self::default();
        for clause in spec.split(',') {
            let clause = clause.trim();
            if clause.is_empty() {
                continue;
            }
            let (name, value) = clause
                .split_once('=')
                .with_context(|| format!("SLO clause {clause:?} is not `metric=milliseconds`"))?;
            let name = name.trim();
            let bound: f64 = value.trim().parse().with_context(|| {
                format!("SLO bound for {name:?} must be a number of milliseconds")
            })?;
            if !bound.is_finite() || bound <= 0.0 {
                bail!("SLO bound for {name:?} must be finite and greater than zero, got {bound}");
            }
            let slot = match name {
                "ttft_ms" => &mut parsed.ttft_ms,
                "tpot_ms" => &mut parsed.tpot_ms,
                "e2e_ms" => &mut parsed.e2e_ms,
                other => bail!(
                    "unknown SLO metric {other:?} (expected one of {:?})",
                    Self::METRICS
                ),
            };
            if slot.is_some() {
                bail!("SLO metric {name:?} is declared more than once");
            }
            *slot = Some(bound);
        }
        if parsed.is_empty() {
            bail!(
                "an SLO must declare at least one bound (one of {:?})",
                Self::METRICS
            );
        }
        Ok(parsed)
    }

    pub fn is_empty(&self) -> bool {
        self.bounds().all(|(_, bound)| bound.is_none())
    }

    /// Each metric with its declared bound, in a fixed order so every report
    /// lists them the same way.
    pub fn bounds(&self) -> impl Iterator<Item = (&'static str, Option<f64>)> + '_ {
        [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("e2e_ms", self.e2e_ms),
        ]
        .into_iter()
    }
}

impl fmt::Display for SloSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let declared: Vec<String> = self
            .bounds()
            .filter_map(|(name, bound)| bound.map(|bound| format!("{name}<={bound}")))
            .collect();
        write!(formatter, "{}", declared.join(","))
    }
}

/// What one step measured, as the objective sees it.
#[derive(Clone, Copy, Debug, Default)]
pub struct SloMeasurement {
    /// False for a failed or skipped step. Such a step is never attained: a
    /// response that did not arrive did not arrive on time either.
    pub succeeded: bool,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
}

/// One metric's outcome for one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricVerdict {
    /// The step met the bound.
    Attained,
    /// The step measured past the bound.
    Violated,
    /// No number to judge — the step failed, or it succeeded without producing
    /// this metric (a single-event response has no measurable TPOT).
    ///
    /// Counted against attainment, but counted *separately* as well, so nobody
    /// reads "94% attained" as "6% were slow" when it was "6% never answered".
    Unmeasured,
}

/// Running attainment counts for one metric.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MetricAttainment {
    pub threshold_ms: f64,
    pub evaluated_steps: usize,
    pub attained_steps: usize,
    pub violated_steps: usize,
    pub unmeasured_steps: usize,
    /// `attained_steps / evaluated_steps`. `None` before any step is evaluated.
    pub attainment: Option<f64>,
}

impl MetricAttainment {
    fn new(threshold_ms: f64) -> Self {
        Self {
            threshold_ms,
            evaluated_steps: 0,
            attained_steps: 0,
            violated_steps: 0,
            unmeasured_steps: 0,
            attainment: None,
        }
    }

    fn add(&mut self, verdict: MetricVerdict) {
        self.evaluated_steps += 1;
        match verdict {
            MetricVerdict::Attained => self.attained_steps += 1,
            MetricVerdict::Violated => self.violated_steps += 1,
            MetricVerdict::Unmeasured => self.unmeasured_steps += 1,
        }
        self.attainment = Some(self.attained_steps as f64 / self.evaluated_steps as f64);
    }
}

/// Attainment for a whole run.
#[derive(Clone, Debug, Serialize)]
pub struct SloSummary {
    /// The objective this run was held to, echoed so a report is readable
    /// without the command line that produced it.
    pub spec: SloSpec,
    /// Where the objective came from, for the same reason.
    pub source: SloSource,
    pub evaluated_steps: usize,
    /// Steps that met **every** declared bound.
    pub attained_steps: usize,
    /// `attained_steps / evaluated_steps`. The one number this exists to report.
    pub attainment: Option<f64>,
    pub ttft_ms: Option<MetricAttainment>,
    pub tpot_ms: Option<MetricAttainment>,
    pub e2e_ms: Option<MetricAttainment>,
}

/// Which scope declared the objective in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SloSource {
    /// Named on the command line, applying to whatever trace this run replays.
    Global,
    /// Declared by the trace itself, in the sidecar beside it.
    Trace,
}

impl SloSummary {
    pub fn new(spec: SloSpec, source: SloSource) -> Self {
        Self {
            spec,
            source,
            evaluated_steps: 0,
            attained_steps: 0,
            attainment: None,
            ttft_ms: spec.ttft_ms.map(MetricAttainment::new),
            tpot_ms: spec.tpot_ms.map(MetricAttainment::new),
            e2e_ms: spec.e2e_ms.map(MetricAttainment::new),
        }
    }

    /// Fold one step's measurements.
    pub fn add(&mut self, measurement: &SloMeasurement) {
        let verdicts = [
            (&mut self.ttft_ms, measurement.ttft_ms),
            (&mut self.tpot_ms, measurement.tpot_ms),
            (&mut self.e2e_ms, measurement.e2e_ms),
        ];
        let mut attained_every_bound = true;
        for (metric, measured) in verdicts {
            let Some(metric) = metric.as_mut() else {
                continue;
            };
            let verdict = judge(measurement.succeeded, measured, metric.threshold_ms);
            attained_every_bound &= verdict == MetricVerdict::Attained;
            metric.add(verdict);
        }
        self.evaluated_steps += 1;
        if attained_every_bound {
            self.attained_steps += 1;
        }
        self.attainment = Some(self.attained_steps as f64 / self.evaluated_steps as f64);
    }

    /// A one-line report, for the run's own stderr.
    pub fn describe(&self) -> String {
        let overall = match self.attainment {
            Some(rate) => format!("{:.4}", rate),
            None => "n/a".to_string(),
        };
        let per_metric: Vec<String> = [
            ("ttft_ms", self.ttft_ms.as_ref()),
            ("tpot_ms", self.tpot_ms.as_ref()),
            ("e2e_ms", self.e2e_ms.as_ref()),
        ]
        .into_iter()
        .filter_map(|(name, metric)| {
            let metric = metric?;
            Some(format!(
                "{name}<={} {:.4} ({} violated, {} unmeasured)",
                metric.threshold_ms,
                metric.attainment.unwrap_or(0.0),
                metric.violated_steps,
                metric.unmeasured_steps,
            ))
        })
        .collect();
        format!(
            "slo attainment | source={:?} steps={} overall={overall} | {}",
            self.source,
            self.evaluated_steps,
            per_metric.join(" | "),
        )
    }
}

fn judge(succeeded: bool, measured: Option<f64>, threshold_ms: f64) -> MetricVerdict {
    if !succeeded {
        return MetricVerdict::Unmeasured;
    }
    match measured {
        Some(value) if value <= threshold_ms => MetricVerdict::Attained,
        Some(_) => MetricVerdict::Violated,
        None => MetricVerdict::Unmeasured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn met(ttft: f64, tpot: f64, e2e: f64) -> SloMeasurement {
        SloMeasurement {
            succeeded: true,
            ttft_ms: Some(ttft),
            tpot_ms: Some(tpot),
            e2e_ms: Some(e2e),
        }
    }

    #[test]
    fn a_spec_parses_the_metrics_it_names_and_nothing_else() {
        let spec = SloSpec::parse("ttft_ms=500, tpot_ms=50").unwrap();

        assert_eq!(spec.ttft_ms, Some(500.0));
        assert_eq!(spec.tpot_ms, Some(50.0));
        assert_eq!(spec.e2e_ms, None);
        assert_eq!(spec.to_string(), "ttft_ms<=500,tpot_ms<=50");
    }

    #[test]
    fn a_misspelled_metric_is_an_error_rather_than_an_objective_of_nothing() {
        // The worst outcome here is a run that reports 100% attainment because
        // it was quietly asked for nothing.
        let err = SloSpec::parse("ttft=500").unwrap_err().to_string();
        assert!(err.contains("unknown SLO metric"), "{err}");

        assert!(SloSpec::parse("").is_err());
        assert!(SloSpec::parse("ttft_ms=500,ttft_ms=600").is_err());
        assert!(SloSpec::parse("ttft_ms=-1").is_err());
        assert!(SloSpec::parse("ttft_ms=abc").is_err());
        assert!(SloSpec::parse("ttft_ms").is_err());
    }

    #[test]
    fn a_step_is_attained_only_when_every_declared_bound_holds() {
        let spec = SloSpec::parse("ttft_ms=500,tpot_ms=50").unwrap();
        let mut summary = SloSummary::new(spec, SloSource::Global);

        summary.add(&met(400.0, 40.0, 9_999.0)); // both bounds met; e2e undeclared
        summary.add(&met(400.0, 60.0, 1.0)); // tpot over
        summary.add(&met(600.0, 40.0, 1.0)); // ttft over

        assert_eq!(summary.evaluated_steps, 3);
        assert_eq!(summary.attained_steps, 1);
        assert_eq!(summary.attainment, Some(1.0 / 3.0));
        assert_eq!(summary.ttft_ms.unwrap().violated_steps, 1);
        assert_eq!(summary.tpot_ms.unwrap().violated_steps, 1);
        // An undeclared metric is not evaluated at all, however slow it was.
        assert!(summary.e2e_ms.is_none());
    }

    #[test]
    fn a_bound_is_a_ceiling_that_the_boundary_itself_satisfies() {
        let mut summary =
            SloSummary::new(SloSpec::parse("ttft_ms=500").unwrap(), SloSource::Global);
        summary.add(&met(500.0, 0.0, 0.0));

        assert_eq!(summary.attained_steps, 1);
    }

    #[test]
    fn a_failed_step_is_unmeasured_rather_than_fast() {
        // It failed at 1 ms. That is not an attained SLO, and it is also not a
        // violated latency bound -- calling it either would misreport the run.
        let mut summary =
            SloSummary::new(SloSpec::parse("ttft_ms=500").unwrap(), SloSource::Global);
        summary.add(&SloMeasurement {
            succeeded: false,
            ttft_ms: Some(1.0),
            ..SloMeasurement::default()
        });

        let ttft = summary.ttft_ms.unwrap();
        assert_eq!(ttft.unmeasured_steps, 1);
        assert_eq!(ttft.violated_steps, 0);
        assert_eq!(ttft.attained_steps, 0);
        assert_eq!(summary.attainment, Some(0.0));
    }

    #[test]
    fn a_successful_step_with_no_measurable_tpot_counts_against_attainment_and_says_why() {
        // A one-event response has no per-token pace to measure. It still fails
        // an objective that asked for one, but the count says it was unmeasured
        // rather than slow.
        let mut summary = SloSummary::new(SloSpec::parse("tpot_ms=50").unwrap(), SloSource::Global);
        summary.add(&SloMeasurement {
            succeeded: true,
            ttft_ms: Some(10.0),
            tpot_ms: None,
            e2e_ms: Some(20.0),
        });

        let tpot = summary.tpot_ms.unwrap();
        assert_eq!(tpot.unmeasured_steps, 1);
        assert_eq!(tpot.violated_steps, 0);
        assert_eq!(summary.attainment, Some(0.0));
    }

    #[test]
    fn a_spec_round_trips_through_json_without_inventing_absent_bounds() {
        let spec = SloSpec::parse("e2e_ms=30000").unwrap();
        let json = serde_json::to_string(&spec).unwrap();

        assert_eq!(json, r#"{"e2e_ms":30000.0}"#);
        assert_eq!(serde_json::from_str::<SloSpec>(&json).unwrap(), spec);
        assert!(serde_json::from_str::<SloSpec>(r#"{"ttft":1.0}"#).is_err());
    }
}
