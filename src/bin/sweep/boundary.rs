//! What counts as having crossed, and why each mode draws the line where it does.
//!
//! The search shape is shared; this is the only thing that differs between
//! `max-sustainable-rate` and `max-rate-under-slo`.
//!
//! **A boundary judges one point, alone.** It is given the run and nothing else
//! — no history, no neighbours. That restriction is the whole design of this
//! file, because bisection is only meaningful if the answer for a rate does not
//! depend on when that rate was measured. The first draft of `max-sustainable-rate`
//! asked "did throughput rise over the best seen so far", which reads like the
//! textbook definition of saturation and is order-dependent: the same rate
//! judged before and after a higher one gives opposite answers, and everything
//! the bisection then concluded was an artifact of visit order. Taking history
//! out of the signature makes that class of mistake unrepresentable.

use anyhow::{bail, Result};
use req_frontend::RunMetrics;
use serde::Serialize;

/// The measured half of one sweep point.
#[derive(Debug, Clone, Copy)]
pub struct Measured {
    pub rate: f64,
    pub metrics: RunMetrics,
}

/// Which question this sweep is asking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum Boundary {
    /// Crossed when the server stopped keeping up with what was offered.
    ///
    /// Stated as *delivered against offered* rather than "throughput stopped
    /// rising", because only the former is a property of the single point being
    /// judged. It is also the more direct statement of the thing: a saturated
    /// server is one that cannot complete work as fast as it arrives.
    ///
    /// The two sides are counted in different things and must be converted
    /// before they can be compared. `--rate` offers *workload units* per second
    /// — sessions for a session trace — while throughput is delivered *steps*
    /// per second, and a session issues several rounds. So the reference is
    /// `rate * steps_per_workload_unit`: what a server that kept up perfectly
    /// would have had to deliver. Comparing the raw numbers instead reads a
    /// saturated server as keeping up with room to spare, by exactly the mean
    /// rounds per session.
    ///
    /// One measurement artifact to know about. The run window runs from the
    /// first submission to the last completion, so it always includes one
    /// request's latency after the last arrival. At rates where the trace's
    /// whole arrival span shrinks toward that single latency, delivered
    /// throughput falls short of offered even on a server that kept up
    /// perfectly. Use enough workload units that the span dominates the tail —
    /// the shortfall from this effect is roughly `latency * rate / units`.
    MaxSustainableRate {
        /// How far delivered throughput may fall behind the offered rate and
        /// still count as keeping up. 0.10 means "within 10%".
        max_shortfall: f64,
    },
    /// Crossed when SLO attainment fell below the target.
    MaxRateUnderSlo {
        target_attainment: f64,
        metric: AttainmentMetric,
    },
}

/// Which attainment number the SLO boundary watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttainmentMetric {
    /// Every bound in force, from `--slo` or the trace's sidecar.
    Overall,
    /// Only the per-request deadlines the trace declared for itself.
    DeclaredDeadline,
}

/// Why a point was judged as it was, in words a reader can check.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub crossed: bool,
    pub reason: String,
}

impl Boundary {
    /// Judge one point on its own.
    pub fn judge(&self, candidate: &Measured) -> Result<Verdict> {
        match *self {
            Self::MaxSustainableRate { max_shortfall } => {
                let Some(delivered) = candidate.metrics.request_throughput_per_s else {
                    bail!(
                        "the run at rate {:.6}/s completed no requests, so there is no delivered \
                         throughput to compare against the offered rate; check its point directory",
                        candidate.rate
                    );
                };
                let steps_per_unit = candidate.metrics.steps_per_workload_unit;
                if !(steps_per_unit.is_finite() && steps_per_unit > 0.0) {
                    bail!(
                        "the run at rate {:.6}/s reported {steps_per_unit} steps per workload \
                         unit, so its offered rate cannot be converted into the steps its \
                         throughput is counted in; check its point directory",
                        candidate.rate
                    );
                }
                let offered_steps = candidate.rate * steps_per_unit;
                let required = offered_steps * (1.0 - max_shortfall);
                Ok(Verdict {
                    crossed: delivered < required,
                    reason: format!(
                        "delivered {delivered:.4} steps/s against {:.4} units/s offered × \
                         {steps_per_unit:.4} steps/unit = {offered_steps:.4} steps/s; keeping up \
                         needs {required:.4} (within {:.0}%)",
                        candidate.rate,
                        max_shortfall * 100.0,
                    ),
                })
            }
            Self::MaxRateUnderSlo {
                target_attainment,
                metric,
            } => {
                let measured = match metric {
                    AttainmentMetric::Overall => candidate.metrics.slo_attainment,
                    AttainmentMetric::DeclaredDeadline => {
                        candidate.metrics.declared_deadline_attainment
                    }
                };
                let Some(attainment) = measured else {
                    bail!(
                        "the run at rate {:.6}/s reported no {} attainment, so this sweep has no \
                         boundary to search for. Give the run an objective with --slo, a \
                         `<trace>.slo.json` sidecar, or a trace declaring the slo tag.",
                        candidate.rate,
                        match metric {
                            AttainmentMetric::Overall => "SLO",
                            AttainmentMetric::DeclaredDeadline => "declared-deadline",
                        },
                    );
                };
                Ok(Verdict {
                    crossed: attainment < target_attainment,
                    reason: format!(
                        "attainment {attainment:.4} against a target of {target_attainment:.4}"
                    ),
                })
            }
        }
    }

    /// The quantity this sweep is maximizing, named for the report.
    pub fn objective(&self) -> &'static str {
        match self {
            Self::MaxSustainableRate { .. } => "request_throughput_per_s",
            Self::MaxRateUnderSlo { .. } => "slo_attainment",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An independent trace: one step per workload unit, so offered and
    /// delivered are already in the same currency.
    fn delivering(rate: f64, delivered: f64) -> Measured {
        delivering_rounds(rate, delivered, 1.0)
    }

    fn delivering_rounds(rate: f64, delivered: f64, steps_per_workload_unit: f64) -> Measured {
        Measured {
            rate,
            metrics: RunMetrics {
                request_throughput_per_s: Some(delivered),
                steps_per_workload_unit,
                ..RunMetrics::default()
            },
        }
    }

    fn attaining(rate: f64, attainment: Option<f64>) -> Measured {
        Measured {
            rate,
            metrics: RunMetrics {
                slo_attainment: attainment,
                ..RunMetrics::default()
            },
        }
    }

    #[test]
    fn saturation_is_delivered_falling_behind_offered() {
        let boundary = Boundary::MaxSustainableRate {
            max_shortfall: 0.10,
        };

        // Keeping up, within the tail allowance.
        let keeping_up = boundary.judge(&delivering(40.0, 39.5)).unwrap();
        assert!(!keeping_up.crossed, "{}", keeping_up.reason);

        // Offered twice the capacity; delivery capped.
        let saturated = boundary.judge(&delivering(100.0, 50.0)).unwrap();
        assert!(saturated.crossed, "{}", saturated.reason);
    }

    #[test]
    fn a_rate_is_judged_the_same_whenever_it_is_measured() {
        // The property bisection depends on. An earlier draft compared against
        // the best throughput seen so far, so measuring 50/s before or after a
        // 200/s point gave opposite answers and the located knee was an artifact
        // of visit order.
        let boundary = Boundary::MaxSustainableRate {
            max_shortfall: 0.10,
        };
        let point = delivering(60.0, 50.0);

        let first = boundary.judge(&point).unwrap();
        let _ = boundary.judge(&delivering(400.0, 50.0)).unwrap();
        let again = boundary.judge(&point).unwrap();

        assert_eq!(first.crossed, again.crossed);
        assert_eq!(first.reason, again.reason);
    }

    #[test]
    fn the_shortfall_allowance_is_what_separates_the_two_sides() {
        let strict = Boundary::MaxSustainableRate {
            max_shortfall: 0.01,
        };
        let loose = Boundary::MaxSustainableRate {
            max_shortfall: 0.20,
        };
        let point = delivering(100.0, 90.0);

        assert!(strict.judge(&point).unwrap().crossed);
        assert!(!loose.judge(&point).unwrap().crossed);
    }

    /// The defect this conversion exists to prevent, in the numbers that
    /// exposed it: a stub server plateauing at ~129 rounds/s under a trace
    /// averaging 2.01 rounds per session. Offered 40 sessions/s it is keeping
    /// up; offered 80 it is not. Comparing the raw numbers called both of them
    /// comfortable and moved the reported knee to 145 sessions/s — more than
    /// twice the truth, because 129 rounds/s *is* 64 sessions/s.
    #[test]
    fn a_session_trace_is_judged_in_rounds_offered_not_sessions_offered() {
        let boundary = Boundary::MaxSustainableRate {
            max_shortfall: 0.10,
        };
        let rounds_per_session = 2.01;

        let below = boundary
            .judge(&delivering_rounds(40.0, 80.0, rounds_per_session))
            .unwrap();
        let above = boundary
            .judge(&delivering_rounds(80.0, 129.6, rounds_per_session))
            .unwrap();

        assert!(!below.crossed, "{}", below.reason);
        assert!(above.crossed, "{}", above.reason);
        // Raw, the same point reads as delivering 129.6 against 80 offered.
        assert!(!boundary.judge(&delivering(80.0, 129.6)).unwrap().crossed);
    }

    #[test]
    fn a_ratio_that_cannot_convert_the_rate_fails_rather_than_guessing_one() {
        // Zero is what an older point record deserializes to. Treating it as 1
        // would apply the independent-trace rule to a session trace silently.
        let boundary = Boundary::MaxSustainableRate {
            max_shortfall: 0.10,
        };

        assert!(boundary.judge(&delivering_rounds(10.0, 20.0, 0.0)).is_err());
    }

    #[test]
    fn a_point_that_completed_nothing_fails_rather_than_reading_as_saturated() {
        // "Crossed" would be the plausible reading, and it would let a broken
        // server or a misconfigured backend masquerade as a located knee.
        let boundary = Boundary::MaxSustainableRate {
            max_shortfall: 0.10,
        };
        let candidate = Measured {
            rate: 10.0,
            metrics: RunMetrics::default(),
        };

        assert!(boundary.judge(&candidate).is_err());
    }

    #[test]
    fn the_slo_boundary_is_a_threshold_on_the_point_alone() {
        let boundary = Boundary::MaxRateUnderSlo {
            target_attainment: 0.99,
            metric: AttainmentMetric::Overall,
        };

        assert!(
            !boundary
                .judge(&attaining(2.0, Some(0.995)))
                .unwrap()
                .crossed
        );
        assert!(boundary.judge(&attaining(2.0, Some(0.98))).unwrap().crossed);
    }

    #[test]
    fn an_slo_sweep_with_no_objective_fails_rather_than_searching_for_nothing() {
        // Every point would report `None`, which read as "not crossed" would ramp
        // to the ceiling and announce a knee that was never tested for.
        let boundary = Boundary::MaxRateUnderSlo {
            target_attainment: 0.99,
            metric: AttainmentMetric::Overall,
        };

        let error = boundary
            .judge(&attaining(1.0, None))
            .unwrap_err()
            .to_string();

        assert!(error.contains("--slo"), "{error}");
    }

    #[test]
    fn the_declared_deadline_metric_is_watched_separately_from_the_overall_one() {
        let boundary = Boundary::MaxRateUnderSlo {
            target_attainment: 0.9,
            metric: AttainmentMetric::DeclaredDeadline,
        };
        let candidate = Measured {
            rate: 1.0,
            // Overall is fine; the trace's own deadlines are not.
            metrics: RunMetrics {
                slo_attainment: Some(1.0),
                declared_deadline_attainment: Some(0.5),
                ..RunMetrics::default()
            },
        };

        assert!(boundary.judge(&candidate).unwrap().crossed);
    }
}
