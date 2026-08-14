//! One search shape, three uses.
//!
//! Every adaptive sweep here asks the same question — *where does the knob stop
//! buying what it was buying?* — and answers it the same way: **ramp** by
//! doubling until the boundary flips, **bisect** back to locate it, then
//! **densify** around it, because a curve whose interesting part is a single
//! bracketed gap is not a curve.
//!
//! Only the predicate differs, and it lives outside this module. That is
//! deliberate: "throughput stopped rising" needs the whole history to judge,
//! "attainment fell below target" needs only the point, and folding either one
//! in here would make the search shape harder to test than the thing it drives.
//!
//! Nothing in this file runs a request. It is a state machine over rates, so the
//! search can be tested against an analytic server whose knee is known exactly.

use serde::Serialize;

/// How the search should behave once it is running.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// First rate to try. The ramp goes up from here, or down if even this
    /// already crossed.
    pub start_rate: f64,
    /// Never offer more than this. A sweep that would otherwise keep doubling
    /// into a server that no longer answers stops and says it never found the
    /// boundary, which is a different finding from having found it.
    pub max_rate: f64,
    /// Never offer less than this while ramping down.
    pub min_rate: f64,
    /// Stop bisecting when the bracket is this narrow, relative to its upper
    /// end. 0.05 locates the knee to within 5%.
    pub tolerance: f64,
    /// Extra points spread across the located bracket, so the knee is drawn
    /// rather than merely bounded.
    pub densify_points: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            start_rate: 1.0,
            max_rate: 4_096.0,
            min_rate: 1.0 / 1_024.0,
            tolerance: 0.05,
            densify_points: 3,
        }
    }
}

/// Which part of the search a rate was offered by.
///
/// Recorded per point so a reader can tell a rate that bounded the knee from
/// one that merely got the search there. Both are real measurements; only one
/// of them is evidence about the knee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Ramp,
    Bisect,
    Densify,
}

/// Why the search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The boundary was bracketed and narrowed to the tolerance.
    Located,
    /// Every rate up to `max_rate` stayed on the good side. The knee is above
    /// the range searched, not absent.
    NeverCrossed,
    /// Every rate down to `min_rate` was already past the boundary. The knee is
    /// below the range searched — usually a server that cannot serve this trace
    /// at any rate, or an objective nothing could meet.
    AlwaysCrossed,
}

/// The located boundary, in the terms a reader can act on.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Knee {
    pub outcome: Outcome,
    /// Highest rate measured that stayed on the good side.
    pub last_good_rate: Option<f64>,
    /// Lowest rate measured that crossed.
    pub first_bad_rate: Option<f64>,
    /// `(first_bad - last_good) / first_bad`. How tightly the knee is pinned.
    pub bracket_width: Option<f64>,
}

/// Ramp, bisect, densify — as a state machine over rates.
///
/// Driven by the caller: [`next_rate`](Self::next_rate) says what to run,
/// [`record`](Self::record) says how it went. The search never runs anything
/// itself, which is what makes it testable.
pub struct Search {
    config: SearchConfig,
    phase: Phase,
    /// `None` before the first rate is offered.
    pending: Option<f64>,
    last_good: Option<f64>,
    first_bad: Option<f64>,
    /// Set once the ramp turned downward, so the two directions cannot fight.
    ramping_down: bool,
    densify_queue: Vec<f64>,
    outcome: Option<Outcome>,
}

impl Search {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            config,
            phase: Phase::Ramp,
            pending: Some(config.start_rate.clamp(config.min_rate, config.max_rate)),
            last_good: None,
            first_bad: None,
            ramping_down: false,
            densify_queue: Vec::new(),
            outcome: None,
        }
    }

    /// The next rate to measure, and which phase asked for it.
    pub fn next_rate(&mut self) -> Option<(f64, Phase)> {
        self.pending.map(|rate| (rate, self.phase))
    }

    /// Feed back whether that rate crossed the boundary.
    pub fn record(&mut self, rate: f64, crossed: bool) {
        match self.phase {
            Phase::Ramp => self.record_ramp(rate, crossed),
            Phase::Bisect => {
                if crossed {
                    self.first_bad = Some(rate);
                } else {
                    self.last_good = Some(rate);
                }
                self.advance_bisect();
            }
            Phase::Densify => {
                // Densification measures the curve; it does not move the
                // bracket. A densified point that disagrees with the bracket is
                // information about noise, and silently re-narrowing on it would
                // hide that.
                self.pending = self.densify_queue.pop();
                if self.pending.is_none() {
                    self.finish();
                }
            }
        }
    }

    fn record_ramp(&mut self, rate: f64, crossed: bool) {
        if crossed {
            self.first_bad = Some(match self.first_bad {
                Some(current) => current.min(rate),
                None => rate,
            });
            if self.last_good.is_some() {
                self.begin_bisect();
                return;
            }
            // The very first rate already crossed: ramp *down* to find a rate
            // that does not, rather than reporting a knee below everything
            // measured on no evidence.
            self.ramping_down = true;
            let next = rate / 2.0;
            if next < self.config.min_rate {
                self.outcome = Some(Outcome::AlwaysCrossed);
                self.pending = None;
                return;
            }
            self.pending = Some(next);
            return;
        }

        self.last_good = Some(match self.last_good {
            Some(current) => current.max(rate),
            None => rate,
        });
        if self.ramping_down {
            // Found the good side below a bad one: the bracket is complete.
            self.begin_bisect();
            return;
        }
        let next = rate * 2.0;
        if next > self.config.max_rate {
            self.outcome = Some(Outcome::NeverCrossed);
            self.pending = None;
            return;
        }
        self.pending = Some(next);
    }

    fn begin_bisect(&mut self) {
        self.phase = Phase::Bisect;
        self.advance_bisect();
    }

    fn advance_bisect(&mut self) {
        let (Some(last_good), Some(first_bad)) = (self.last_good, self.first_bad) else {
            self.finish();
            return;
        };
        if bracket_width(last_good, first_bad) <= self.config.tolerance {
            self.begin_densify();
            return;
        }
        let midpoint = (last_good + first_bad) / 2.0;
        // Guard against a bracket so narrow that the midpoint is one of its own
        // ends in floating point: that would re-measure a rate forever.
        if midpoint <= last_good || midpoint >= first_bad {
            self.begin_densify();
            return;
        }
        self.pending = Some(midpoint);
    }

    fn begin_densify(&mut self) {
        self.phase = Phase::Densify;
        self.densify_queue = self.densify_rates();
        self.densify_queue.reverse(); // popped from the back, so run ascending
        self.pending = self.densify_queue.pop();
        if self.pending.is_none() {
            self.finish();
        }
    }

    /// Points spread across the bracket and a little beyond it.
    ///
    /// A little beyond on purpose: the knee is a shape, and a reader needs to
    /// see the curve bend rather than only the two rates that trapped it.
    fn densify_rates(&self) -> Vec<f64> {
        let (Some(last_good), Some(first_bad)) = (self.last_good, self.first_bad) else {
            return Vec::new();
        };
        if self.config.densify_points == 0 {
            return Vec::new();
        }
        let low = (last_good * 0.75).max(self.config.min_rate);
        let high = (first_bad * 1.25).min(self.config.max_rate);
        if high <= low {
            return Vec::new();
        }
        let count = self.config.densify_points;
        (1..=count)
            .map(|index| low + (high - low) * index as f64 / (count + 1) as f64)
            .collect()
    }

    fn finish(&mut self) {
        self.pending = None;
        if self.outcome.is_none() {
            self.outcome = Some(match (self.last_good, self.first_bad) {
                (Some(_), Some(_)) => Outcome::Located,
                (Some(_), None) => Outcome::NeverCrossed,
                _ => Outcome::AlwaysCrossed,
            });
        }
    }

    /// What the search concluded. Meaningful once `next_rate` returns `None`.
    pub fn knee(&self) -> Knee {
        Knee {
            outcome: self
                .outcome
                .unwrap_or(match (self.last_good, self.first_bad) {
                    (Some(_), Some(_)) => Outcome::Located,
                    (Some(_), None) => Outcome::NeverCrossed,
                    _ => Outcome::AlwaysCrossed,
                }),
            last_good_rate: self.last_good,
            first_bad_rate: self.first_bad,
            bracket_width: match (self.last_good, self.first_bad) {
                (Some(last_good), Some(first_bad)) => Some(bracket_width(last_good, first_bad)),
                _ => None,
            },
        }
    }
}

fn bracket_width(last_good: f64, first_bad: f64) -> f64 {
    if first_bad <= 0.0 {
        return 0.0;
    }
    ((first_bad - last_good) / first_bad).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the search against a server whose knee is known exactly.
    fn run(config: SearchConfig, knee: f64) -> (Vec<(f64, Phase)>, Knee) {
        let mut search = Search::new(config);
        let mut visited = Vec::new();
        while let Some((rate, phase)) = search.next_rate() {
            visited.push((rate, phase));
            assert!(
                visited.len() < 200,
                "the search did not terminate: {visited:?}"
            );
            search.record(rate, rate > knee);
        }
        (visited, search.knee())
    }

    #[test]
    fn the_ramp_doubles_and_the_bisection_pins_the_knee_to_the_tolerance() {
        let config = SearchConfig {
            start_rate: 1.0,
            tolerance: 0.05,
            densify_points: 0,
            ..SearchConfig::default()
        };
        let (visited, knee) = run(config, 10.0);

        let ramp: Vec<f64> = visited
            .iter()
            .filter(|(_, phase)| *phase == Phase::Ramp)
            .map(|(rate, _)| *rate)
            .collect();
        assert_eq!(ramp, vec![1.0, 2.0, 4.0, 8.0, 16.0]);

        assert_eq!(knee.outcome, Outcome::Located);
        let last_good = knee.last_good_rate.unwrap();
        let first_bad = knee.first_bad_rate.unwrap();
        assert!(last_good <= 10.0 && first_bad > 10.0, "{knee:?}");
        assert!(knee.bracket_width.unwrap() <= 0.05, "{knee:?}");
    }

    #[test]
    fn a_tighter_tolerance_costs_points_and_buys_a_narrower_bracket() {
        let loose = run(
            SearchConfig {
                tolerance: 0.2,
                densify_points: 0,
                ..SearchConfig::default()
            },
            10.0,
        );
        let tight = run(
            SearchConfig {
                tolerance: 0.01,
                densify_points: 0,
                ..SearchConfig::default()
            },
            10.0,
        );

        assert!(tight.0.len() > loose.0.len());
        assert!(tight.1.bracket_width.unwrap() < loose.1.bracket_width.unwrap());
    }

    #[test]
    fn a_knee_above_the_ceiling_is_reported_as_never_found_not_as_the_ceiling() {
        // The distinction matters: "the server never saturated below 64/s" is a
        // result, and quietly returning 64 as the knee would be a fabrication.
        let config = SearchConfig {
            start_rate: 1.0,
            max_rate: 64.0,
            densify_points: 0,
            ..SearchConfig::default()
        };
        let (_, knee) = run(config, 10_000.0);

        assert_eq!(knee.outcome, Outcome::NeverCrossed);
        assert_eq!(knee.first_bad_rate, None);
        assert_eq!(knee.last_good_rate, Some(64.0));
        assert!(knee.bracket_width.is_none());
    }

    #[test]
    fn a_first_point_that_already_crossed_ramps_downward_instead_of_giving_up() {
        let config = SearchConfig {
            start_rate: 100.0,
            tolerance: 0.05,
            densify_points: 0,
            ..SearchConfig::default()
        };
        let (visited, knee) = run(config, 3.0);

        let ramp: Vec<f64> = visited
            .iter()
            .filter(|(_, phase)| *phase == Phase::Ramp)
            .map(|(rate, _)| *rate)
            .collect();
        assert_eq!(ramp, vec![100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625]);
        assert_eq!(knee.outcome, Outcome::Located);
        assert!(knee.last_good_rate.unwrap() <= 3.0);
        assert!(knee.first_bad_rate.unwrap() > 3.0);
    }

    #[test]
    fn a_boundary_below_every_rate_searched_says_so_rather_than_bisecting_nothing() {
        let config = SearchConfig {
            start_rate: 4.0,
            min_rate: 1.0,
            densify_points: 0,
            ..SearchConfig::default()
        };
        let (_, knee) = run(config, 0.0);

        assert_eq!(knee.outcome, Outcome::AlwaysCrossed);
        assert_eq!(knee.last_good_rate, None);
        assert_eq!(knee.first_bad_rate, Some(1.0));
    }

    #[test]
    fn densification_fills_the_bracket_and_a_little_past_both_sides_of_it() {
        let config = SearchConfig {
            start_rate: 1.0,
            tolerance: 0.05,
            densify_points: 3,
            ..SearchConfig::default()
        };
        let (visited, knee) = run(config, 10.0);

        let densified: Vec<f64> = visited
            .iter()
            .filter(|(_, phase)| *phase == Phase::Densify)
            .map(|(rate, _)| *rate)
            .collect();
        assert_eq!(densified.len(), 3);
        assert!(
            densified.windows(2).all(|pair| pair[0] < pair[1]),
            "points must be measured in ascending rate order: {densified:?}"
        );
        // Straddling the knee is the point: a bracket alone does not draw a bend.
        let last_good = knee.last_good_rate.unwrap();
        let first_bad = knee.first_bad_rate.unwrap();
        assert!(
            densified.iter().any(|rate| *rate < last_good),
            "{densified:?}"
        );
        assert!(
            densified.iter().any(|rate| *rate > first_bad),
            "{densified:?}"
        );
    }

    #[test]
    fn densified_points_do_not_move_a_bracket_that_was_already_measured() {
        // Densification is drawing, not searching. If a densified point were
        // allowed to re-narrow the bracket, a single noisy run near the knee
        // would silently rewrite a result that several runs had agreed on.
        let mut search = Search::new(SearchConfig {
            start_rate: 1.0,
            tolerance: 0.05,
            densify_points: 2,
            ..SearchConfig::default()
        });
        while let Some((rate, phase)) = search.next_rate() {
            if phase == Phase::Densify {
                break;
            }
            search.record(rate, rate > 10.0);
        }
        let before = search.knee();

        while let Some((rate, _)) = search.next_rate() {
            // Every densified point claims to have crossed, contradicting the
            // bracket. The knee must not move.
            search.record(rate, true);
        }

        let after = search.knee();
        assert_eq!(before.last_good_rate, after.last_good_rate);
        assert_eq!(before.first_bad_rate, after.first_bad_rate);
    }
}
