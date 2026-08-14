//! Finding the highest throughput a server will produce, at any rate.
//!
//! A different question from the one [`super::search`] answers, and it needs a
//! different shape. *"What is the highest arrival rate this deployment keeps up
//! with?"* is a boundary: the answer is one rate, and bisection finds it. *"What
//! is the most work this deployment will ever do per second?"* is a maximum: the
//! answer lives **past** that boundary, usually on a plateau rather than at a
//! point, and bisection has nothing to bisect.
//!
//! The two are genuinely different numbers. On a server whose batch size grows
//! with load, peak throughput arrives well after latency has stopped being
//! acceptable — and it can then *decline* as the scheduler starts preempting and
//! recomputing. That decline is one of the more useful things a sweep can find,
//! so this search keeps going past the peak rather than stopping at it.
//!
//! Reported as a region, not a point. Calling one rate "the peak" when a decade
//! of rates are within noise of each other would invent a precision the
//! measurement does not have.

use serde::Serialize;

use crate::search::Phase;

/// How the peak search should behave.
#[derive(Debug, Clone, Copy)]
pub struct PeakConfig {
    pub start_rate: f64,
    pub max_rate: f64,
    /// Relative improvement over the best throughput seen at any lower rate that
    /// still counts as rising.
    pub min_gain: f64,
    /// Consecutive non-improving ramp points before the ramp stops. More than
    /// one, because a single noisy run should not end the search.
    pub patience: usize,
    /// Extra points drawn around the best rate found.
    pub densify_points: usize,
    /// Throughput within this fraction of the peak counts as being on the
    /// plateau.
    pub plateau_tolerance: f64,
}

impl Default for PeakConfig {
    fn default() -> Self {
        Self {
            start_rate: 1.0,
            max_rate: 4_096.0,
            min_gain: 0.03,
            patience: 2,
            densify_points: 3,
            plateau_tolerance: 0.02,
        }
    }
}

/// Why the peak search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakOutcome {
    /// Throughput stopped improving and the ramp ended on its own.
    Located,
    /// `max_rate` was reached while throughput was still climbing. The peak is
    /// above the range searched, and the number below is a floor on it rather
    /// than the peak.
    StillRisingAtMaxRate,
}

/// The peak, stated as the region it actually is.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Peak {
    pub outcome: PeakOutcome,
    /// Highest throughput measured anywhere, and the rate that produced it.
    pub peak_throughput: Option<f64>,
    pub peak_rate: Option<f64>,
    /// The contiguous span of rates whose throughput was within
    /// `plateau_tolerance` of the peak. Where they differ from `peak_rate`, the
    /// span is the honest answer: the peak is flat, not sharp.
    pub plateau_low_rate: Option<f64>,
    pub plateau_high_rate: Option<f64>,
    /// How far throughput at the *highest rate measured* had fallen below the
    /// peak. Positive beyond the plateau tolerance means the server got worse
    /// under more load, not merely no better — usually preemption or cache
    /// thrashing, and worth knowing.
    pub decline_from_peak: Option<f64>,
}

/// Ramp until throughput stops improving, then draw the top of the curve.
pub struct PeakSearch {
    config: PeakConfig,
    phase: Phase,
    pending: Option<f64>,
    /// Every `(rate, throughput)` measured. Read sorted by rate, never in visit
    /// order, so the conclusion does not depend on the sequence that produced it.
    measured: Vec<(f64, f64)>,
    stale_rounds: usize,
    densify_queue: Vec<f64>,
    outcome: Option<PeakOutcome>,
}

impl PeakSearch {
    pub fn new(config: PeakConfig) -> Self {
        Self {
            config,
            phase: Phase::Ramp,
            pending: Some(config.start_rate.min(config.max_rate)),
            measured: Vec::new(),
            stale_rounds: 0,
            densify_queue: Vec::new(),
            outcome: None,
        }
    }

    pub fn next_rate(&mut self) -> Option<(f64, Phase)> {
        self.pending.map(|rate| (rate, self.phase))
    }

    pub fn record(&mut self, rate: f64, throughput: f64) {
        // The best at *strictly lower* rates, before this point joins them.
        let best_below = self
            .measured
            .iter()
            .filter(|(measured_rate, _)| *measured_rate < rate)
            .map(|(_, measured)| *measured)
            .fold(f64::NEG_INFINITY, f64::max);
        self.measured.push((rate, throughput));

        match self.phase {
            Phase::Ramp => {
                let improved = !best_below.is_finite()
                    || throughput > best_below * (1.0 + self.config.min_gain);
                self.stale_rounds = if improved { 0 } else { self.stale_rounds + 1 };
                if self.stale_rounds >= self.config.patience {
                    self.begin_densify();
                    return;
                }
                let next = rate * 2.0;
                if next > self.config.max_rate {
                    // Still climbing when the ceiling arrived: say so rather than
                    // reporting the ceiling's throughput as the peak.
                    self.outcome = Some(if self.stale_rounds == 0 {
                        PeakOutcome::StillRisingAtMaxRate
                    } else {
                        PeakOutcome::Located
                    });
                    self.pending = None;
                    return;
                }
                self.pending = Some(next);
            }
            Phase::Densify => {
                self.pending = self.densify_queue.pop();
                if self.pending.is_none() {
                    self.finish();
                }
            }
            Phase::Bisect => unreachable!("a maximum has nothing to bisect"),
        }
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

    /// Points below the plateau's lower edge, where the curve actually bends.
    ///
    /// Not around the *peak rate*: on a flat top the peak rate is wherever noise
    /// put it, often at the very end of the ramp, and the octave below it is
    /// already plateau. Nothing is learned by drawing more of a flat line.
    ///
    /// The lower edge is the actionable number — the **cheapest** rate that
    /// still gets peak throughput — and the doubling ramp leaves it somewhere in
    /// an octave-wide gap. These points close that gap.
    fn densify_rates(&self) -> Vec<f64> {
        if self.config.densify_points == 0 {
            return Vec::new();
        }
        let Some(plateau_low) = self.peak().plateau_low_rate else {
            return Vec::new();
        };
        let below = self
            .measured
            .iter()
            .map(|(rate, _)| *rate)
            .filter(|rate| *rate < plateau_low)
            .fold(f64::NEG_INFINITY, f64::max);
        if !below.is_finite() {
            // The plateau starts at the first rate measured, so there is no gap
            // below it to draw: the edge is outside the range searched.
            return Vec::new();
        }
        let count = self.config.densify_points;
        (1..=count)
            .map(|index| below + (plateau_low - below) * index as f64 / (count + 1) as f64)
            .collect()
    }

    fn finish(&mut self) {
        self.pending = None;
        if self.outcome.is_none() {
            self.outcome = Some(PeakOutcome::Located);
        }
    }

    fn best(&self) -> Option<(f64, f64)> {
        self.measured
            .iter()
            .copied()
            .max_by(|left, right| left.1.total_cmp(&right.1))
    }

    /// What the search concluded. Meaningful once `next_rate` returns `None`.
    pub fn peak(&self) -> Peak {
        let Some((peak_rate, peak_throughput)) = self.best() else {
            return Peak {
                outcome: self.outcome.unwrap_or(PeakOutcome::Located),
                peak_throughput: None,
                peak_rate: None,
                plateau_low_rate: None,
                plateau_high_rate: None,
                decline_from_peak: None,
            };
        };
        let mut sorted = self.measured.clone();
        sorted.sort_by(|left, right| left.0.total_cmp(&right.0));

        // The plateau is the *contiguous* run containing the peak: a low-rate
        // point that happens to land within tolerance is not part of the same
        // flat top, and including it would report a plateau with a hole in it.
        let floor = peak_throughput * (1.0 - self.config.plateau_tolerance);
        let peak_index = sorted
            .iter()
            .position(|(rate, _)| *rate == peak_rate)
            .expect("the peak was measured");
        let mut low_index = peak_index;
        while low_index > 0 && sorted[low_index - 1].1 >= floor {
            low_index -= 1;
        }
        let mut high_index = peak_index;
        while high_index + 1 < sorted.len() && sorted[high_index + 1].1 >= floor {
            high_index += 1;
        }

        let highest_measured = sorted.last().expect("at least one point").1;
        Peak {
            outcome: self.outcome.unwrap_or(PeakOutcome::Located),
            peak_throughput: Some(peak_throughput),
            peak_rate: Some(peak_rate),
            plateau_low_rate: Some(sorted[low_index].0),
            plateau_high_rate: Some(sorted[high_index].0),
            decline_from_peak: Some((peak_throughput - highest_measured) / peak_throughput),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the search against a server whose curve is known exactly.
    fn run(config: PeakConfig, curve: impl Fn(f64) -> f64) -> (Vec<f64>, Peak) {
        let mut search = PeakSearch::new(config);
        let mut visited = Vec::new();
        while let Some((rate, _)) = search.next_rate() {
            visited.push(rate);
            assert!(visited.len() < 200, "the search did not terminate");
            search.record(rate, curve(rate));
        }
        (visited, search.peak())
    }

    /// Delivered throughput of a server that saturates hard at `capacity`.
    fn capped(capacity: f64) -> impl Fn(f64) -> f64 {
        move |rate| rate.min(capacity)
    }

    #[test]
    fn the_peak_is_found_past_the_rate_where_the_server_stopped_keeping_up() {
        // The whole reason this is a separate mode: a server capped at 50/s
        // stops *keeping up* at 50, and its peak throughput is also 50 — but the
        // peak is achieved at every rate above 50, not at the knee.
        let (_, peak) = run(PeakConfig::default(), capped(50.0));

        assert_eq!(peak.outcome, PeakOutcome::Located);
        assert!((peak.peak_throughput.unwrap() - 50.0).abs() < 1e-9);
        assert!(
            peak.peak_rate.unwrap() >= 50.0,
            "peak throughput is reached at or above capacity, got {:?}",
            peak.peak_rate
        );
        assert!(peak.plateau_high_rate.unwrap() > peak.plateau_low_rate.unwrap());
    }

    #[test]
    fn a_flat_top_is_reported_as_a_span_rather_than_a_single_rate() {
        let (_, peak) = run(PeakConfig::default(), capped(50.0));

        let low = peak.plateau_low_rate.unwrap();
        let high = peak.plateau_high_rate.unwrap();
        // Everything at or above capacity delivers the same throughput, so the
        // plateau must start at the first such rate, not at the last one visited.
        assert!(low <= 64.0, "plateau should start near capacity, got {low}");
        assert!(high >= low);
    }

    #[test]
    fn densification_closes_the_gap_below_the_plateau_not_the_one_below_the_peak() {
        // On a flat top the peak rate is wherever noise put it — often the last
        // rate of the ramp — and the octave below *that* is already plateau.
        // Drawing more of a flat line teaches nothing; the cheapest rate that
        // still reaches peak throughput is the number worth having.
        let config = PeakConfig {
            start_rate: 1.0,
            densify_points: 3,
            ..PeakConfig::default()
        };
        let (visited, peak) = run(config, capped(50.0));

        // The ramp only ever visits 1, 2, 4, ... 64; at capacity 50 the best edge
        // it alone could report is 64. Densification searches the octave below
        // that and must bring the reported edge down toward the true 50.
        let plateau_low = peak.plateau_low_rate.unwrap();
        assert!(
            plateau_low < 64.0,
            "densification did not tighten the edge the doubling ramp reported: {plateau_low}"
        );
        assert!(plateau_low >= 50.0, "the edge cannot be below capacity");

        let densified: Vec<f64> = visited
            .iter()
            .copied()
            .filter(|rate| rate.log2().fract() != 0.0)
            .collect();
        assert!(!densified.is_empty(), "nothing was densified: {visited:?}");
        assert!(
            densified.iter().all(|rate| (32.0..=64.0).contains(rate)),
            "densification must search the gap the ramp left, got {densified:?}"
        );
    }

    #[test]
    fn throughput_that_falls_away_past_the_peak_is_reported_as_a_decline() {
        // A server that thrashes under overload: this is the finding a sweep
        // that stopped at the knee would never make.
        let (_, peak) = run(PeakConfig::default(), |rate| {
            if rate <= 50.0 {
                rate
            } else {
                (50.0 - (rate - 50.0) * 0.5).max(1.0)
            }
        });

        assert!(
            peak.decline_from_peak.unwrap() > 0.1,
            "a thrashing server must report its decline, got {:?}",
            peak.decline_from_peak
        );
    }

    #[test]
    fn a_curve_still_climbing_at_the_ceiling_says_so_instead_of_calling_it_the_peak() {
        let config = PeakConfig {
            start_rate: 1.0,
            max_rate: 64.0,
            ..PeakConfig::default()
        };
        // Linear forever: the ceiling is the only thing that stops it.
        let (_, peak) = run(config, |rate| rate);

        assert_eq!(peak.outcome, PeakOutcome::StillRisingAtMaxRate);
        assert_eq!(peak.peak_rate, Some(64.0));
    }

    #[test]
    fn one_noisy_point_does_not_end_the_ramp() {
        // patience exists for this: a single run that came back flat, on a curve
        // that is still climbing, must not be read as the top.
        let config = PeakConfig {
            start_rate: 1.0,
            patience: 2,
            ..PeakConfig::default()
        };
        let (visited, peak) = run(
            config,
            |rate| if rate == 8.0 { 4.0 } else { rate.min(200.0) },
        );

        assert!(
            visited.contains(&32.0),
            "the ramp stopped at the noisy point: {visited:?}"
        );
        assert!(peak.peak_throughput.unwrap() > 100.0);
    }

    #[test]
    fn the_conclusion_does_not_depend_on_the_order_points_were_measured_in() {
        // Peak, plateau and decline are all read off the rate-sorted curve. The
        // ramp happens to visit ascending, but densification does not, and a
        // resumed sweep may hand back points in any order at all.
        let mut ascending = PeakSearch::new(PeakConfig::default());
        let mut descending = PeakSearch::new(PeakConfig::default());
        let points = [(10.0, 10.0), (20.0, 20.0), (40.0, 30.0), (80.0, 29.0)];
        for (rate, throughput) in points {
            ascending.measured.push((rate, throughput));
        }
        for (rate, throughput) in points.iter().rev() {
            descending.measured.push((*rate, *throughput));
        }

        let left = ascending.peak();
        let right = descending.peak();
        assert_eq!(left.peak_rate, right.peak_rate);
        assert_eq!(left.plateau_low_rate, right.plateau_low_rate);
        assert_eq!(left.plateau_high_rate, right.plateau_high_rate);
        assert_eq!(left.decline_from_peak, right.decline_from_peak);
    }

    #[test]
    fn a_plateau_does_not_swallow_a_low_rate_point_that_happens_to_match() {
        // A U-shaped curve: 5/s and 40/s both deliver 30, with a dip between.
        // Reporting a plateau from 5 to 40 would describe a flat top that has a
        // hole in the middle of it.
        let mut search = PeakSearch::new(PeakConfig::default());
        for (rate, throughput) in [(5.0, 30.0), (10.0, 1.0), (20.0, 2.0), (40.0, 30.5)] {
            search.measured.push((rate, throughput));
        }

        let peak = search.peak();

        assert_eq!(peak.peak_rate, Some(40.0));
        assert_eq!(peak.plateau_low_rate, Some(40.0));
    }
}
