//! Synthetic session arrivals, and the reproducible randomness behind them.
//!
//! The corpus has no arrival timestamps — it records what each session did, not
//! when it showed up. So a replayable trace has to invent a timeline, and that
//! invention is a workload-shaping decision: it belongs here, beside the context
//! policy, recorded in the same manifest.
//!
//! The generator is written out rather than pulled from a crate. Reproducing a
//! published trace years later means reproducing this exact bit stream, and a
//! dependency that reserves the right to change its default algorithm across a
//! major version cannot promise that. Thirty lines we own can.

/// xoshiro256\*\* seeded through SplitMix64, as published. Chosen because the
/// algorithm is fixed in the literature, so "seed 0" means one specific stream
/// forever.
pub(crate) struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // SplitMix64: expands one seed into four well-mixed words, so that
        // seed 0 and seed 1 do not produce correlated streams.
        let mut z = seed;
        let mut next_seed_word = move || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            state: [
                next_seed_word(),
                next_seed_word(),
                next_seed_word(),
                next_seed_word(),
            ],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform on the *open* interval (0, 1). Open at zero because the
    /// exponential inverse CDF below takes a logarithm of it.
    fn next_open_unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }

    /// Exponential with the given mean, by inverse CDF.
    fn exponential(&mut self, mean: f64) -> f64 {
        -mean * self.next_open_unit().ln()
    }

    /// Uniform on `[0, bound)`, rejecting the short tail so the result is
    /// unbiased rather than modulo-skewed toward small values.
    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let limit = u64::MAX - u64::MAX % bound;
        loop {
            let value = self.next_u64();
            if value < limit {
                return value % bound;
            }
        }
    }

    /// Fisher-Yates, in place.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let swap_with = self.below(index as u64 + 1) as usize;
            items.swap(index, swap_with);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ArrivalPattern {
    /// Poisson process: exponential gaps at the requested rate.
    Poisson,
    /// Evenly spaced at the requested rate.
    Constant,
}

impl ArrivalPattern {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Poisson => "poisson",
            Self::Constant => "constant",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum SessionOrder {
    /// Keep the source file's session order.
    Source,
    /// Permute sessions before assigning arrivals.
    Shuffle,
}

impl SessionOrder {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Shuffle => "shuffle",
        }
    }
}

/// Arrival offsets in milliseconds for `count` sessions, first one at zero.
///
/// The first session always arrives at 0 so a trace starts at its own origin;
/// `rate_per_second` then governs the gaps, not the absolute offsets.
pub(crate) fn synthesize(
    rng: &mut Rng,
    count: usize,
    rate_per_second: f64,
    pattern: ArrivalPattern,
) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let interval_ms = 1000.0 / rate_per_second;
    let mut arrivals = Vec::with_capacity(count);
    match pattern {
        ArrivalPattern::Constant => {
            for index in 0..count {
                arrivals.push(index as f64 * interval_ms);
            }
        }
        ArrivalPattern::Poisson => {
            let mut clock = 0.0;
            arrivals.push(clock);
            for _ in 1..count {
                clock += rng.exponential(interval_ms);
                arrivals.push(clock);
            }
        }
    }
    arrivals
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The xoshiro256\*\* stream for a SplitMix64-expanded seed, pinned as a
    /// literal and cross-checked against an independent implementation of both
    /// published algorithms. Two things depend on this: that the generator is
    /// really xoshiro256\*\* and not a typo that happens to look random, and
    /// that a refactor cannot silently invalidate every trace anyone has
    /// already generated with this tool.
    #[test]
    fn the_bit_stream_is_pinned() {
        let mut rng = Rng::new(0);
        let drawn: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(
            drawn,
            vec![
                11091344671253066420,
                13793997310169335082,
                1900383378846508768,
                7684712102626143532,
            ]
        );
        assert_eq!(Rng::new(1).next_u64(), 12966619160104079557);
    }

    #[test]
    fn the_open_unit_never_reaches_a_boundary() {
        let mut rng = Rng::new(7);
        for _ in 0..100_000 {
            let value = rng.next_open_unit();
            assert!(value > 0.0 && value < 1.0, "{value} left (0, 1)");
        }
    }

    #[test]
    fn constant_arrivals_are_evenly_spaced_at_the_requested_rate() {
        let mut rng = Rng::new(0);
        let arrivals = synthesize(&mut rng, 4, 2.0, ArrivalPattern::Constant);
        assert_eq!(arrivals, vec![0.0, 500.0, 1000.0, 1500.0]);
    }

    #[test]
    fn poisson_arrivals_start_at_zero_and_increase() {
        let mut rng = Rng::new(0);
        let arrivals = synthesize(&mut rng, 500, 1.0, ArrivalPattern::Poisson);
        assert_eq!(arrivals[0], 0.0);
        assert!(arrivals.windows(2).all(|pair| pair[1] > pair[0]));
    }

    /// The rate is the contract, so check the realized mean gap rather than
    /// only that the numbers move in the right direction.
    #[test]
    fn poisson_gaps_average_to_the_requested_rate() {
        let mut rng = Rng::new(42);
        let count = 200_000;
        let arrivals = synthesize(&mut rng, count, 4.0, ArrivalPattern::Poisson);
        let mean_gap = arrivals[count - 1] / (count - 1) as f64;
        // 1000 ms / 4 per second = 250 ms expected.
        assert!(
            (mean_gap - 250.0).abs() < 5.0,
            "mean gap {mean_gap} is not near 250 ms"
        );
    }

    #[test]
    fn a_single_session_arrives_at_the_origin() {
        let mut rng = Rng::new(0);
        assert_eq!(
            synthesize(&mut rng, 1, 1.0, ArrivalPattern::Poisson),
            vec![0.0]
        );
        assert!(synthesize(&mut rng, 0, 1.0, ArrivalPattern::Poisson).is_empty());
    }

    #[test]
    fn shuffle_is_a_permutation_and_is_seed_reproducible() {
        let mut first: Vec<usize> = (0..64).collect();
        let mut second = first.clone();
        Rng::new(3).shuffle(&mut first);
        Rng::new(3).shuffle(&mut second);
        assert_eq!(
            first, second,
            "the same seed must give the same permutation"
        );

        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(
            first, sorted,
            "a 64-element shuffle that changes nothing is a bug"
        );
    }
}
