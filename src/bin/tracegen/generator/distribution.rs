//! Scalar distributions a synthetic generator draws lengths from.
//!
//! Three shapes, because they answer three different questions a study asks.
//! `fixed` isolates one variable by removing every other; `uniform` sweeps a
//! range evenly; `lognormal` is what real prompt and completion lengths actually
//! look like — a long right tail that a uniform range cannot produce and that
//! dominates tail latency.
//!
//! Parameterized by **median** rather than by the underlying normal's mean,
//! because `lognormal:2048,0.8` is a sentence somebody can check against a real
//! corpus and `mu = 7.62` is not.

use std::fmt;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::arrivals::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Distribution {
    Fixed(f64),
    /// Inclusive of both ends: `uniform:8..16` can draw 8 and can draw 16.
    Uniform {
        low: f64,
        high: f64,
    },
    /// `median` is the 50th percentile; `sigma` is the standard deviation of the
    /// underlying normal, so it is a shape parameter, not a token count.
    LogNormal {
        median: f64,
        sigma: f64,
    },
}

impl Distribution {
    /// Draw one value, rounded to a whole count and floored at 1.
    ///
    /// Floored because every field this feeds is a token count that must be
    /// positive to be a valid canonical row. A distribution whose tail reaches
    /// zero is a legitimate thing to ask for; emitting a zero-token request is
    /// not.
    pub(crate) fn draw(&self, rng: &mut Rng) -> usize {
        let value = match *self {
            Self::Fixed(value) => value,
            Self::Uniform { low, high } => low + (high - low) * rng.unit(),
            Self::LogNormal { median, sigma } => median * (sigma * rng.standard_normal()).exp(),
        };
        value.round().max(1.0) as usize
    }
}

impl FromStr for Distribution {
    type Err = anyhow::Error;

    fn from_str(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        // A bare number is the common case and should not need a prefix.
        if let Ok(value) = spec.parse::<f64>() {
            return finite_positive(value).map(Self::Fixed);
        }
        let (kind, rest) = spec.split_once(':').with_context(|| {
            format!(
                "{spec:?} is not a distribution (try `512`, \
                 `uniform:256..1024`, or `lognormal:2048,0.8`)"
            )
        })?;
        match kind.trim() {
            "fixed" => finite_positive(number(rest, "fixed")?).map(Self::Fixed),
            "uniform" => {
                let (low, high) = rest.split_once("..").with_context(|| {
                    format!("uniform needs a range: `uniform:256..1024`, got {rest:?}")
                })?;
                let low = finite_positive(number(low, "uniform low")?)?;
                let high = finite_positive(number(high, "uniform high")?)?;
                if high < low {
                    bail!("uniform range {low}..{high} runs backwards");
                }
                Ok(Self::Uniform { low, high })
            }
            "lognormal" => {
                let (median, sigma) = rest
                    .split_once(',')
                    .with_context(|| format!("lognormal needs `median,sigma`, got {rest:?}"))?;
                let median = finite_positive(number(median, "lognormal median")?)?;
                let sigma = number(sigma, "lognormal sigma")?;
                if !sigma.is_finite() || sigma < 0.0 {
                    bail!("lognormal sigma must be finite and non-negative, got {sigma}");
                }
                Ok(Self::LogNormal { median, sigma })
            }
            other => {
                bail!("unknown distribution {other:?} (expected fixed, uniform, or lognormal)")
            }
        }
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(value) => write!(formatter, "fixed:{value}"),
            Self::Uniform { low, high } => write!(formatter, "uniform:{low}..{high}"),
            Self::LogNormal { median, sigma } => {
                write!(formatter, "lognormal:{median},{sigma}")
            }
        }
    }
}

/// Recorded in the manifest as the string it was written as, so a trace's
/// parameters read back the way they were typed.
impl Serialize for Distribution {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

fn number(text: &str, what: &str) -> Result<f64> {
    text.trim()
        .parse()
        .with_context(|| format!("{what} must be a number, got {:?}", text.trim()))
}

fn finite_positive(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        bail!("a length must be finite and greater than zero, got {value}");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_number_is_a_constant_because_that_is_the_common_case() {
        assert_eq!(
            "512".parse::<Distribution>().unwrap(),
            Distribution::Fixed(512.0)
        );
        assert_eq!(
            "fixed:512".parse::<Distribution>().unwrap(),
            Distribution::Fixed(512.0)
        );
    }

    #[test]
    fn every_spec_round_trips_through_the_string_the_manifest_records() {
        for spec in ["fixed:512", "uniform:256..1024", "lognormal:2048,0.8"] {
            let parsed: Distribution = spec.parse().unwrap();
            assert_eq!(parsed.to_string(), spec);
            assert_eq!(spec.parse::<Distribution>().unwrap(), parsed);
        }
    }

    #[test]
    fn a_malformed_spec_names_what_it_expected() {
        for spec in [
            "",
            "gaussian:1,2",
            "uniform:256",
            "uniform:1024..256",
            "lognormal:2048",
            "lognormal:0,1",
            "fixed:-1",
            "fixed:abc",
        ] {
            assert!(
                spec.parse::<Distribution>().is_err(),
                "{spec:?} was accepted"
            );
        }
    }

    #[test]
    fn a_draw_is_never_zero_however_far_the_tail_reaches() {
        // A zero-token request is not a valid canonical row, and a lognormal
        // with a wide sigma will reach there eventually.
        let wide = Distribution::LogNormal {
            median: 2.0,
            sigma: 4.0,
        };
        let mut rng = Rng::new(7);

        assert!((0..10_000).all(|_| wide.draw(&mut rng) >= 1));
    }

    #[test]
    fn uniform_covers_its_whole_declared_range_and_stays_inside_it() {
        let uniform = Distribution::Uniform {
            low: 10.0,
            high: 20.0,
        };
        let mut rng = Rng::new(3);
        let draws: Vec<usize> = (0..5_000).map(|_| uniform.draw(&mut rng)).collect();

        assert!(draws.iter().all(|value| (10..=20).contains(value)));
        assert!(draws.contains(&10) && draws.contains(&20));
    }

    #[test]
    fn a_lognormals_median_is_the_number_it_was_given() {
        // The whole reason it is parameterized this way: the knob must mean what
        // it says without anyone converting to and from the normal's mean.
        let distribution = Distribution::LogNormal {
            median: 2_048.0,
            sigma: 0.8,
        };
        let mut rng = Rng::new(11);
        let mut draws: Vec<usize> = (0..20_000).map(|_| distribution.draw(&mut rng)).collect();
        draws.sort_unstable();

        let median = draws[draws.len() / 2] as f64;
        assert!(
            (median - 2_048.0).abs() / 2_048.0 < 0.05,
            "median drifted to {median}"
        );
        // And the tail is where a uniform range cannot go.
        assert!(*draws.last().unwrap() > 8_000, "{:?}", draws.last());
    }

    #[test]
    fn the_same_seed_draws_the_same_sequence() {
        let distribution: Distribution = "lognormal:100,0.5".parse().unwrap();
        let draw_ten = |seed| {
            let mut rng = Rng::new(seed);
            (0..10)
                .map(|_| distribution.draw(&mut rng))
                .collect::<Vec<_>>()
        };

        assert_eq!(draw_ten(42), draw_ten(42));
        assert_ne!(draw_ten(42), draw_ten(43));
    }
}
