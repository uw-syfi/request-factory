//! Decode behaviour added by the `speculative` trace tag.

use anyhow::{bail, Result};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Conditional draft-token acceptance used by speculative decoding.
///
/// The scalar form preserves the original trace contract. The position form
/// carries one conditional probability per draft position, so a measured
/// non-geometric acceptance distribution is not collapsed into one number.
#[derive(Clone, Debug, PartialEq)]
pub enum AcceptanceProfile {
    Uniform(f32),
    ByPosition(Vec<f32>),
}

impl AcceptanceProfile {
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        let raw = raw.trim();
        if raw.starts_with('[') {
            serde_json::from_str(raw)
                .map(Self::ByPosition)
                .map_err(|error| format!("invalid accept_rate probability vector: {error}"))
        } else {
            raw.parse::<f32>()
                .map(Self::Uniform)
                .map_err(|error| format!("invalid scalar accept_rate: {error}"))
        }
    }

    pub fn validate(&self, at: &str) -> Result<()> {
        let rates = match self {
            Self::Uniform(rate) => std::slice::from_ref(rate),
            Self::ByPosition(rates) if rates.is_empty() => {
                bail!("{at}: accept_rate probability vector must not be empty")
            }
            Self::ByPosition(rates) => rates,
        };
        if rates
            .iter()
            .any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
        {
            bail!("{at}: accept_rate probabilities must be finite and between 0 and 1");
        }
        Ok(())
    }

    /// Resolve one conditional probability and enforce the worker/trace depth
    /// contract before any random draw is made.
    pub fn at_position(&self, draft_position: u32, draft_tokens: u32) -> f32 {
        match self {
            Self::Uniform(rate) => *rate,
            Self::ByPosition(rates) => {
                assert_eq!(
                    rates.len(),
                    draft_tokens as usize,
                    "accept_rate probability vector length must equal speculative draft_tokens"
                );
                rates[draft_position as usize]
            }
        }
    }
}

impl From<f32> for AcceptanceProfile {
    fn from(rate: f32) -> Self {
        Self::Uniform(rate)
    }
}

impl<'de> Deserialize<'de> for AcceptanceProfile {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ProfileVisitor;

        impl<'de> Visitor<'de> for ProfileVisitor {
            type Value = AcceptanceProfile;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a scalar probability or a JSON probability vector")
            }

            fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(AcceptanceProfile::Uniform(value as f32))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(AcceptanceProfile::Uniform(value as f32))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(AcceptanceProfile::Uniform(value as f32))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                AcceptanceProfile::parse(value).map_err(E::custom)
            }

            fn visit_seq<A>(self, mut values: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut rates = Vec::new();
                while let Some(rate) = values.next_element()? {
                    rates.push(rate);
                }
                Ok(AcceptanceProfile::ByPosition(rates))
            }
        }

        deserializer.deserialize_any(ProfileVisitor)
    }
}

impl Serialize for AcceptanceProfile {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Uniform(rate) => serializer.serialize_f32(*rate),
            Self::ByPosition(rates) => {
                let encoded = serde_json::to_string(rates).map_err(serde::ser::Error::custom)?;
                serializer.serialize_str(&encoded)
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestSpeculative {
    #[serde(default)]
    pub accept_rate: Option<AcceptanceProfile>,
}

impl RequestSpeculative {
    pub fn validate(&self, at: &str) -> Result<()> {
        if let Some(profile) = &self.accept_rate {
            profile.validate(at)?;
        }
        Ok(())
    }

    pub fn strategy(self) -> DecodingStrategy {
        self.accept_rate
            .map_or(DecodingStrategy::Standard, |accept_rate| {
                DecodingStrategy::Speculative { accept_rate }
            })
    }
}

/// A replay client cannot honour this — the server decides how it decodes — but
/// the trace still declares it, and a simulator reading the same file must get
/// the same value.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum DecodingStrategy {
    #[default]
    Standard,
    Speculative {
        accept_rate: AcceptanceProfile,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_position_profiles_parse_from_the_same_csv_column() {
        #[derive(Deserialize)]
        struct Row {
            #[serde(flatten)]
            speculative: RequestSpeculative,
        }

        let mut scalar = csv::Reader::from_reader("accept_rate\n0.75\n".as_bytes());
        let row: Row = scalar.deserialize().next().unwrap().unwrap();
        assert_eq!(
            row.speculative.accept_rate,
            Some(AcceptanceProfile::Uniform(0.75))
        );

        let mut vector =
            csv::Reader::from_reader("accept_rate\n\"[0.9,0.7,0.5,0.4,0.6]\"\n".as_bytes());
        let row: Row = vector.deserialize().next().unwrap().unwrap();
        assert_eq!(
            row.speculative.accept_rate,
            Some(AcceptanceProfile::ByPosition(vec![0.9, 0.7, 0.5, 0.4, 0.6]))
        );
    }

    #[test]
    fn optional_profiles_round_trip_through_csv_and_json() {
        for profile in [
            None,
            Some(AcceptanceProfile::Uniform(0.75)),
            Some(AcceptanceProfile::ByPosition(vec![0.9, 0.7, 0.5, 0.4, 0.6])),
        ] {
            let row = RequestSpeculative {
                accept_rate: profile,
            };
            let json = serde_json::to_string(&row).unwrap();
            assert_eq!(
                serde_json::from_str::<RequestSpeculative>(&json).unwrap(),
                row
            );
            let mut writer = csv::Writer::from_writer(Vec::new());
            writer.serialize(&row).unwrap();
            let bytes = writer.into_inner().unwrap();
            let mut reader = csv::Reader::from_reader(bytes.as_slice());
            assert_eq!(
                reader
                    .deserialize::<RequestSpeculative>()
                    .next()
                    .unwrap()
                    .unwrap(),
                row
            );
        }
    }

    #[test]
    fn position_profile_validates_values_and_worker_depth() {
        let profile = AcceptanceProfile::ByPosition(vec![1.0, 0.5, 0.0]);
        profile.validate("row").unwrap();
        assert_eq!(profile.at_position(1, 3), 0.5);
        assert!(AcceptanceProfile::ByPosition(vec![])
            .validate("row")
            .is_err());
        assert!(AcceptanceProfile::ByPosition(vec![1.1])
            .validate("row")
            .is_err());
    }
}
