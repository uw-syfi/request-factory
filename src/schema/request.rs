//! Canonical, modality-compositional requests.
//!
//! Benchmark converters target these types instead of one Rust type per
//! input/output pair. Backends validate the modalities they support, encode
//! each input part, and select an observer for each requested output. This
//! keeps adding a modality linear rather than requiring an executor for every
//! pair of modalities.

use std::collections::BTreeSet;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Tensor,
}

/// Reproducible reference to an immutable local benchmark asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    /// Absolute, or relative to the request artifact.
    pub path: String,
    /// Optional for hand-authored files; benchmark materializers should set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl AssetRef {
    pub fn validate(&self, at: &str) -> Result<()> {
        if self.path.trim().is_empty() {
            bail!("{at}: asset path must not be empty");
        }
        if let Some(digest) = &self.sha256 {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("{at}: sha256 must be 64 lowercase hexadecimal characters");
            }
        }
        if self
            .media_type
            .as_deref()
            .is_some_and(|value| !value.contains('/'))
        {
            bail!("{at}: media_type must be a MIME type such as image/jpeg");
        }
        Ok(())
    }
}

/// One ordered input part. A request may mix and repeat modalities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    Text { text: String },
    Image { asset: AssetRef },
    Audio { asset: AssetRef },
    Video { asset: AssetRef },
    Tensor { asset: AssetRef },
}

impl InputPart {
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio { .. } => Modality::Audio,
            Self::Video { .. } => Modality::Video,
            Self::Tensor { .. } => Modality::Tensor,
        }
    }

    pub fn asset(&self) -> Option<&AssetRef> {
        match self {
            Self::Text { .. } => None,
            Self::Image { asset }
            | Self::Audio { asset }
            | Self::Video { asset }
            | Self::Tensor { asset } => Some(asset),
        }
    }

    fn validate(&self, at: &str) -> Result<()> {
        match self {
            Self::Text { text } if text.is_empty() => bail!("{at}: text must not be empty"),
            Self::Text { .. } => Ok(()),
            _ => self.asset().expect("media input has an asset").validate(at),
        }
    }
}

/// Requested output with controls expressed in that modality's natural units.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputSpec {
    Text {
        max_tokens: usize,
    },
    Image {
        width: u32,
        height: u32,
        steps: usize,
        #[serde(default = "one")]
        count: usize,
    },
    Audio {
        sample_rate_hz: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_samples: Option<u64>,
    },
    Video {
        width: u32,
        height: u32,
        frames: u32,
        steps: usize,
    },
    Tensor {
        shape: Vec<usize>,
        dtype: String,
    },
}

const fn one() -> usize {
    1
}

impl OutputSpec {
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio { .. } => Modality::Audio,
            Self::Video { .. } => Modality::Video,
            Self::Tensor { .. } => Modality::Tensor,
        }
    }

    fn validate(&self, at: &str) -> Result<()> {
        let valid = match self {
            Self::Text { max_tokens } => *max_tokens > 0,
            Self::Image {
                width,
                height,
                steps,
                count,
            } => *width > 0 && *height > 0 && *steps > 0 && *count > 0,
            Self::Audio {
                sample_rate_hz,
                max_samples,
            } => *sample_rate_hz > 0 && max_samples.is_none_or(|samples| samples > 0),
            Self::Video {
                width,
                height,
                frames,
                steps,
            } => *width > 0 && *height > 0 && *frames > 0 && *steps > 0,
            Self::Tensor { shape, dtype } => {
                !shape.is_empty()
                    && shape.iter().all(|dimension| *dimension > 0)
                    && !dtype.trim().is_empty()
            }
        };
        if !valid {
            bail!("{at}: output parameters must all be non-zero and non-empty");
        }
        Ok(())
    }
}

/// Backend-independent semantics of one replay request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestSpec {
    pub id: String,
    pub arrival_time_ms: f64,
    pub inputs: Vec<InputPart>,
    pub outputs: Vec<OutputSpec>,
}

/// A backend/model's compositional modality contract.
///
/// This is deliberately sets-plus-flags, not a list of input/output pairs:
/// accepting a new input modality does not require declaring it once for every
/// output modality. Pair-specific restrictions remain possible through a
/// backend's own validation hook when a model genuinely has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityProfile {
    pub accepted_inputs: BTreeSet<Modality>,
    pub produced_outputs: BTreeSet<Modality>,
    pub supports_mixed_inputs: bool,
    pub supports_multiple_outputs: bool,
}

impl CapabilityProfile {
    pub fn validate(&self, request: &RequestSpec) -> Result<()> {
        request.validate("request")?;
        let inputs = request.input_modalities();
        let outputs = request.output_modalities();
        let unsupported_inputs: Vec<_> =
            inputs.difference(&self.accepted_inputs).copied().collect();
        let unsupported_outputs: Vec<_> = outputs
            .difference(&self.produced_outputs)
            .copied()
            .collect();
        if !unsupported_inputs.is_empty() || !unsupported_outputs.is_empty() {
            bail!(
                "unsupported modalities: inputs={unsupported_inputs:?}, outputs={unsupported_outputs:?}"
            );
        }
        if !self.supports_mixed_inputs && inputs.len() > 1 {
            bail!("backend does not support mixed input modalities: {inputs:?}");
        }
        if !self.supports_multiple_outputs && request.outputs.len() > 1 {
            bail!(
                "backend supports one output per request, got {}",
                request.outputs.len()
            );
        }
        Ok(())
    }
}

impl RequestSpec {
    pub fn validate(&self, at: &str) -> Result<()> {
        if self.id.trim().is_empty() {
            bail!("{at}: request id must not be empty");
        }
        if !self.arrival_time_ms.is_finite() || self.arrival_time_ms < 0.0 {
            bail!("{at}: arrival_time_ms must be finite and non-negative");
        }
        if self.inputs.is_empty() || self.outputs.is_empty() {
            bail!("{at}: inputs and outputs must both be non-empty");
        }
        for (index, input) in self.inputs.iter().enumerate() {
            input.validate(&format!("{at}.inputs[{index}]"))?;
        }
        for (index, output) in self.outputs.iter().enumerate() {
            output.validate(&format!("{at}.outputs[{index}]"))?;
        }
        Ok(())
    }

    pub fn input_modalities(&self) -> BTreeSet<Modality> {
        self.inputs.iter().map(InputPart::modality).collect()
    }

    pub fn output_modalities(&self) -> BTreeSet<Modality> {
        self.outputs.iter().map(OutputSpec::modality).collect()
    }

    pub fn assets(&self) -> impl Iterator<Item = &AssetRef> {
        self.inputs.iter().filter_map(InputPart::asset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(path: &str) -> AssetRef {
        AssetRef {
            path: path.into(),
            sha256: Some("a".repeat(64)),
            media_type: None,
        }
    }

    #[test]
    fn mixed_repeated_inputs_and_multiple_outputs_round_trip() {
        let request = RequestSpec {
            id: "omni-1".into(),
            arrival_time_ms: 12.0,
            inputs: vec![
                InputPart::Text {
                    text: "compare these".into(),
                },
                InputPart::Image {
                    asset: asset("one.jpg"),
                },
                InputPart::Image {
                    asset: asset("two.jpg"),
                },
                InputPart::Audio {
                    asset: asset("question.wav"),
                },
            ],
            outputs: vec![
                OutputSpec::Text { max_tokens: 64 },
                OutputSpec::Audio {
                    sample_rate_hz: 24_000,
                    max_samples: None,
                },
            ],
        };
        request.validate("request").unwrap();

        let encoded = serde_json::to_string(&request).unwrap();
        let decoded: RequestSpec = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, request);
        assert_eq!(decoded.assets().count(), 3);
        assert_eq!(
            decoded.input_modalities(),
            [Modality::Text, Modality::Image, Modality::Audio]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn validation_rejects_bad_assets_and_output_controls() {
        assert!(asset("").validate("asset").is_err());
        assert!(AssetRef {
            path: "x.jpg".into(),
            sha256: Some("ABC".into()),
            media_type: None,
        }
        .validate("asset")
        .is_err());
        assert!(OutputSpec::Video {
            width: 1024,
            height: 1024,
            frames: 0,
            steps: 50,
        }
        .validate("output")
        .is_err());
    }

    #[test]
    fn capabilities_compose_inputs_and_outputs_without_a_pair_matrix() {
        let profile = CapabilityProfile {
            accepted_inputs: [Modality::Text, Modality::Image, Modality::Audio]
                .into_iter()
                .collect(),
            produced_outputs: [Modality::Text, Modality::Audio].into_iter().collect(),
            supports_mixed_inputs: true,
            supports_multiple_outputs: false,
        };
        let mut request = RequestSpec {
            id: "request-1".into(),
            arrival_time_ms: 0.0,
            inputs: vec![
                InputPart::Image {
                    asset: asset("food.jpg"),
                },
                InputPart::Text {
                    text: "describe it".into(),
                },
            ],
            outputs: vec![OutputSpec::Text { max_tokens: 64 }],
        };
        profile.validate(&request).unwrap();

        request.outputs = vec![OutputSpec::Image {
            width: 512,
            height: 512,
            steps: 20,
            count: 1,
        }];
        assert!(profile.validate(&request).is_err());
    }
}
