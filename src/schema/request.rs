//! Canonical, modality-compositional requests.
//!
//! Benchmark converters target these types instead of one Rust type per
//! input/output pair. Backends validate the modalities they support, encode
//! each input part, and select an observer for each requested output. This
//! keeps adding a modality linear rather than requiring an executor for every
//! pair of modalities.

use std::collections::{BTreeMap, BTreeSet};

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    /// A chat-level instruction. It must precede every user input part.
    System {
        text: String,
    },
    Text {
        text: String,
    },
    Image {
        #[serde(flatten)]
        source: MediaSource,
    },
    Audio {
        #[serde(flatten)]
        source: MediaSource,
    },
    Video {
        #[serde(flatten)]
        source: MediaSource,
    },
    Tensor {
        asset: AssetRef,
    },
}

impl InputPart {
    pub fn modality(&self) -> Modality {
        match self {
            Self::System { .. } => Modality::Text,
            Self::Text { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio { .. } => Modality::Audio,
            Self::Video { .. } => Modality::Video,
            Self::Tensor { .. } => Modality::Tensor,
        }
    }

    /// The recorded asset behind this input, if it has one. Synthetic inputs
    /// have no file and no digest, so callers that need bytes go through
    /// [`Self::source`] instead.
    pub fn asset(&self) -> Option<&AssetRef> {
        match self.source() {
            Some(MediaSource::Asset(asset)) => Some(asset),
            _ => match self {
                Self::Tensor { asset } => Some(asset),
                _ => None,
            },
        }
    }

    pub fn source(&self) -> Option<&MediaSource> {
        match self {
            Self::Image { source } | Self::Audio { source } | Self::Video { source } => {
                Some(source)
            }
            _ => None,
        }
    }

    fn validate(&self, at: &str) -> Result<()> {
        match self {
            Self::System { text } | Self::Text { text } if text.trim().is_empty() => {
                bail!("{at}: text must not be empty")
            }
            Self::System { .. } | Self::Text { .. } => Ok(()),
            Self::Tensor { asset } => asset.validate(at),
            _ => match self.source().expect("media input has a source") {
                MediaSource::Asset(asset) => asset.validate(at),
                MediaSource::Synthetic(spec) => spec.validate(self.modality(), at),
            },
        }
    }
}

/// Where one media input's bytes come from.
///
/// A benchmark replays real recorded assets. A capacity run does not need them:
/// it needs bytes of a chosen size, in a format the server will accept, at a
/// chosen rate. Making that a source rather than a second input type keeps every
/// downstream layer -- encoding, dialects, capability checks -- unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSource {
    Asset(AssetRef),
    Synthetic(SyntheticMedia),
}

/// A generated media input, described by shape rather than by content.
///
/// Content is a pure function of `seed` and the shape fields, which is what
/// makes "non-unique" expressible: two requests sharing a seed carry
/// byte-identical media and the generator hands out one shared buffer. Omit the
/// seed and each request gets its own bytes, derived from its request id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMedia {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

impl SyntheticMedia {
    /// Check that this description names a shape the generator can build for
    /// `modality`. The modality comes from the input part, so the fields a
    /// given kind needs are known here rather than guessed at generation time.
    pub fn validate(&self, modality: Modality, at: &str) -> Result<()> {
        let missing = |what: &str| -> Result<()> {
            bail!("{at}: synthetic {modality:?} input requires {what}")
        };
        match modality {
            Modality::Image => {
                if !matches!((self.width, self.height), (Some(w), Some(h)) if w > 0 && h > 0) {
                    return missing("positive width and height");
                }
            }
            Modality::Audio => {
                if !matches!(self.sample_rate_hz, Some(rate) if rate > 0) {
                    return missing("a positive sample_rate_hz");
                }
                if !matches!(self.duration_ms, Some(ms) if ms > 0) {
                    return missing("a positive duration_ms");
                }
            }
            Modality::Video => {
                if !matches!((self.width, self.height), (Some(w), Some(h)) if w > 0 && h > 0) {
                    return missing("positive width and height");
                }
                if !matches!(self.frames, Some(frames) if frames > 0) {
                    return missing("a positive frames");
                }
                if !self.fps.is_none_or(|fps| fps.is_finite() && fps > 0.0) {
                    return missing("a positive fps");
                }
            }
            Modality::Text | Modality::Tensor => {
                bail!("{at}: synthetic content is not defined for {modality:?}")
            }
        }
        Ok(())
    }
}

/// Requested output with controls expressed in that modality's natural units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
        /// Classifier-free-guidance strength. A cross-model concept, spelled
        /// `guidance_scale` by some servers and `cfg_text_scale` by others; the
        /// dialect owns the spelling, the trace owns the value.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        guidance: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        model_params: ModelParams,
    },
    Audio {
        sample_rate_hz: u32,
        /// Backend-native cap on generated audio tokens, when supported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_tokens: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_samples: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        voice: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        model_params: ModelParams,
    },
    Video {
        width: u32,
        height: u32,
        frames: u32,
        steps: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fps: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        guidance: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        model_params: ModelParams,
    },
    Tensor {
        shape: Vec<usize>,
        dtype: String,
    },
}

/// Model-specific generation knobs, namespaced by the dialect that reads them:
/// `{"mstar": {"cfg_renorm_type": "text_channel"}}`.
///
/// These are deliberately not fields on [`OutputSpec`]. A trace carrying
/// `cfg_renorm_type` among its declared outputs is a BAGEL trace rather than an
/// image-generation trace, and cannot be replayed against any other model. The
/// namespace also makes the blast radius explicit: a dialect only ever reads its
/// own entry, so an unrecognized key is inert instead of silently mis-sent.
pub type ModelParams = BTreeMap<String, BTreeMap<String, serde_json::Value>>;

fn model_params_valid(params: &ModelParams) -> bool {
    params.iter().all(|(dialect, knobs)| {
        !dialect.trim().is_empty() && knobs.keys().all(|key| !key.trim().is_empty())
    })
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
                guidance,
                model_params,
                ..
            } => {
                *width > 0
                    && *height > 0
                    && *steps > 0
                    && *count > 0
                    && guidance.is_none_or(|value| value.is_finite() && value > 0.0)
                    && model_params_valid(model_params)
            }
            Self::Audio {
                sample_rate_hz,
                max_tokens,
                max_samples,
                voice,
                model_params,
            } => {
                *sample_rate_hz > 0
                    && max_tokens.is_none_or(|tokens| tokens > 0)
                    && max_samples.is_none_or(|samples| samples > 0)
                    && voice
                        .as_deref()
                        .is_none_or(|value| !value.trim().is_empty())
                    && model_params_valid(model_params)
            }
            Self::Video {
                width,
                height,
                frames,
                steps,
                fps,
                guidance,
                model_params,
                ..
            } => {
                *width > 0
                    && *height > 0
                    && *frames > 0
                    && *steps > 0
                    && fps.is_none_or(|value| value.is_finite() && value > 0.0)
                    && guidance.is_none_or(|value| value.is_finite() && value > 0.0)
                    && model_params_valid(model_params)
            }
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
        if self
            .inputs
            .iter()
            .skip_while(|input| matches!(input, InputPart::System { .. }))
            .any(|input| matches!(input, InputPart::System { .. }))
        {
            bail!("{at}: system inputs must precede all user inputs");
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

    /// Inputs whose bytes are generated rather than read from a file. Counted
    /// separately from [`Self::assets`] so a record showing bytes but no assets
    /// is legible instead of looking like an accounting bug.
    pub fn synthetic_inputs(&self) -> usize {
        self.inputs
            .iter()
            .filter(|input| matches!(input.source(), Some(MediaSource::Synthetic(_))))
            .count()
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
                    source: MediaSource::Asset(asset("one.jpg")),
                },
                InputPart::Image {
                    source: MediaSource::Asset(asset("two.jpg")),
                },
                InputPart::Audio {
                    source: MediaSource::Asset(asset("question.wav")),
                },
            ],
            outputs: vec![
                OutputSpec::Text { max_tokens: 64 },
                OutputSpec::Audio {
                    sample_rate_hz: 24_000,
                    max_tokens: None,
                    max_samples: None,
                    voice: None,
                    model_params: ModelParams::new(),
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
            fps: None,
            guidance: None,
            seed: None,
            model_params: ModelParams::new(),
        }
        .validate("output")
        .is_err());
        assert!(OutputSpec::Audio {
            sample_rate_hz: 24_000,
            max_tokens: Some(0),
            max_samples: None,
            voice: None,
            model_params: ModelParams::new(),
        }
        .validate("output")
        .is_err());
    }

    #[test]
    fn system_inputs_are_text_but_must_precede_user_content() {
        let mut request = RequestSpec {
            id: "speech-1".into(),
            arrival_time_ms: 0.0,
            inputs: vec![
                InputPart::System {
                    text: "Speak clearly.".into(),
                },
                InputPart::Text {
                    text: "Hello".into(),
                },
            ],
            outputs: vec![OutputSpec::Audio {
                sample_rate_hz: 24_000,
                max_tokens: Some(256),
                max_samples: None,
                voice: None,
                model_params: ModelParams::new(),
            }],
        };
        request.validate("request").unwrap();
        assert_eq!(
            request.input_modalities(),
            [Modality::Text].into_iter().collect()
        );
        request.inputs.swap(0, 1);
        assert!(request.validate("request").is_err());
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
                    source: MediaSource::Asset(asset("food.jpg")),
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
            guidance: None,
            seed: None,
            model_params: ModelParams::new(),
        }];
        assert!(profile.validate(&request).is_err());
    }
}
