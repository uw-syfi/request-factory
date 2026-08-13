//! Heterogeneous input and output sequences for an omni model.
//!
//! Unlike every other kind, an omni row cannot be described by a fixed set of
//! columns: its input is a sequence that may mix modalities and repeat one. So
//! the two `input_segments`/`output_segments` columns carry JSON, and these
//! types are that JSON's schema.
//!
//! Encoded-token counts stay explicit in every media segment. The trace layer
//! must not guess how a particular model's patching expands an image into
//! tokens; the file says, and both readers get the same number.

use serde::{Deserialize, Serialize};

use super::media::{AudioExtent, ImageExtent, VideoExtent};

/// One item in an omni model's input sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OmniInputSegment {
    Text {
        tokens: u32,
    },
    Image {
        extent: ImageExtent,
        encoded_tokens: u32,
    },
    Audio {
        extent: AudioExtent,
        encoded_tokens: u32,
    },
    Video {
        extent: VideoExtent,
        encoded_tokens: u32,
    },
}

/// One requested output segment from a token-generating omni model.
///
/// Media targets count model or codec tokens. Step-based diffusion pipelines are
/// separate kinds ([`super::TraceKind::TextToVideo`] and
/// [`super::TraceKind::ImageToVideo`]), because steps and tokens are not the
/// same unit of progress.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OmniOutputSpec {
    Text {
        target_tokens: u32,
    },
    Image {
        extent: ImageExtent,
        target_tokens: u32,
    },
    Audio {
        extent: AudioExtent,
        target_tokens: u32,
    },
    Video {
        extent: VideoExtent,
        target_tokens: u32,
    },
}

impl OmniOutputSpec {
    pub fn target_tokens(&self) -> u32 {
        match self {
            Self::Text { target_tokens }
            | Self::Image { target_tokens, .. }
            | Self::Audio { target_tokens, .. }
            | Self::Video { target_tokens, .. } => *target_tokens,
        }
    }
}
