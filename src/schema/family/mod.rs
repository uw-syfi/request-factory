mod media;
mod omni;

/// The request semantics selected by one complete input-file format.
///
/// This is a property of [`crate::schema::InputFileFormat`], not an independent
/// parsing axis and never a value inferred from CSV headers or individual rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestFamily {
    TextGeneration,
    ImageToText,
    VideoToText,
    AudioToText,
    TextToImage,
    TextToVideo,
    TextToSpeech,
    ImageToVideo,
    OmniGeneration,
}

pub use media::{AudioExtent, ImageExtent, VideoExtent};
pub use omni::{OmniInputSegment, OmniOutputSpec};
