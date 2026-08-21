//! Input-file schema composition.
//!
//! [`InputFileFormat`] is a complete format that determines its request family.
//! [`InputFileSchema`] adds orthogonal tags and resolves the exact header before
//! that format's loader reads any row.

pub mod family;
pub mod format;
mod input_file_schema;
mod request;
pub mod tag;

pub use family::{
    AudioExtent, ImageExtent, OmniInputSegment, OmniOutputSpec, RequestFamily, VideoExtent,
};
pub use format::InputFileFormat;
pub use input_file_schema::InputFileSchema;
pub use request::{
    AssetRef, CapabilityProfile, InputPart, MediaSource, Modality, ModelParams, OutputSpec,
    RequestSpec, SyntheticMedia,
};
pub use tag::{
    DecodingStrategy, RequestPriority, RequestSession, RequestSlo, RequestSpeculative, TraceTag,
};
