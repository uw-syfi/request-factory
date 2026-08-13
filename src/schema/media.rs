//! Physical media shapes, and the decode behaviour a row may request.
//!
//! Extents stay concrete per modality rather than collapsing into one
//! width/height/duration bag: a consumer of image rows should not have to handle
//! a frame count that an image never has.
//!
//! These are *trace data* — a generator writes them into columns and a reader
//! parses them back — which is why they live beside the schema rather than in
//! whichever program happens to consume them.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageExtent {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoExtent {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioExtent {
    pub samples: u64,
}

/// Decode behaviour requested by a row carrying the `speculative` tag.
///
/// A replay client cannot honour this — the server decides how it decodes — but
/// the trace still declares it, and a simulator reading the same file must get
/// the same value. Recording it here keeps that one meaning in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DecodingStrategy {
    #[default]
    Standard,
    Speculative {
        accept_rate: f32,
    },
}
