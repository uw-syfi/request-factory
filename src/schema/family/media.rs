//! Physical media shapes carried by request-family payloads.
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
