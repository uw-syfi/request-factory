use anyhow::Result;
use serde::Deserialize;

use super::load_utils::{
    self, validate_identity_and_arrival, validate_positive, validate_positive_f64, IndependentRow,
    ParsedIndependentRow,
};
use crate::schema::InputFileSchema;

pub const COLUMNS: &[&str] = &[
    "id",
    "arrival_time",
    "input_len",
    "denoise_steps",
    "media_duration_s",
    "media_sample_rate_hz",
];

#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub id: String,
    pub arrival_time: f64,
    pub input_len: usize,
    pub denoise_steps: usize,
    pub media_duration_s: f64,
    pub media_sample_rate_hz: u32,
}

impl IndependentRow for Row {
    fn validate(&self, at: &str) -> Result<()> {
        validate_identity_and_arrival(&self.id, self.arrival_time, at)?;
        for (value, column) in [
            (self.input_len, "input_len"),
            (self.denoise_steps, "denoise_steps"),
            (self.media_sample_rate_hz as usize, "media_sample_rate_hz"),
        ] {
            validate_positive(value, column, at)?;
        }
        validate_positive_f64(self.media_duration_s, "media_duration_s", at)
    }
}

pub fn load(path: &str, schema: &InputFileSchema) -> Result<Vec<ParsedIndependentRow<Row>>> {
    load_utils::load(path, schema)
}
