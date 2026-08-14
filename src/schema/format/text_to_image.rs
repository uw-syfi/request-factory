use anyhow::Result;
use serde::Deserialize;

use super::load_utils::{
    self, validate_identity_and_arrival, validate_positive, IndependentRow, ParsedIndependentRow,
};
use crate::schema::InputFileSchema;

pub const COLUMNS: &[&str] = &[
    "id",
    "arrival_time",
    "input_len",
    "denoise_steps",
    "media_width",
    "media_height",
];

#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub id: String,
    pub arrival_time: f64,
    pub input_len: usize,
    pub denoise_steps: usize,
    pub media_width: u32,
    pub media_height: u32,
}

impl IndependentRow for Row {
    fn validate(&self, at: &str) -> Result<()> {
        validate_identity_and_arrival(&self.id, self.arrival_time, at)?;
        for (value, column) in [
            (self.input_len, "input_len"),
            (self.denoise_steps, "denoise_steps"),
            (self.media_width as usize, "media_width"),
            (self.media_height as usize, "media_height"),
        ] {
            validate_positive(value, column, at)?;
        }
        Ok(())
    }
}

pub fn load(path: &str, schema: &InputFileSchema) -> Result<Vec<ParsedIndependentRow<Row>>> {
    load_utils::load(path, schema)
}
