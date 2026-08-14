use anyhow::{bail, Result};
use serde::{de::DeserializeOwned, de::Error as _, Deserialize, Deserializer};

use super::load_utils::{
    self, validate_identity_and_arrival, IndependentRow, ParsedIndependentRow,
};
use crate::schema::{InputFileSchema, OmniInputSegment, OmniOutputSpec};

pub const COLUMNS: &[&str] = &["id", "arrival_time", "input_segments", "output_segments"];

#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub id: String,
    pub arrival_time: f64,
    #[serde(deserialize_with = "deserialize_json")]
    pub input_segments: Vec<OmniInputSegment>,
    #[serde(deserialize_with = "deserialize_json")]
    pub output_segments: Vec<OmniOutputSpec>,
}

impl IndependentRow for Row {
    fn validate(&self, at: &str) -> Result<()> {
        validate_identity_and_arrival(&self.id, self.arrival_time, at)?;
        if self.input_segments.is_empty() || self.output_segments.is_empty() {
            bail!("{at}: omni input_segments and output_segments must both be non-empty");
        }
        if self
            .output_segments
            .iter()
            .any(|segment| segment.target_tokens() == 0)
        {
            bail!("{at}: every omni output segment must target at least one token");
        }
        Ok(())
    }
}

pub fn load(path: &str, schema: &InputFileSchema) -> Result<Vec<ParsedIndependentRow<Row>>> {
    load_utils::load(path, schema)
}

fn deserialize_json<'de, DeserializerType, Value>(
    deserializer: DeserializerType,
) -> std::result::Result<Value, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
    Value: DeserializeOwned,
{
    let value = String::deserialize(deserializer)?;
    serde_json::from_str(&value).map_err(DeserializerType::Error::custom)
}
