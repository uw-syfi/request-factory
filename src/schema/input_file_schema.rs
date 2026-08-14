//! The exact contract used to open one input file.
//!
//! [`InputFileFormat`] already determines the request family, base columns,
//! loader shape, and structural rules. This module only validates orthogonal
//! tags and combines their columns with that complete format.

use anyhow::{bail, Result};

use super::{InputFileFormat, RequestFamily, TraceTag};

#[derive(Clone, Debug)]
pub struct InputFileSchema {
    pub input_file_format: InputFileFormat,
    pub tags: Vec<TraceTag>,
}

impl InputFileSchema {
    pub fn new(input_file_format: InputFileFormat, tags: Vec<TraceTag>) -> Result<Self> {
        let request_family = input_file_format.request_family();
        let mut unique_tags = Vec::with_capacity(tags.len());
        for tag in tags {
            if input_file_format.has_session_topology() && tag == TraceTag::Session {
                bail!(
                    "input file format {:?} already defines its session topology; the session tag \
                     would declare a second, conflicting set of session columns",
                    input_file_format.name(),
                );
            }
            if unique_tags.contains(&tag) {
                bail!("input file tags list {:?} more than once", tag.name());
            }
            if !tag.applies_to(request_family) {
                bail!(
                    "input file tag {:?} does not apply to request family {request_family:?}",
                    tag.name(),
                );
            }
            unique_tags.push(tag);
        }
        Ok(Self {
            input_file_format,
            tags: unique_tags,
        })
    }

    pub fn text_generation_independent() -> Self {
        Self::new(InputFileFormat::TextGenerationIndependent, Vec::new())
            .expect("the built-in text-generation format is valid")
    }

    pub fn request_family(&self) -> RequestFamily {
        self.input_file_format.request_family()
    }

    pub fn carries(&self, tag: TraceTag) -> bool {
        self.tags.contains(&tag)
    }

    pub fn expected_columns(&self) -> Vec<&'static str> {
        let mut columns = self.input_file_format.columns().to_vec();
        for tag in &self.tags {
            columns.extend_from_slice(tag.columns());
        }
        columns
    }

    /// Verify the exact header before any row is parsed.
    pub fn verify_header<'a>(&self, present: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let present: Vec<&str> = present.into_iter().collect();
        let expected = self.expected_columns();
        let missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|column| !present.contains(column))
            .collect();
        let unexpected: Vec<&str> = present
            .iter()
            .copied()
            .filter(|column| !expected.contains(column))
            .collect();
        if missing.is_empty() && unexpected.is_empty() {
            return Ok(());
        }
        bail!(
            "header does not match input file format {:?} with tags {:?}\n  missing: \
             {missing:?}\n  unexpected: {unexpected:?}\n  expected exactly: {expected:?}",
            self.input_file_format.name(),
            self.tags,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_format_determines_the_family_instead_of_accepting_a_second_selector() {
        let schema = InputFileSchema::new(
            InputFileFormat::TextGenerationSessionExecutionV2,
            Vec::new(),
        )
        .unwrap();

        assert_eq!(schema.request_family(), RequestFamily::TextGeneration);
        assert!(schema.expected_columns().contains(&"round_idx"));
    }

    #[test]
    fn tags_extend_the_complete_format_without_becoming_part_of_it() {
        let schema = InputFileSchema::new(
            InputFileFormat::ImageToTextIndependent,
            vec![TraceTag::Slo, TraceTag::Priority],
        )
        .unwrap();
        let columns = schema.expected_columns();

        assert!(columns.contains(&"media_width"));
        assert!(columns.contains(&"ttft_slo_ms"));
        assert!(columns.contains(&"priority"));
    }

    #[test]
    fn header_validation_rejects_both_missing_and_undeclared_columns() {
        let schema = InputFileSchema::text_generation_independent();
        schema
            .verify_header(["id", "arrival_time", "input_len", "output_len"])
            .unwrap();

        assert!(schema
            .verify_header(["id", "arrival_time", "input_len"])
            .unwrap_err()
            .to_string()
            .contains("output_len"));
        assert!(schema
            .verify_header(["id", "arrival_time", "input_len", "output_len", "extra"])
            .unwrap_err()
            .to_string()
            .contains("extra"));
    }

    #[test]
    fn duplicate_and_inapplicable_tags_are_rejected() {
        assert!(InputFileSchema::new(
            InputFileFormat::TextGenerationIndependent,
            vec![TraceTag::Slo, TraceTag::Slo],
        )
        .is_err());
        assert!(InputFileSchema::new(
            InputFileFormat::TextToImageIndependent,
            vec![TraceTag::Speculative],
        )
        .is_err());
    }
}
