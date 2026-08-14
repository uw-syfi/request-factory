//! Shared CSV loading mechanics for family-specific independent formats.
//!
//! Each family module owns its columns, row type, and semantic validation. This
//! file owns only the identical CSV loop and tag decoding so nine loaders do not
//! drift in how they enforce the common input-file contract.

use anyhow::{anyhow, Context, Result};
use serde::de::DeserializeOwned;
use std::ops::Deref;

use crate::schema::{
    InputFileSchema, RequestPriority, RequestSession, RequestSlo, RequestSpeculative, TraceTag,
};

pub trait IndependentRow: DeserializeOwned {
    fn validate(&self, at: &str) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct ParsedIndependentRow<Row> {
    pub row: Row,
    pub session: RequestSession,
    pub slo: RequestSlo,
    pub priority: RequestPriority,
    pub speculative: RequestSpeculative,
}

impl<Row> Deref for ParsedIndependentRow<Row> {
    type Target = Row;

    fn deref(&self) -> &Self::Target {
        &self.row
    }
}

pub fn load<Row: IndependentRow>(
    path: &str,
    input_file_schema: &InputFileSchema,
) -> Result<Vec<ParsedIndependentRow<Row>>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open input file: {path}"))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read the header of {path}"))?
        .clone();
    input_file_schema
        .verify_header(headers.iter())
        .map_err(|mismatch| anyhow!("{path}: {mismatch}"))?;

    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let line = index + 2;
        let at = format!("{path} line {line}");
        let record = record.with_context(|| format!("{at}: failed to read row"))?;
        let row: Row = record
            .deserialize(Some(&headers))
            .with_context(|| format!("{at}: failed to parse base columns"))?;
        row.validate(&at)?;

        let session: RequestSession = deserialize_tag(
            &record,
            &headers,
            input_file_schema.carries(TraceTag::Session),
            &at,
        )?;
        session.validate(&at)?;
        let slo: RequestSlo = deserialize_tag(
            &record,
            &headers,
            input_file_schema.carries(TraceTag::Slo),
            &at,
        )?;
        slo.validate(&at)?;
        let priority: RequestPriority = deserialize_tag(
            &record,
            &headers,
            input_file_schema.carries(TraceTag::Priority),
            &at,
        )?;
        priority.validate(&at)?;
        let speculative: RequestSpeculative = deserialize_tag(
            &record,
            &headers,
            input_file_schema.carries(TraceTag::Speculative),
            &at,
        )?;
        speculative.validate(&at)?;

        rows.push(ParsedIndependentRow {
            row,
            session,
            slo,
            priority,
            speculative,
        });
    }
    Ok(rows)
}

fn deserialize_tag<Tag: Default + DeserializeOwned>(
    record: &csv::StringRecord,
    headers: &csv::StringRecord,
    declared: bool,
    at: &str,
) -> Result<Tag> {
    if !declared {
        return Ok(Tag::default());
    }
    record
        .deserialize(Some(headers))
        .with_context(|| format!("{at}: failed to parse declared tag columns"))
}

pub fn validate_identity_and_arrival(id: &str, arrival_time: f64, at: &str) -> Result<()> {
    if id.is_empty() {
        anyhow::bail!("{at}: id is empty");
    }
    if !arrival_time.is_finite() || arrival_time < 0.0 {
        anyhow::bail!("{at}: arrival_time must be finite and non-negative");
    }
    Ok(())
}

pub fn validate_positive(value: usize, column: &str, at: &str) -> Result<()> {
    if value == 0 {
        anyhow::bail!("{at}: {column} must be greater than zero");
    }
    Ok(())
}

pub fn validate_positive_f64(value: f64, column: &str, at: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("{at}: {column} must be finite and greater than zero");
    }
    Ok(())
}
