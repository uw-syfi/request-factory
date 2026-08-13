//! What a trace file declares itself to be, and which columns that obliges it to
//! carry.
//!
//! A file declares one [`TraceKind`] plus any number of orthogonal [`TraceTag`]s.
//! The declaration is read *before* any row is parsed: headers never infer the
//! request family, and no row carries a request-kind union. Adding a modality is
//! then a new variant plus a column block, not a new optional field on everything
//! that already exists.
//!
//! This module owns the vocabulary and nothing else. It says which kinds exist
//! and what each one's columns are called; it does not say how to open a CSV,
//! what a parsed row becomes, or what any consumer does with it. That is why
//! [`TraceDeclaration::verify_header`] takes header *names* rather than a reader:
//! a generator writing a file and a simulator replaying it check the same rule
//! with their own I/O.
//!
//! The rule this module exists to enforce is that adding a tenth kind requires an
//! edit here and nowhere else.

pub mod media;
pub mod omni;
pub mod session_execution_v2;

use anyhow::{bail, Result};

pub use media::{AudioExtent, DecodingStrategy, ImageExtent, VideoExtent};
pub use omni::{OmniInputSegment, OmniOutputSpec};

/// Columns every native row carries, whatever its kind: who it is and when it
/// arrives.
const RELEASE_COLUMNS: &[&str] = &["id", "arrival_time"];
/// Token-in, token-out.
const AUTOREGRESSIVE_COLUMNS: &[&str] = &["input_len", "output_len"];
/// Step-generated media: progress is denoise steps, not tokens.
const GENERATED_MEDIA_COLUMNS: &[&str] = &["input_len", "denoise_steps"];
/// The count a media encoder expands its input into. Explicit because the trace
/// layer must not guess a model-specific patching.
const ENCODED_INPUT_COLUMNS: &[&str] = &["encoded_tokens"];
const IMAGE_COLUMNS: &[&str] = &["media_width", "media_height"];
const VIDEO_COLUMNS: &[&str] = &[
    "media_width",
    "media_height",
    "media_duration_s",
    "media_fps",
];
const AUDIO_COLUMNS: &[&str] = &["media_duration_s", "media_sample_rate_hz"];
/// Present only where a row has both an input and an output extent, and they are
/// different shapes.
const INPUT_IMAGE_COLUMNS: &[&str] = &["input_media_width", "input_media_height"];
const OMNI_COLUMNS: &[&str] = &["input_segments", "output_segments"];

/// A column that means something in the canonical schema and nothing in the
/// native one. Rejected rather than ignored: a native file carrying round
/// indices is a file whose author expected chaining that will not happen.
const FOREIGN_MULTI_ROUND: &str = "round_idx";

/// One statically selected request family per trace file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
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

impl TraceKind {
    pub const CHOICES: &'static [&'static str] = &[
        "text_generation",
        "image_to_text",
        "video_to_text",
        "audio_to_text",
        "text_to_image",
        "text_to_video",
        "text_to_speech",
        "image_to_video",
        "omni_generation",
    ];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "text_generation" => Self::TextGeneration,
            "image_to_text" => Self::ImageToText,
            "video_to_text" => Self::VideoToText,
            "audio_to_text" => Self::AudioToText,
            "text_to_image" => Self::TextToImage,
            "text_to_video" => Self::TextToVideo,
            "text_to_speech" => Self::TextToSpeech,
            "image_to_video" => Self::ImageToVideo,
            "omni_generation" => Self::OmniGeneration,
            other => bail!(
                "unknown trace_kind {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    /// The name this kind is declared by, which is also how it round-trips
    /// through a manifest.
    pub fn name(self) -> &'static str {
        Self::CHOICES[self as usize]
    }

    /// Whether output progress is measured in generated tokens rather than
    /// generation steps. The two are not interchangeable units, and several
    /// tags are meaningful for only one of them.
    pub fn is_autoregressive(self) -> bool {
        matches!(
            self,
            Self::TextGeneration
                | Self::ImageToText
                | Self::VideoToText
                | Self::AudioToText
                | Self::OmniGeneration
        )
    }

    /// Columns this kind requires beyond [`RELEASE_COLUMNS`].
    ///
    /// Composed from blocks rather than written out per kind, so a modality's
    /// column names are defined once and every kind that uses that modality
    /// agrees with every other by construction.
    fn definition_columns(self) -> Vec<&'static str> {
        let mut columns = match self {
            Self::TextGeneration | Self::ImageToText | Self::VideoToText | Self::AudioToText => {
                AUTOREGRESSIVE_COLUMNS.to_vec()
            }
            Self::TextToImage | Self::TextToVideo | Self::TextToSpeech => {
                GENERATED_MEDIA_COLUMNS.to_vec()
            }
            Self::ImageToVideo => {
                let mut columns = GENERATED_MEDIA_COLUMNS.to_vec();
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(INPUT_IMAGE_COLUMNS);
                columns
            }
            // An omni row's shape is a sequence, not a fixed column set, so its
            // two columns carry JSON and nothing else is added below.
            Self::OmniGeneration => OMNI_COLUMNS.to_vec(),
        };
        match self {
            Self::TextGeneration | Self::OmniGeneration => {}
            Self::ImageToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(IMAGE_COLUMNS);
            }
            Self::VideoToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(VIDEO_COLUMNS);
            }
            Self::AudioToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(AUDIO_COLUMNS);
            }
            Self::TextToImage => columns.extend_from_slice(IMAGE_COLUMNS),
            Self::TextToVideo | Self::ImageToVideo => columns.extend_from_slice(VIDEO_COLUMNS),
            Self::TextToSpeech => columns.extend_from_slice(AUDIO_COLUMNS),
        }
        columns
    }
}

/// A declaration that is orthogonal to the kind, adding columns to any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceTag {
    Session,
    Slo,
    Speculative,
}

impl TraceTag {
    pub const CHOICES: &'static [&'static str] = &["session", "slo", "speculative"];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "session" => Self::Session,
            "slo" => Self::Slo,
            "speculative" => Self::Speculative,
            other => bail!(
                "unknown trace_tag {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    pub fn name(self) -> &'static str {
        Self::CHOICES[self as usize]
    }

    pub fn columns(self) -> &'static [&'static str] {
        match self {
            Self::Session => &["session_id", "prefix_kv", "tool_wait_after_ms"],
            Self::Slo => &["deadline_ms", "priority"],
            Self::Speculative => &["accept_rate"],
        }
    }

    /// Whether this tag can mean anything for the given kind.
    ///
    /// Only speculative decoding is excluded, and only because it *is* a
    /// statement about token-by-token output. A deadline applies to anything,
    /// and a session of image generations is a real workload -- what a
    /// non-autoregressive session may not carry is a reusable KV prefix, which
    /// is a value-level rule its consumer enforces per row.
    pub fn applies_to(self, kind: TraceKind) -> bool {
        match self {
            Self::Session | Self::Slo => true,
            Self::Speculative => kind.is_autoregressive(),
        }
    }
}

/// Which wire format a trace file is written in.
///
/// Orthogonal to [`TraceKind`], which says what a row *is*. A source schema says
/// what the columns are called and, for the canonical form, that every number in
/// them was already resolved upstream. Declared, never sniffed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SourceSchema {
    /// The native column vocabulary defined above.
    #[default]
    Native,
    /// The canonical, already-materialized session execution trace. The same
    /// bytes drive a measured replay and a simulated run, so nothing in it may
    /// be reinterpreted: `prefix_len` is the eligible prefix and `input_len` the
    /// fresh suffix, exactly as the generator resolved them.
    SessionExecutionV2,
}

impl SourceSchema {
    pub const CHOICES: &'static [&'static str] = &["native", session_execution_v2::SCHEMA_NAME];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "native" => Self::Native,
            session_execution_v2::SCHEMA_NAME => Self::SessionExecutionV2,
            other => bail!(
                "unknown trace source schema {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    pub fn name(self) -> &'static str {
        Self::CHOICES[self as usize]
    }
}

/// What one file says it contains, before a single row is read.
#[derive(Clone, Debug)]
pub struct TraceDeclaration {
    pub kind: TraceKind,
    pub tags: Vec<TraceTag>,
    pub source_schema: SourceSchema,
}

impl TraceDeclaration {
    pub fn parse(kind: &str, tags: &[String]) -> Result<Self> {
        let kind = TraceKind::parse(kind)?;
        let mut parsed_tags = Vec::with_capacity(tags.len());
        for tag_name in tags {
            let tag = TraceTag::parse(tag_name)?;
            if parsed_tags.contains(&tag) {
                bail!("trace_tags lists {tag:?} more than once");
            }
            if !tag.applies_to(kind) {
                bail!(
                    "trace_tag {:?} is not meaningful for trace_kind {kind:?}: it describes \
                     token-by-token output, which this kind does not produce",
                    tag.name(),
                );
            }
            parsed_tags.push(tag);
        }
        Ok(Self {
            kind,
            tags: parsed_tags,
            source_schema: SourceSchema::Native,
        })
    }

    /// Parse a declaration that also names its wire format.
    ///
    /// The canonical schema implies its own kind and tag, because the format
    /// exists for exactly one shape of workload. Declaring anything else is a
    /// configuration mistake worth naming rather than silently overriding.
    pub fn parse_with_schema(kind: &str, tags: &[String], schema: &str) -> Result<Self> {
        let source_schema = SourceSchema::parse(schema)?;
        let mut declaration = Self::parse(kind, tags)?;
        if source_schema == SourceSchema::SessionExecutionV2 {
            if declaration.kind != TraceKind::TextGeneration {
                bail!(
                    "{} is a text-generation session format; trace_kind {:?} cannot be read \
                     from it",
                    session_execution_v2::SCHEMA_NAME,
                    declaration.kind
                );
            }
            if !declaration.tags.is_empty() && declaration.tags != [TraceTag::Session] {
                bail!(
                    "{} already declares its session columns; drop trace_tags {:?}",
                    session_execution_v2::SCHEMA_NAME,
                    declaration.tags
                );
            }
            declaration.tags = vec![TraceTag::Session];
        }
        declaration.source_schema = source_schema;
        Ok(declaration)
    }

    /// The plain single-round text workload, which is what a caller wants when
    /// it has no declaration to parse.
    pub fn text() -> Self {
        Self {
            kind: TraceKind::TextGeneration,
            tags: Vec::new(),
            source_schema: SourceSchema::Native,
        }
    }

    pub fn carries(&self, tag: TraceTag) -> bool {
        self.tags.contains(&tag)
    }

    /// Exactly the columns a file with this declaration must have — no more, no
    /// fewer.
    pub fn expected_columns(&self) -> Vec<&'static str> {
        if self.source_schema == SourceSchema::SessionExecutionV2 {
            return session_execution_v2::COLUMNS.to_vec();
        }
        let mut columns = RELEASE_COLUMNS.to_vec();
        columns.extend(self.kind.definition_columns());
        for tag in &self.tags {
            columns.extend_from_slice(tag.columns());
        }
        columns
    }

    /// Check a file's header against this declaration.
    ///
    /// Takes the header names rather than a reader so both a generator and a
    /// replay client apply one rule with their own I/O. Both directions are
    /// errors: a missing column means the file cannot be read, and an unexpected
    /// one means the file describes something the declaration does not, which is
    /// the case where silently ignoring it loses data the author meant to supply.
    ///
    /// The caller adds the path; this returns the mismatch.
    pub fn verify_header<'a>(&self, present: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let present: Vec<&str> = present.into_iter().collect();
        // A canonical trace carries round indices by design; they are the chain
        // it exists to describe, not a foreign column.
        if self.source_schema == SourceSchema::Native && present.contains(&FOREIGN_MULTI_ROUND) {
            bail!(
                "multi-round traces (a {FOREIGN_MULTI_ROUND} column) are not supported in the \
                 native schema; declare the session trace tag and group rows with session_id, or \
                 read a {} file",
                session_execution_v2::SCHEMA_NAME,
            );
        }
        let expected = self.expected_columns();
        let missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|column| !present.iter().any(|found| found == column))
            .collect();
        let unexpected: Vec<&str> = present
            .iter()
            .copied()
            .filter(|column| !expected.iter().any(|known| known == column))
            .collect();
        if missing.is_empty() && unexpected.is_empty() {
            return Ok(());
        }
        bail!(
            "header does not match the declared trace (kind={:?}, tags={:?}, schema={:?})\n  \
             missing: {missing:?}\n  unexpected: {unexpected:?}\n  expected exactly: {expected:?}",
            self.kind,
            self.tags,
            self.source_schema,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_names_itself_and_parses_back() {
        for name in TraceKind::CHOICES {
            let kind = TraceKind::parse(name).expect("declared choice must parse");
            assert_eq!(kind.name(), *name);
        }
        for name in TraceTag::CHOICES {
            let tag = TraceTag::parse(name).expect("declared choice must parse");
            assert_eq!(tag.name(), *name);
        }
        for name in SourceSchema::CHOICES {
            let schema = SourceSchema::parse(name).expect("declared choice must parse");
            assert_eq!(schema.name(), *name);
        }
    }

    #[test]
    fn no_kind_declares_the_same_column_twice() {
        // The column blocks compose, and two blocks naming one column would
        // make `verify_header` demand it once and reject it as unexpected the
        // second time -- a contradiction only a test can catch cheaply.
        for name in TraceKind::CHOICES {
            let columns = TraceKind::parse(name).unwrap().definition_columns();
            let mut seen = columns.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), columns.len(), "{name} repeats a column");
        }
    }

    #[test]
    fn a_media_column_name_means_one_thing_across_every_kind_that_uses_it() {
        // image_to_text and text_to_image both carry an image extent; if the two
        // ever spelled it differently, a generator and a reader could disagree
        // about the same modality.
        let to_text = TraceKind::ImageToText.definition_columns();
        let to_image = TraceKind::TextToImage.definition_columns();
        for column in IMAGE_COLUMNS {
            assert!(to_text.contains(column), "image_to_text lacks {column}");
            assert!(to_image.contains(column), "text_to_image lacks {column}");
        }
    }

    #[test]
    fn a_matching_header_passes_and_both_kinds_of_mismatch_fail() {
        let declaration = TraceDeclaration::text();
        declaration
            .verify_header(["id", "arrival_time", "input_len", "output_len"])
            .expect("the exact expected header must pass");

        let missing = declaration.verify_header(["id", "arrival_time", "input_len"]);
        assert!(missing.unwrap_err().to_string().contains("output_len"));

        let unexpected =
            declaration.verify_header(["id", "arrival_time", "input_len", "output_len", "extra"]);
        assert!(unexpected.unwrap_err().to_string().contains("extra"));
    }

    #[test]
    fn a_tag_adds_its_columns_to_whatever_kind_declared_it() {
        let declaration =
            TraceDeclaration::parse("image_to_text", &["session".to_string()]).unwrap();
        let columns = declaration.expected_columns();

        for column in ["id", "arrival_time", "input_len", "output_len"] {
            assert!(columns.contains(&column), "missing {column}");
        }
        for column in IMAGE_COLUMNS.iter().chain(ENCODED_INPUT_COLUMNS) {
            assert!(columns.contains(column), "missing {column}");
        }
        for column in TraceTag::Session.columns() {
            assert!(columns.contains(column), "missing {column}");
        }
    }

    #[test]
    fn a_tag_that_presupposes_token_output_is_rejected_on_a_step_generated_kind() {
        let err = TraceDeclaration::parse("text_to_image", &["speculative".to_string()])
            .expect_err("speculative decoding has no meaning without token output");
        assert!(err.to_string().contains("speculative"));

        // A deadline is a deadline, and a session of image generations is a real
        // workload -- neither is excluded by the kind.
        TraceDeclaration::parse("text_to_image", &["slo".to_string()])
            .expect("every kind can carry a deadline");
        TraceDeclaration::parse("text_to_image", &["session".to_string()])
            .expect("a session need not be autoregressive");
    }

    #[test]
    fn a_native_file_may_not_smuggle_in_round_indices() {
        let err = TraceDeclaration::text()
            .verify_header(["id", "arrival_time", "input_len", "output_len", "round_idx"])
            .expect_err("round_idx means chaining the native schema does not do");
        assert!(err.to_string().contains("round_idx"));
    }

    #[test]
    fn the_canonical_schema_declares_its_own_columns_and_implies_its_own_tag() {
        let declaration =
            TraceDeclaration::parse_with_schema("text_generation", &[], "session-execution-v2")
                .unwrap();

        assert_eq!(declaration.tags, vec![TraceTag::Session]);
        assert_eq!(
            declaration.expected_columns(),
            session_execution_v2::COLUMNS.to_vec()
        );
        // round_idx is native-only foreign; here it is required.
        declaration
            .verify_header(session_execution_v2::COLUMNS.iter().copied())
            .expect("the canonical header must satisfy its own declaration");
    }

    #[test]
    fn the_canonical_schema_refuses_a_kind_it_cannot_describe() {
        let err = TraceDeclaration::parse_with_schema("image_to_text", &[], "session-execution-v2")
            .expect_err("the canonical format carries no media columns");
        assert!(err.to_string().contains("text-generation session format"));
    }
}
