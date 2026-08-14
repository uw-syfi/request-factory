//! Complete input-file formats, grouped by the request family they encode.
//!
//! A format is complete: it determines the request family, exact base columns,
//! loader shape, and structural validation rules. There is no generic `Native`
//! format that must be paired with a second family selector.

pub mod audio_to_text;
pub mod image_to_text;
pub mod image_to_video;
mod load_utils;
pub mod omni_generation;
pub mod text_generation;
pub mod text_to_image;
pub mod text_to_speech;
pub mod text_to_video;
pub mod video_to_text;

use anyhow::{bail, Result};

use super::RequestFamily;

pub use load_utils::ParsedIndependentRow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFileFormat {
    TextGenerationIndependent,
    TextGenerationSessionExecutionV2,
    ImageToTextIndependent,
    VideoToTextIndependent,
    AudioToTextIndependent,
    TextToImageIndependent,
    TextToVideoIndependent,
    TextToSpeechIndependent,
    ImageToVideoIndependent,
    OmniGenerationIndependent,
}

impl InputFileFormat {
    pub const CHOICES: &'static [&'static str] = &[
        "text-generation-independent",
        "text-generation-session-execution-v2",
        "image-to-text-independent",
        "video-to-text-independent",
        "audio-to-text-independent",
        "text-to-image-independent",
        "text-to-video-independent",
        "text-to-speech-independent",
        "image-to-video-independent",
        "omni-generation-independent",
    ];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "text-generation-independent" => Self::TextGenerationIndependent,
            "text-generation-session-execution-v2" => Self::TextGenerationSessionExecutionV2,
            "image-to-text-independent" => Self::ImageToTextIndependent,
            "video-to-text-independent" => Self::VideoToTextIndependent,
            "audio-to-text-independent" => Self::AudioToTextIndependent,
            "text-to-image-independent" => Self::TextToImageIndependent,
            "text-to-video-independent" => Self::TextToVideoIndependent,
            "text-to-speech-independent" => Self::TextToSpeechIndependent,
            "image-to-video-independent" => Self::ImageToVideoIndependent,
            "omni-generation-independent" => Self::OmniGenerationIndependent,
            other => bail!(
                "unknown input file format {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    pub fn name(self) -> &'static str {
        Self::CHOICES[self as usize]
    }

    pub fn request_family(self) -> RequestFamily {
        match self {
            Self::TextGenerationIndependent | Self::TextGenerationSessionExecutionV2 => {
                RequestFamily::TextGeneration
            }
            Self::ImageToTextIndependent => RequestFamily::ImageToText,
            Self::VideoToTextIndependent => RequestFamily::VideoToText,
            Self::AudioToTextIndependent => RequestFamily::AudioToText,
            Self::TextToImageIndependent => RequestFamily::TextToImage,
            Self::TextToVideoIndependent => RequestFamily::TextToVideo,
            Self::TextToSpeechIndependent => RequestFamily::TextToSpeech,
            Self::ImageToVideoIndependent => RequestFamily::ImageToVideo,
            Self::OmniGenerationIndependent => RequestFamily::OmniGeneration,
        }
    }

    pub fn columns(self) -> &'static [&'static str] {
        match self {
            Self::TextGenerationIndependent => text_generation::independent::COLUMNS,
            Self::TextGenerationSessionExecutionV2 => text_generation::session::COLUMNS,
            Self::ImageToTextIndependent => image_to_text::COLUMNS,
            Self::VideoToTextIndependent => video_to_text::COLUMNS,
            Self::AudioToTextIndependent => audio_to_text::COLUMNS,
            Self::TextToImageIndependent => text_to_image::COLUMNS,
            Self::TextToVideoIndependent => text_to_video::COLUMNS,
            Self::TextToSpeechIndependent => text_to_speech::COLUMNS,
            Self::ImageToVideoIndependent => image_to_video::COLUMNS,
            Self::OmniGenerationIndependent => omni_generation::COLUMNS,
        }
    }

    pub fn has_session_topology(self) -> bool {
        self == Self::TextGenerationSessionExecutionV2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{InputFileSchema, TraceTag};

    #[test]
    fn every_declared_format_round_trips_and_has_unique_columns() {
        for name in InputFileFormat::CHOICES {
            let format = InputFileFormat::parse(name).unwrap();
            assert_eq!(format.name(), *name);
            let mut unique = format.columns().to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), format.columns().len(), "{name}");
        }
    }

    #[test]
    fn every_format_determines_exactly_one_family() {
        assert_eq!(
            InputFileFormat::TextGenerationSessionExecutionV2.request_family(),
            RequestFamily::TextGeneration
        );
        assert_eq!(
            InputFileFormat::ImageToVideoIndependent.request_family(),
            RequestFamily::ImageToVideo
        );
    }

    #[test]
    fn a_media_format_loads_its_typed_row_and_declared_tags() {
        let path = std::env::temp_dir().join(format!(
            "req_frontend_image_to_text_{}.csv",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "id,arrival_time,input_len,output_len,encoded_tokens,media_width,media_height,priority\n\
             image-1,0,16,8,256,1024,768,4\n",
        )
        .unwrap();
        let schema = InputFileSchema::new(
            InputFileFormat::ImageToTextIndependent,
            vec![TraceTag::Priority],
        )
        .unwrap();

        let rows = image_to_text::load(path.to_str().unwrap(), &schema).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!(rows[0].row.media_width, 1024);
        assert_eq!(rows[0].row.media_height, 768);
        assert_eq!(rows[0].priority.priority, Some(4));
    }
}
