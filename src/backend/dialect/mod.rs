//! Serving dialects: the same request, named differently by every server.
//!
//! Five systems accept multimodal generation over OpenAI-shaped paths and no two
//! agree on how to say it. Audio input is `input_audio` (OpenAI), `audio_url`
//! (vLLM), or a top-level `audios` list (SGLang-Omni). Model knobs sit flat
//! (M*), under `extra_body` (vLLM-Omni), or under `nvext` (Dynamo). Streamed
//! audio arrives at `delta.audio.data` (OpenAI, M*) or behind a top-level
//! `"modality":"audio"` gate (vLLM-Omni).
//!
//! Encoding that matrix as `match (backend, output)` arms costs one Rust edit
//! per model and hides the divergence inside a function body. A [`Dialect`] is
//! the same knowledge as data: one const per target, greppable, diffable, and
//! forced to state every axis. Adding a server is a table entry, not a new code
//! path, and the ways it differs from OpenAI are the fields you had to fill in.
//!
//! What is deliberately *not* here: transport, timing, retries, or measurement.
//! A dialect only renames and re-nests. `media_client` still owns the clock.

mod profiles;

use std::collections::BTreeSet;

use anyhow::{bail, Result};

use crate::schema::Modality;

pub(crate) use profiles::dialect_for;

/// How media attaches to a chat request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaInput {
    /// `{"type":"image_url","image_url":{"url":...}}`, and the `audio_url` /
    /// `video_url` siblings vLLM added. Video has no OpenAI spelling at all, so
    /// any server accepting video accepts it this way or not at all.
    UrlParts,
    /// OpenAI proper: `image_url` for images, `{"type":"input_audio",
    /// "input_audio":{"data","format"}}` for audio, and no video.
    OpenAiParts,
    /// SGLang-Omni keeps `content` plain text and hangs media off the request
    /// root as `images` / `audios` / `videos` arrays of data URLs.
    TopLevelLists,
}

/// Where model-specific knobs go on the wire.
///
/// `Flat` is not a fallback — it is what a pydantic `extra="allow"` server reads
/// (`model_extra`). The OpenAI Python SDK's `extra_body` is a *client-side*
/// convenience that flattens into the body before sending; a literal
/// `"extra_body"` key on the wire is just an unknown field, and the knobs inside
/// it are never seen. Servers that genuinely want a nested envelope say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KnobSlot {
    Flat,
    Nested(&'static str),
}

/// Where the requested output voice goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VoiceSlot {
    /// `audio: {"voice": ..., "format": ...}` — OpenAI, M*, SGLang-Omni.
    AudioObject,
    /// A top-level key. No shipped profile needs one -- every server surveyed
    /// puts voice in `audio` -- but the axis is real and the next one may not,
    /// so it stays a table entry rather than a future patch to `build_payload`.
    #[allow(dead_code)]
    TopLevel(&'static str),
}

/// How streamed generated audio is recognized and decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AudioDeltas {
    /// `choices[0].delta.audio.data`, base64. OpenAI and M*.
    OpenAiDelta,
    /// vLLM-Omni tags the whole chunk `"modality":"audio"` and puts base64 in
    /// `choices[0].delta.content`, where text would otherwise be.
    ModalityGated,
    /// vLLM-Omni's speech endpoint: `{"type":"speech.audio.delta","audio":...}`.
    SpeechSse,
    /// The response body is the audio. No framing to parse.
    RawBytes,
}

/// Ordered attempts at locating generated image bytes in a response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaRead {
    /// `data[*].b64_json` — every element, not just the first: a request that
    /// asked for `n` images and measured one would under-report by `n-1`.
    DataB64Json,
    /// `choices[0].message.content[*].image_url.url` as a data URL.
    ChatContentDataUrl,
    /// M* streaming puts the whole data URL in `choices[0].delta.content`.
    ChatDeltaDataUrl,
}

/// Semantic knob -> wire spelling. `None` means the dialect has no such knob and
/// the value is dropped rather than guessed at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KnobNames {
    pub(crate) steps: Option<&'static str>,
    pub(crate) guidance: Option<&'static str>,
    /// `None` renders dimensions as OpenAI's `"size": "WxH"`.
    pub(crate) width: Option<&'static str>,
    pub(crate) height: Option<&'static str>,
    pub(crate) count: Option<&'static str>,
    pub(crate) seed: Option<&'static str>,
    /// Sent under every name listed. M* needs `max_tokens` *and* its own
    /// `max_output_tokens`; with only the first it ignores the cap and runs to
    /// natural EOS, which silently changes what the benchmark measures.
    pub(crate) max_tokens: &'static [&'static str],
}

/// The `/audio/speech` surface, which drifted further from OpenAI than chat did.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpeechShape {
    pub(crate) max_tokens_key: Option<&'static str>,
    /// How the speech endpoint frames its stream. Distinct from the chat
    /// framing: a server can hand back raw PCM here and SSE deltas there.
    pub(crate) deltas: AudioDeltas,
    /// OpenAI has no boolean `stream` here; streaming is selected by
    /// `stream_format`. M* added a bool. vLLM-Omni accepts both.
    pub(crate) stream_bool: bool,
    pub(crate) stream_format: Option<&'static str>,
    pub(crate) stream_format_value: &'static str,
    pub(crate) response_format: &'static str,
}

/// One serving target's complete wire vocabulary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dialect {
    pub(crate) name: &'static str,
    pub(crate) chat_suffix: &'static str,
    pub(crate) images_suffix: &'static str,
    pub(crate) speech_suffix: &'static str,
    /// `None` means this system does not expose the surface at all, which is a
    /// load-time rejection rather than a 404 discovered mid-run.
    pub(crate) image_edits_suffix: Option<&'static str>,
    pub(crate) videos_suffix: Option<&'static str>,
    pub(crate) transcriptions_suffix: Option<&'static str>,
    pub(crate) translations_suffix: Option<&'static str>,
    pub(crate) media_input: MediaInput,
    pub(crate) knobs: KnobSlot,
    /// Output modality declaration. OpenAI rejects audio-only and requires
    /// `["text","audio"]`; vLLM-Omni accepts `["audio"]` alone.
    pub(crate) audio_modalities: &'static [&'static str],
    pub(crate) image_modalities: Option<&'static [&'static str]>,
    pub(crate) voice: VoiceSlot,
    pub(crate) audio_deltas: AudioDeltas,
    pub(crate) image_reads: &'static [MediaRead],
    pub(crate) video_reads: &'static [MediaRead],
    pub(crate) knob_names: KnobNames,
    pub(crate) speech: SpeechShape,
}

impl Dialect {
    /// Place one model knob according to this dialect's namespacing rule.
    pub(crate) fn put_knob(
        &self,
        payload: &mut serde_json::Value,
        key: &str,
        value: serde_json::Value,
    ) {
        match self.knobs {
            KnobSlot::Flat => {
                if let Some(map) = payload.as_object_mut() {
                    map.insert(key.to_string(), value);
                }
            }
            KnobSlot::Nested(envelope) => {
                if let Some(map) = payload.as_object_mut() {
                    map.entry(envelope)
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .map(|nested| nested.insert(key.to_string(), value));
                }
            }
        }
    }

    /// Input modalities this dialect can actually express on the wire.
    ///
    /// Not every encoding can say everything: OpenAI has no video content part
    /// at all. Intersecting this with the backend's own set is what turns a
    /// video-under-`openai` trace into one startup error instead of N failed
    /// requests discovered mid-run.
    pub(crate) fn accepted_input_modalities(&self) -> BTreeSet<Modality> {
        let mut set: BTreeSet<Modality> = [Modality::Text, Modality::Image, Modality::Audio]
            .into_iter()
            .collect();
        if !matches!(self.media_input, MediaInput::OpenAiParts) {
            set.insert(Modality::Video);
        }
        set
    }

    /// The endpoint suffix for one surface, or an error naming what this system
    /// does not serve.
    pub(crate) fn surface(&self, backend: crate::cli::BackendKind) -> Result<&'static str> {
        use crate::cli::BackendKind as B;
        let (suffix, what) = match backend {
            B::OpenaiChat => (Some(self.chat_suffix), "chat"),
            B::OpenaiImages => (Some(self.images_suffix), "image generation"),
            B::OpenaiSpeech => (Some(self.speech_suffix), "speech"),
            B::OpenaiImageEdits => (self.image_edits_suffix, "image editing"),
            B::OpenaiVideos => (self.videos_suffix, "video generation"),
            B::OpenaiTranscriptions => (self.transcriptions_suffix, "transcription"),
            B::OpenaiTranslations => (self.translations_suffix, "translation"),
            other => bail!("backend {other:?} is not a media surface"),
        };
        suffix.ok_or_else(|| anyhow::anyhow!("dialect {} does not serve {what}", self.name))
    }

    /// Reject an image-over-chat request this dialect cannot express, before
    /// anything is timed. Distinct from the images endpoint: OpenAI serves
    /// `/images/generations` happily but cannot generate an image through chat
    /// at all, so the two surfaces gate on different fields.
    pub(crate) fn check_chat_image_generation(&self) -> Result<()> {
        if self.image_modalities.is_none() {
            bail!(
                "dialect {} cannot generate images over chat: it declares no image output modality",
                self.name
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_knobs_land_at_the_request_root() {
        let dialect = dialect_for("mstar").unwrap();
        let mut payload = serde_json::json!({"prompt": "x"});
        dialect.put_knob(&mut payload, "cfg_renorm_type", "text_channel".into());
        // M* reads pydantic `model_extra`, so a nested envelope would be one
        // unknown field and the knob inside it would never be applied.
        assert_eq!(payload["cfg_renorm_type"], "text_channel");
        assert!(payload.get("extra_body").is_none());
    }

    #[test]
    fn nested_knobs_accumulate_in_one_envelope() {
        let dialect = dialect_for("dynamo").unwrap();
        let mut payload = serde_json::json!({"prompt": "x"});
        dialect.put_knob(&mut payload, "num_inference_steps", 50.into());
        dialect.put_knob(&mut payload, "guidance_scale", 4.0.into());
        assert_eq!(payload["nvext"]["num_inference_steps"], 50);
        assert_eq!(payload["nvext"]["guidance_scale"], 4.0);
    }

    #[test]
    fn every_profile_names_itself_consistently() {
        for name in profiles::ALL {
            assert_eq!(dialect_for(name).unwrap().name, *name);
        }
    }

    #[test]
    fn unknown_dialect_is_rejected_with_the_known_set() {
        let err = dialect_for("gpt-9").unwrap_err().to_string();
        assert!(
            err.contains("mstar"),
            "error should list known dialects: {err}"
        );
    }
}
