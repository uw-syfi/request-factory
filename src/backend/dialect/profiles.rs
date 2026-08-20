//! One const per serving target.
//!
//! Every field here was read off that server's request model or serving docs,
//! not inferred from a family resemblance. Where two servers disagree the table
//! disagrees with them; that is the point. Sources are named per profile so the
//! next person can re-check a field without re-deriving the whole matrix.

use anyhow::{anyhow, Result};

use super::{
    AudioDeltas, Dialect, KnobNames, KnobSlot, MediaInput, MediaRead, SpeechShape, VoiceSlot,
};

pub(crate) const ALL: &[&str] = &[
    "openai",
    "vllm",
    "vllm-omni",
    "sglang-omni",
    "mstar",
    "dynamo",
];

pub(crate) fn dialect_for(name: &str) -> Result<&'static Dialect> {
    match name {
        "openai" => Ok(&OPENAI),
        "vllm" => Ok(&VLLM),
        "vllm-omni" => Ok(&VLLM_OMNI),
        "sglang-omni" => Ok(&SGLANG_OMNI),
        "mstar" => Ok(&MSTAR),
        "dynamo" => Ok(&DYNAMO),
        other => Err(anyhow!(
            "unknown dialect {other:?}; known dialects: {}",
            ALL.join(", ")
        )),
    }
}

/// OpenAI proper. Present as the conformance baseline: if a payload renders
/// identically under this profile and another, the divergence is zero.
///
/// Source: developers.openai.com API reference (chat, images, audio/speech).
const OPENAI: Dialect = Dialect {
    name: "openai",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: Some("/images/edits"),
    videos_suffix: Some("/videos"),
    transcriptions_suffix: Some("/audio/transcriptions"),
    translations_suffix: Some("/audio/translations"),
    media_input: MediaInput::OpenAiParts,
    // No extension surface at all: OpenAI rejects unrecognized body params with
    // 400 rather than ignoring them, so `Flat` here means "send nothing extra".
    knobs: KnobSlot::Flat,
    // Audio-only is rejected; text must accompany it.
    audio_modalities: &["text", "audio"],
    // Chat cannot generate images.
    image_modalities: None,
    voice: VoiceSlot::AudioObject,
    audio_deltas: AudioDeltas::OpenAiDelta,
    image_reads: &[MediaRead::DataB64Json],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: None,
        guidance: None,
        width: None,
        height: None,
        count: Some("n"),
        seed: None,
        max_tokens: &["max_completion_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: None,
        // OpenAI has no boolean `stream` on this endpoint.
        stream_bool: false,
        stream_format: Some("stream_format"),

        stream_format_value: "audio",
        deltas: AudioDeltas::RawBytes,
        response_format: "pcm",
    },
};

/// Stock vLLM's OpenAI-compatible server: OpenAI plus the `_url` media parts and
/// unknown-key tolerance.
///
/// Source: docs.vllm.ai features/multimodal_inputs — `image_url`, `audio_url`,
/// `video_url`, `input_audio`, `image_embeds` are all accepted.
const VLLM: Dialect = Dialect {
    name: "vllm",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: None,
    videos_suffix: None,
    transcriptions_suffix: Some("/audio/transcriptions"),
    translations_suffix: Some("/audio/translations"),
    media_input: MediaInput::UrlParts,
    knobs: KnobSlot::Flat,
    audio_modalities: &["text", "audio"],
    image_modalities: None,
    voice: VoiceSlot::AudioObject,
    audio_deltas: AudioDeltas::OpenAiDelta,
    image_reads: &[],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: None,
        guidance: None,
        width: None,
        height: None,
        count: Some("n"),
        seed: Some("seed"),
        max_tokens: &["max_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: None,
        stream_bool: false,
        stream_format: Some("stream_format"),

        stream_format_value: "audio",
        deltas: AudioDeltas::RawBytes,
        response_format: "pcm",
    },
};

/// vLLM-Omni. The one profile with a real nested envelope: its serving docs say
/// to "wrap generation parameters inside an `extra_body` key in the JSON body",
/// so unlike everywhere else the literal key belongs on the wire.
///
/// Source: vllm-project/vllm-omni docs/serving/chat_completions_api.md,
/// docs/serving/speech_api.md, recipes/Bagel/BAGEL-7B-MoT.md.
const VLLM_OMNI: Dialect = Dialect {
    name: "vllm-omni",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: None,
    videos_suffix: None,
    transcriptions_suffix: Some("/audio/transcriptions"),
    translations_suffix: Some("/audio/translations"),
    media_input: MediaInput::UrlParts,
    knobs: KnobSlot::Nested("extra_body"),
    // Accepts audio-only, unlike OpenAI.
    audio_modalities: &["audio"],
    image_modalities: Some(&["image"]),
    voice: VoiceSlot::AudioObject,
    // Tags the whole chunk rather than nesting under `delta.audio`.
    audio_deltas: AudioDeltas::ModalityGated,
    image_reads: &[MediaRead::DataB64Json, MediaRead::ChatContentDataUrl],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: Some("num_inference_steps"),
        guidance: Some("guidance_scale"),
        width: Some("width"),
        height: Some("height"),
        count: Some("num_outputs_per_prompt"),
        seed: Some("seed"),
        max_tokens: &["max_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: Some("max_new_tokens"),
        stream_bool: true,
        stream_format: Some("stream_format"),
        // Its SSE stream carries explicit `speech.audio.delta` events plus a
        // terminal usage block, so first-output timing is observed rather than
        // inferred from when raw bytes happened to arrive.
        stream_format_value: "sse",
        deltas: AudioDeltas::SpeechSse,
        response_format: "pcm",
    },
};

/// SGLang-Omni. Keeps `content` plain and hangs media off the request root.
///
/// Source: sgl-project.github.io/sglang-omni basic_usage/qwen3_omni — media go
/// in `images` / `audios` / `videos`; output audio uses OpenAI's
/// `modalities: ["text","audio"]` with `audio: {"voice","format"}`.
const SGLANG_OMNI: Dialect = Dialect {
    name: "sglang-omni",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: None,
    videos_suffix: None,
    transcriptions_suffix: Some("/audio/transcriptions"),
    translations_suffix: Some("/audio/translations"),
    media_input: MediaInput::TopLevelLists,
    // Staged sampling has its own named field (`stage_sampling`); it is a model
    // knob like any other and reaches the root the same way.
    knobs: KnobSlot::Flat,
    audio_modalities: &["text", "audio"],
    image_modalities: None,
    voice: VoiceSlot::AudioObject,
    audio_deltas: AudioDeltas::OpenAiDelta,
    image_reads: &[MediaRead::DataB64Json],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: None,
        guidance: None,
        width: None,
        height: None,
        count: Some("n"),
        seed: Some("seed"),
        max_tokens: &["max_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: None,
        stream_bool: false,
        stream_format: Some("stream_format"),

        stream_format_value: "audio",
        deltas: AudioDeltas::RawBytes,
        response_format: "wav",
    },
};

/// M*. Pydantic `extra="allow"`, so unknown keys are read from `model_extra` —
/// flat, at the request root. A nested `extra_body` would arrive as one unknown
/// field named `extra_body` and every knob inside it would be dropped.
///
/// Source: mstar-project/mstar mstar/api_server/openai/{protocol,adapters,
/// serving_chat}.py and benchmark/base.py.
const MSTAR: Dialect = Dialect {
    name: "mstar",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: Some("/images/edits"),
    videos_suffix: Some("/videos/generations"),
    transcriptions_suffix: None,
    translations_suffix: None,
    media_input: MediaInput::UrlParts,
    knobs: KnobSlot::Flat,
    audio_modalities: &["text", "audio"],
    image_modalities: Some(&["image"]),
    voice: VoiceSlot::AudioObject,
    // serving_chat.py yields `{"audio": {"id", "data"}}` deltas.
    audio_deltas: AudioDeltas::OpenAiDelta,
    image_reads: &[
        MediaRead::DataB64Json,
        MediaRead::ChatContentDataUrl,
        MediaRead::ChatDeltaDataUrl,
    ],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: Some("num_inference_steps"),
        // BAGEL's text-guidance knob is `cfg_text_scale` here, not the
        // diffusers-style `guidance_scale` vLLM-Omni uses.
        guidance: Some("cfg_text_scale"),
        width: None,
        height: None,
        count: Some("n"),
        seed: Some("seed"),
        // Both names, deliberately: M* reads its own `max_output_tokens`, and
        // without it the cap is ignored and generation runs to natural EOS.
        max_tokens: &["max_tokens", "max_output_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: None,
        // SpeechRequest declares a real boolean `stream`.
        stream_bool: true,
        stream_format: None,

        stream_format_value: "audio",
        deltas: AudioDeltas::RawBytes,
        response_format: "wav",
    },
};

/// NVIDIA Dynamo. Namespaces its extensions under `nvext`.
///
/// Source: docs.nvidia.com/dynamo diffusion/text-to-image and
/// user-guides/multimodal — `nvext.num_inference_steps`, and chat image output
/// inline at `choices[].delta.content[].image_url`.
const DYNAMO: Dialect = Dialect {
    name: "dynamo",
    chat_suffix: "/chat/completions",
    images_suffix: "/images/generations",
    speech_suffix: "/audio/speech",
    image_edits_suffix: None,
    videos_suffix: None,
    transcriptions_suffix: None,
    translations_suffix: None,
    media_input: MediaInput::UrlParts,
    knobs: KnobSlot::Nested("nvext"),
    audio_modalities: &["text", "audio"],
    image_modalities: Some(&["image"]),
    voice: VoiceSlot::AudioObject,
    audio_deltas: AudioDeltas::OpenAiDelta,
    image_reads: &[
        MediaRead::DataB64Json,
        MediaRead::ChatContentDataUrl,
        MediaRead::ChatDeltaDataUrl,
    ],
    video_reads: &[MediaRead::DataB64Json],
    knob_names: KnobNames {
        steps: Some("num_inference_steps"),
        guidance: Some("guidance_scale"),
        width: None,
        height: None,
        count: Some("n"),
        seed: Some("seed"),
        max_tokens: &["max_tokens"],
    },
    speech: SpeechShape {
        max_tokens_key: None,
        stream_bool: false,
        stream_format: Some("stream_format"),

        stream_format_value: "audio",
        deltas: AudioDeltas::RawBytes,
        response_format: "pcm",
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_and_vllm_differ_only_where_vllm_extended_openai() {
        assert_eq!(OPENAI.media_input, MediaInput::OpenAiParts);
        assert_eq!(VLLM.media_input, MediaInput::UrlParts);
        assert_eq!(OPENAI.audio_deltas, VLLM.audio_deltas);
    }

    #[test]
    fn only_declared_envelopes_nest() {
        assert_eq!(MSTAR.knobs, KnobSlot::Flat);
        assert_eq!(VLLM_OMNI.knobs, KnobSlot::Nested("extra_body"));
        assert_eq!(DYNAMO.knobs, KnobSlot::Nested("nvext"));
    }

    #[test]
    fn mstar_sends_both_output_caps() {
        assert_eq!(
            MSTAR.knob_names.max_tokens,
            &["max_tokens", "max_output_tokens"]
        );
    }

    #[test]
    fn guidance_is_not_spelled_the_same_everywhere() {
        assert_eq!(MSTAR.knob_names.guidance, Some("cfg_text_scale"));
        assert_eq!(VLLM_OMNI.knob_names.guidance, Some("guidance_scale"));
    }

    #[test]
    fn openai_alone_forbids_audio_only_output() {
        assert_eq!(OPENAI.audio_modalities, &["text", "audio"]);
        assert_eq!(VLLM_OMNI.audio_modalities, &["audio"]);
    }
}
