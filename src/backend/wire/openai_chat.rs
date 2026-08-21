use anyhow::{bail, Result};
use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;

use super::super::{Backend, ChatInputs, Dialect, GenRequest, Prompt, StreamEvent};
use super::parse_openai_event;

/// vLLM/OpenAI-compatible chat transport for mixed text and media inputs with
/// streamed text output.
pub(crate) struct OpenAiChatBackend(pub(crate) &'static Dialect);

/// The request body, written straight to bytes.
///
/// Hand-written `Serialize` rather than a derive because this body's field
/// *names* are dialect data, not source text: the decode cap is sent under
/// every name the dialect lists, and media may leave the message content for
/// keys at the request root. A derive can only spell names known at compile
/// time, and a `Value` tree would cost exactly the allocation the token path
/// exists to avoid.
struct Body<'a> {
    dialect: &'static Dialect,
    req: &'a GenRequest<'a>,
    inputs: &'a ChatInputs<'a>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl Serialize for Body<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        // Flattened rather than nested: the inputs own the whole input half of
        // the body, which under one encoding is more than just `messages`.
        self.inputs.serialize_entries(&mut map)?;
        map.serialize_entry("model", self.req.model)?;
        map.serialize_entry("temperature", &self.req.temperature)?;
        map.serialize_entry("stream", &self.req.stream)?;
        map.serialize_entry(
            "stream_options",
            &StreamOptions {
                include_usage: true,
            },
        )?;
        // A vLLM/SGLang sampling extension, not OpenAI: without it a synthetic
        // prompt emits EOS almost immediately and the declared decode length --
        // the thing the trace exists to reproduce -- collapses.
        if self.dialect.name != "openai" {
            map.serialize_entry("ignore_eos", &true)?;
        }
        // OpenAI deprecated `max_tokens` for `max_completion_tokens`, and M*
        // needs its own second name or it ignores the cap entirely.
        for key in self.dialect.knob_names.max_tokens {
            map.serialize_entry(key, &self.req.max_tokens)?;
        }
        map.end()
    }
}

impl Backend for OpenAiChatBackend {
    fn endpoint_suffix(&self) -> &str {
        self.0.chat_suffix
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Parts(parts) = req.prompt else {
            bail!("openai-chat requires prepared multimodal input parts")
        };
        let inputs = ChatInputs::plan(parts, self.0.media_input)?;
        Ok(serde_json::to_vec(&Body {
            dialect: self.0,
            req,
            inputs: &inputs,
        })?)
    }

    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent> {
        parse_openai_event(data, true, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::PreparedInputPart;
    use crate::schema::Modality;

    #[test]
    fn encodes_repeated_mixed_inputs_and_text_output() {
        let parts = vec![
            PreparedInputPart::Media {
                modality: Modality::Image,
                data_url: "data:image/jpeg;base64,AA==".into(),
            },
            PreparedInputPart::Text("compare".into()),
            PreparedInputPart::Media {
                modality: Modality::Image,
                data_url: "data:image/png;base64,AQ==".into(),
            },
        ];
        let payload = OpenAiChatBackend(crate::backend::dialect_for("vllm").unwrap())
            .serialize_payload(&GenRequest {
                model: "bagel",
                request_id: "r1",
                prompt: Prompt::Parts(&parts),
                max_tokens: 64,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        let content = payload
            .pointer("/messages/0/content")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[1]["text"], "compare");
        assert_eq!(content[2]["type"], "image_url");
    }

    #[test]
    fn preserves_system_role_before_user_content() {
        let parts = vec![
            PreparedInputPart::System("be concise".into()),
            PreparedInputPart::Text("hello".into()),
        ];
        let payload = OpenAiChatBackend(crate::backend::dialect_for("vllm").unwrap())
            .serialize_payload(&GenRequest {
                model: "omni",
                request_id: "r1",
                prompt: Prompt::Parts(&parts),
                max_tokens: 8,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["role"], "user");
    }

    #[test]
    fn parses_chat_delta_and_usage() {
        let event = OpenAiChatBackend(crate::backend::dialect_for("vllm").unwrap())
            .parse_event(
                br#"{"choices":[{"delta":{"content":"pizza"},"finish_reason":null}],"usage":{"prompt_tokens":300,"completion_tokens":1,"total_tokens":301}}"#,
            )
            .unwrap();
        assert_eq!(event.text_delta.as_deref(), Some("pizza"));
        assert_eq!(event.usage.unwrap().completion_tokens, Some(1));
    }

    fn payload(dialect: &'static Dialect, parts: &[PreparedInputPart]) -> serde_json::Value {
        let bytes = OpenAiChatBackend(dialect)
            .serialize_payload(&GenRequest {
                model: "m",
                request_id: "r1",
                prompt: Prompt::Parts(parts),
                max_tokens: 64,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The reason this body is hand-serialized: the decode cap's field *name*
    /// comes from the dialect, and M* needs two of them. A derive could only
    /// have spelled one.
    #[test]
    fn sends_the_decode_cap_under_every_name_the_dialect_lists() {
        let parts = vec![PreparedInputPart::Text("hi".into())];
        let mstar = payload(crate::backend::dialect_for("mstar").unwrap(), &parts);
        assert_eq!(mstar["max_tokens"], 64);
        assert_eq!(mstar["max_output_tokens"], 64);

        let openai = payload(crate::backend::dialect_for("openai").unwrap(), &parts);
        assert_eq!(openai["max_completion_tokens"], 64);
        assert!(openai.get("max_tokens").is_none());
    }

    #[test]
    fn keeps_ignore_eos_off_the_wire_for_openai_proper() {
        let parts = vec![PreparedInputPart::Text("hi".into())];
        assert!(
            payload(crate::backend::dialect_for("openai").unwrap(), &parts)
                .get("ignore_eos")
                .is_none()
        );
        assert_eq!(
            payload(crate::backend::dialect_for("vllm").unwrap(), &parts)["ignore_eos"],
            true
        );
    }

    #[test]
    fn top_level_lists_move_media_off_the_turn_and_leave_text_behind() {
        let parts = vec![
            PreparedInputPart::Text("describe".into()),
            PreparedInputPart::Media {
                modality: Modality::Audio,
                data_url: "data:audio/wav;base64,AA==".into(),
            },
        ];
        let body = payload(crate::backend::dialect_for("sglang-omni").unwrap(), &parts);
        // Content is a plain string here, not an array of parts.
        assert_eq!(body["messages"][0]["content"], "describe");
        assert_eq!(body["audios"][0], "data:audio/wav;base64,AA==");
    }

    #[test]
    fn openai_spells_audio_as_input_audio_with_a_bare_format() {
        let parts = vec![PreparedInputPart::Media {
            modality: Modality::Audio,
            data_url: "data:audio/wav;base64,QUJD".into(),
        }];
        let body = payload(crate::backend::dialect_for("openai").unwrap(), &parts);
        let part = &body["messages"][0]["content"][0];
        assert_eq!(part["type"], "input_audio");
        assert_eq!(part["input_audio"]["format"], "wav");
        assert_eq!(part["input_audio"]["data"], "QUJD");
    }

    #[test]
    fn a_dialect_without_a_name_for_the_modality_fails_before_the_wire() {
        let parts = vec![PreparedInputPart::Media {
            modality: Modality::Video,
            data_url: "data:video/mp4;base64,AA==".into(),
        }];
        let error = OpenAiChatBackend(crate::backend::dialect_for("openai").unwrap())
            .serialize_payload(&GenRequest {
                model: "m",
                request_id: "r1",
                prompt: Prompt::Parts(&parts),
                max_tokens: 8,
                temperature: 0.0,
                stream: true,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("video"), "{error}");
    }

    /// The anti-drift check for rendering the same shaping twice: the bytes the
    /// generation path writes must carry the same input half as the `Value` the
    /// media surfaces build. If these two ever disagree, one of the two
    /// transports is sending something nobody tested.
    #[test]
    fn the_byte_path_and_the_value_path_shape_inputs_identically() {
        let parts = vec![
            PreparedInputPart::System("be concise".into()),
            PreparedInputPart::Text("compare".into()),
            PreparedInputPart::Media {
                modality: Modality::Image,
                data_url: "data:image/png;base64,AQ==".into(),
            },
        ];
        for name in [
            "openai",
            "vllm",
            "vllm-omni",
            "sglang-omni",
            "mstar",
            "dynamo",
        ] {
            let dialect = crate::backend::dialect_for(name).unwrap();
            let from_bytes = payload(dialect, &parts);
            let from_value = serde_json::Value::Object(
                ChatInputs::plan(&parts, dialect.media_input)
                    .unwrap()
                    .to_object()
                    .unwrap(),
            );
            for (key, value) in from_value.as_object().unwrap() {
                assert_eq!(&from_bytes[key], value, "{name} disagrees on {key}");
            }
        }
    }
}
