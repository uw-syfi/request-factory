use anyhow::{bail, Result};
use serde_json::Value;

use super::super::{chat_inputs, Backend, Dialect, GenRequest, Prompt, StreamEvent, Usage};
use super::{usage_cached_prompt_tokens, usage_usize};

/// vLLM/OpenAI-compatible chat transport for mixed text and media inputs with
/// streamed text output.
pub(crate) struct OpenAiChatBackend(pub(crate) &'static Dialect);

impl Backend for OpenAiChatBackend {
    fn endpoint_suffix(&self) -> &str {
        self.0.chat_suffix
    }

    fn build_payload(&self, req: &GenRequest) -> Result<Value> {
        let Prompt::Parts(parts) = req.prompt else {
            bail!("openai-chat requires prepared multimodal input parts")
        };
        let mut payload = Value::Object(chat_inputs(parts, self.0.media_input)?);
        payload["model"] = req.model.into();
        payload["temperature"] = req.temperature.into();
        payload["stream"] = req.stream.into();
        payload["stream_options"] = serde_json::json!({"include_usage": true});
        // A vLLM/SGLang sampling extension, not OpenAI: without it a synthetic
        // prompt emits EOS almost immediately and the declared decode length --
        // the thing the trace exists to reproduce -- collapses.
        if self.0.name != "openai" {
            payload["ignore_eos"] = true.into();
        }
        // OpenAI deprecated `max_tokens` for `max_completion_tokens`, and M*
        // needs its own second name or it ignores the cap entirely.
        for key in self.0.knob_names.max_tokens {
            payload[*key] = req.max_tokens.into();
        }
        Ok(payload)
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let text_delta = choice
            .and_then(|choice| choice.pointer("/delta/content"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let finish_reason = choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let usage = value
            .get("usage")
            .filter(|usage| !usage.is_null())
            .map(|usage| Usage {
                prompt_tokens: usage_usize(usage, "prompt_tokens"),
                completion_tokens: usage_usize(usage, "completion_tokens"),
                total_tokens: usage_usize(usage, "total_tokens"),
                cached_prompt_tokens: usage_cached_prompt_tokens(usage),
            });
        StreamEvent {
            text_delta,
            token_ids: None,
            finish_reason,
            usage,
        }
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
            .build_payload(&GenRequest {
                model: "bagel",
                request_id: "r1",
                prompt: Prompt::Parts(&parts),
                max_tokens: 64,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
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
            .build_payload(&GenRequest {
                model: "omni",
                request_id: "r1",
                prompt: Prompt::Parts(&parts),
                max_tokens: 8,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["role"], "user");
    }

    #[test]
    fn parses_chat_delta_and_usage() {
        let event = OpenAiChatBackend(crate::backend::dialect_for("vllm").unwrap()).parse_event(
            &serde_json::json!({
                "choices": [{"delta": {"content": "pizza"}, "finish_reason": null}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 1, "total_tokens": 301}
            }),
        );
        assert_eq!(event.text_delta.as_deref(), Some("pizza"));
        assert_eq!(event.usage.unwrap().completion_tokens, Some(1));
    }
}
