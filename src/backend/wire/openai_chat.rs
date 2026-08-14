use anyhow::{bail, Result};
use serde_json::Value;

use super::super::{Backend, GenRequest, PreparedInputPart, Prompt, StreamEvent, Usage};
use super::{usage_cached_prompt_tokens, usage_usize};
use crate::schema::Modality;

/// vLLM/OpenAI-compatible chat transport for mixed text and media inputs with
/// streamed text output.
pub(crate) struct OpenAiChatBackend;

impl Backend for OpenAiChatBackend {
    fn endpoint_suffix(&self) -> &str {
        "/chat/completions"
    }

    fn build_payload(&self, req: &GenRequest) -> Result<Value> {
        let Prompt::Parts(parts) = req.prompt else {
            bail!("openai-chat requires prepared multimodal input parts")
        };
        let content: Vec<Value> = parts
            .iter()
            .map(|part| -> Result<Value> {
                Ok(match part {
                    PreparedInputPart::Text(text) => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    PreparedInputPart::Media { modality, data_url } => match modality {
                        Modality::Image => serde_json::json!({
                            "type": "image_url", "image_url": {"url": data_url}
                        }),
                        Modality::Audio => serde_json::json!({
                            "type": "audio_url", "audio_url": {"url": data_url}
                        }),
                        Modality::Video => serde_json::json!({
                            "type": "video_url", "video_url": {"url": data_url}
                        }),
                        Modality::Text | Modality::Tensor => {
                            bail!("unsupported prepared media modality {modality:?}")
                        }
                    },
                })
            })
            .collect::<Result<_>>()?;
        Ok(serde_json::json!({
            "model": req.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
            "stream_options": {"include_usage": true},
            "ignore_eos": true
        }))
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
        let payload = OpenAiChatBackend
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
    fn parses_chat_delta_and_usage() {
        let event = OpenAiChatBackend.parse_event(&serde_json::json!({
            "choices": [{"delta": {"content": "pizza"}, "finish_reason": null}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 1, "total_tokens": 301}
        }));
        assert_eq!(event.text_delta.as_deref(), Some("pizza"));
        assert_eq!(event.usage.unwrap().completion_tokens, Some(1));
    }
}
