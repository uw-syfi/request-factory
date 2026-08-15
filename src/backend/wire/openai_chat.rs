use anyhow::{bail, Result};
use serde::Serialize;

use super::super::{Backend, GenRequest, PreparedInputPart, Prompt, StreamEvent};
use super::parse_openai_event;
use crate::schema::Modality;

/// vLLM/OpenAI-compatible chat transport for mixed text and media inputs with
/// streamed text output.
pub(crate) struct OpenAiChatBackend;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    max_tokens: usize,
    temperature: f64,
    stream: bool,
    stream_options: StreamOptions,
    ignore_eos: bool,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: MessageContent<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MessageContent<'a> {
    Text(&'a str),
    Parts(Vec<ContentPart<'a>>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart<'a> {
    #[serde(rename = "text")]
    Text { text: &'a str },
    #[serde(rename = "image_url")]
    Image { image_url: MediaUrl<'a> },
    #[serde(rename = "audio_url")]
    Audio { audio_url: MediaUrl<'a> },
    #[serde(rename = "video_url")]
    Video { video_url: MediaUrl<'a> },
}

#[derive(Serialize)]
struct MediaUrl<'a> {
    url: &'a str,
}

fn messages(parts: &[PreparedInputPart]) -> Result<Vec<Message<'_>>> {
    let mut messages = Vec::new();
    let mut user_content = Vec::new();
    let mut saw_user = false;
    for part in parts {
        match part {
            PreparedInputPart::System(text) if !saw_user => messages.push(Message {
                role: "system",
                content: MessageContent::Text(text),
            }),
            PreparedInputPart::System(_) => bail!("system inputs must precede user inputs"),
            PreparedInputPart::Text(text) => {
                saw_user = true;
                user_content.push(ContentPart::Text { text });
            }
            PreparedInputPart::Media { modality, data_url } => {
                saw_user = true;
                let url = MediaUrl { url: data_url };
                user_content.push(match modality {
                    Modality::Image => ContentPart::Image { image_url: url },
                    Modality::Audio => ContentPart::Audio { audio_url: url },
                    Modality::Video => ContentPart::Video { video_url: url },
                    Modality::Text | Modality::Tensor => {
                        bail!("unsupported prepared media modality {modality:?}")
                    }
                });
            }
        }
    }
    if user_content.is_empty() {
        bail!("request has no user input")
    }
    messages.push(Message {
        role: "user",
        content: MessageContent::Parts(user_content),
    });
    Ok(messages)
}

impl Backend for OpenAiChatBackend {
    fn endpoint_suffix(&self) -> &str {
        "/chat/completions"
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Parts(parts) = req.prompt else {
            bail!("openai-chat requires prepared multimodal input parts")
        };
        Ok(serde_json::to_vec(&Request {
            model: req.model,
            messages: messages(parts)?,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: req.stream,
            stream_options: StreamOptions {
                include_usage: true,
            },
            ignore_eos: true,
        })?)
    }

    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent> {
        parse_openai_event(data, true, false)
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
        let payload = OpenAiChatBackend
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
        let event = OpenAiChatBackend.parse_event(br#"{"choices":[{"delta":{"content":"pizza"},"finish_reason":null}],"usage":{"prompt_tokens":300,"completion_tokens":1,"total_tokens":301}}"#).unwrap();
        assert_eq!(event.text_delta.as_deref(), Some("pizza"));
        assert_eq!(event.usage.unwrap().completion_tokens, Some(1));
    }
}
