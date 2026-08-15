use anyhow::{bail, Result};
use serde::Serialize;

use super::super::{Backend, GenRequest, Prompt, StreamEvent};
use super::parse_openai_event;

/// OpenAI-compatible `/completions` protocol. Works against vLLM and SGLang's OpenAI endpoint.
pub(crate) struct OpenAiCompletionsBackend;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    rid: &'a str,
    prompt: &'a [u32],
    max_tokens: usize,
    temperature: f64,
    stream: bool,
    ignore_eos: bool,
    return_token_ids: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl Backend for OpenAiCompletionsBackend {
    fn endpoint_suffix(&self) -> &str {
        "/completions"
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Tokens(prompt) = req.prompt else {
            bail!("openai completions requires token-id prompts")
        };
        Ok(serde_json::to_vec(&Request {
            model: req.model,
            rid: req.request_id,
            prompt,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: req.stream,
            ignore_eos: true,
            return_token_ids: true,
            stream_options: req.stream.then_some(StreamOptions {
                include_usage: true,
            }),
        })?)
    }

    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent> {
        parse_openai_event(data, false, true)
    }
}
