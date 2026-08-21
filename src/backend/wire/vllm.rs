use anyhow::{bail, Result};
use serde::Serialize;

use super::super::{Backend, GenRequest, Prompt, StreamEvent};
use super::parse_openai_event;

/// vLLM native token-in/token-out protocol. The server must be launched with
/// `--tokens-only`, which forces `SamplingParams.detokenize = false` for this endpoint.
pub(crate) struct VllmTokensBackend;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    /// SGLang's OpenAI layer forwards this straight to GenerateReqInput, so its
    /// records carry the id this replay logged. vLLM has no such field; its base
    /// model allows unknown keys and ignores this one, taking the id from the
    /// x-request-id header instead.
    rid: &'a str,
    token_ids: &'a [u32],
    sampling_params: SamplingParams,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct SamplingParams {
    max_tokens: usize,
    temperature: f64,
    ignore_eos: bool,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl Backend for VllmTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/inference/v1/generate"
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Tokens(token_ids) = req.prompt else {
            bail!("vllm-tokens requires token-id prompts")
        };
        Ok(serde_json::to_vec(&Request {
            model: req.model,
            rid: req.request_id,
            token_ids,
            sampling_params: SamplingParams {
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                ignore_eos: true,
            },
            stream: req.stream,
            stream_options: req.stream.then_some(StreamOptions {
                include_usage: true,
            }),
        })?)
    }

    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent> {
        parse_openai_event(data, false, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_tokens_backend_uses_native_token_protocol() {
        let backend = VllmTokensBackend;
        let payload = backend
            .serialize_payload(&GenRequest {
                model: "meta-llama/Meta-Llama-3-8B",
                request_id: "req-1",
                prompt: Prompt::Tokens(&[11, 22, 33]),
                max_tokens: 4,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&payload).unwrap();

        assert_eq!(backend.endpoint_suffix(), "/inference/v1/generate");
        assert_eq!(payload["token_ids"], serde_json::json!([11, 22, 33]));
        assert_eq!(payload["sampling_params"]["max_tokens"], 4);
        assert_eq!(payload["sampling_params"]["temperature"], 0.0);
        assert_eq!(payload["sampling_params"]["ignore_eos"], true);
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["stream_options"]["include_usage"], true);
        assert!(payload.get("prompt").is_none());
        assert!(payload.get("return_token_ids").is_none());
    }

    #[test]
    fn vllm_tokens_backend_normalizes_stream_and_usage_events() {
        let backend = VllmTokensBackend;
        let token_event = backend
            .parse_event(
                br#"{
            "request_id": "generate-tokens-request-1", "usage": null,
            "choices": [{"index": 0, "finish_reason": "length", "token_ids": [101, 102]}]
        }"#,
            )
            .unwrap();
        assert_eq!(token_event.token_ids, Some(vec![101, 102]));
        assert_eq!(token_event.finish_reason.as_deref(), Some("length"));
        assert!(token_event.text_delta.is_none());
        assert!(token_event.usage.is_none());

        let usage_event = backend
            .parse_event(
                br#"{
            "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2,
            "total_tokens": 5, "prompt_tokens_details": {"cached_tokens": 1}}
        }"#,
            )
            .unwrap();
        let usage = usage_event.usage.expect("usage event");
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(5));
        assert_eq!(usage.cached_prompt_tokens, Some(1));
    }
}
