use serde_json::Value;

use super::super::{Backend, GenRequest, Prompt, StreamEvent, Usage};
use super::{usage_cached_prompt_tokens, usage_usize};

/// vLLM native token-in/token-out protocol. The server must be launched with
/// `--tokens-only`, which forces `SamplingParams.detokenize = false` for this endpoint.
pub(crate) struct VllmTokensBackend;

impl Backend for VllmTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/inference/v1/generate"
    }

    fn build_payload(&self, req: &GenRequest) -> Value {
        let Prompt::Tokens(prompt_ids) = req.prompt;
        let mut payload = serde_json::json!({
            "model": req.model,
            // SGLang's OpenAI layer forwards this straight to GenerateReqInput,
            // so its records carry the id this replay logged. vLLM has no such
            // field; its base model allows unknown keys and ignores this one,
            // taking the id from the x-request-id header instead.
            "rid": req.request_id,
            "token_ids": prompt_ids,
            "sampling_params": {
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "ignore_eos": true,
            },
            "stream": req.stream,
        });
        if req.stream {
            payload["stream_options"] = serde_json::json!({"include_usage": true});
        }
        payload
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let token_ids = choice
            .and_then(|choice| choice.get("token_ids"))
            .and_then(Value::as_array)
            .map(|token_ids| {
                token_ids
                    .iter()
                    .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect::<Vec<u32>>()
            });
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
            text_delta: None,
            token_ids,
            finish_reason,
            usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_tokens_backend_uses_native_token_protocol() {
        let backend = VllmTokensBackend;
        let payload = backend.build_payload(&GenRequest {
            model: "meta-llama/Meta-Llama-3-8B",
            request_id: "req-1",
            prompt: Prompt::Tokens(&[11, 22, 33]),
            max_tokens: 4,
            temperature: 0.0,
            stream: true,
        });

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
        let token_event = backend.parse_event(&serde_json::json!({
            "request_id": "generate-tokens-request-1",
            "usage": null,
            "choices": [{
                "index": 0,
                "finish_reason": "length",
                "token_ids": [101, 102]
            }]
        }));
        assert_eq!(token_event.token_ids, Some(vec![101, 102]));
        assert_eq!(token_event.finish_reason.as_deref(), Some("length"));
        assert!(token_event.text_delta.is_none());
        assert!(token_event.usage.is_none());

        let usage_event = backend.parse_event(&serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1}
            }
        }));
        let usage = usage_event.usage.expect("usage event");
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(5));
        assert_eq!(usage.cached_prompt_tokens, Some(1));
    }
}
