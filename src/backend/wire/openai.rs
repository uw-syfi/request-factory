use serde_json::Value;

use super::super::{Backend, GenRequest, Prompt, StreamEvent, Usage};
use super::{usage_cached_prompt_tokens, usage_usize};

/// OpenAI-compatible `/completions` protocol. Works against vLLM and SGLang's OpenAI endpoint.
pub(crate) struct OpenAiCompletionsBackend;

impl Backend for OpenAiCompletionsBackend {
    fn endpoint_suffix(&self) -> &str {
        "/completions"
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
            // Submit raw token ids (OpenAI `prompt` accepts an int array): no client-side decode,
            // and the server uses the exact ids so prefix-cache keys match what we constructed.
            "prompt": prompt_ids,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
            // Always run decode to the trace's target length; synthetic prompts otherwise emit
            // EOS almost immediately and collapse the decode workload.
            "ignore_eos": true,
            // Echo the generated token ids (recent vLLM) so we carry the real output forward
            // exactly. Older servers ignore this; we fall back to re-encoding the output text.
            "return_token_ids": true,
        });
        if req.stream {
            // Ask for the trailing usage chunk: server token counts and prefix-cache details
            // are the whole point of this runner.
            payload["stream_options"] = serde_json::json!({"include_usage": true});
        }
        payload
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let text_delta = choice
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let token_ids = choice
            .and_then(|c| c.get("token_ids"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
                    .collect::<Vec<u32>>()
            });
        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
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
            token_ids,
            finish_reason,
            usage,
        }
    }
}
