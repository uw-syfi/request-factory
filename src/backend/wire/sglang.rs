use serde_json::Value;

use super::super::{Backend, GenRequest, Prompt, StreamEvent, Usage};
// Only `usage_usize`: SGLang's `meta_info` schema is fixed, so it reads exact
// keys rather than searching the provider alias list.
use super::usage_usize;

/// SGLang native token-in/token-out `/generate` protocol.
///
/// The server must be launched with two flags. `--skip-tokenizer-init` is the
/// counterpart of vLLM's `--tokens-only`: it accepts `input_ids` and returns
/// `output_ids` without ever detokenizing. `--stream-output` (renamed
/// `--incremental-streaming-output` in newer builds) makes streamed chunks
/// disjoint deltas; SGLang's default resends the whole output every chunk,
/// which is O(n^2) on the wire and inflates late-token latency.
/// [`restates_accumulated_output`] fails the round if that default is still in
/// effect rather than silently reporting the polluted timings.
///
/// Verified against SGLang 0.5.9: `output_ids` arrives as a top-level per-chunk
/// delta, `meta_info` carries running `prompt_tokens`/`completion_tokens`/
/// `cached_tokens`, `finish_reason` is an object, and no `text` field is sent.
pub(crate) struct SglangTokensBackend;

impl Backend for SglangTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/generate"
    }

    fn build_payload(&self, req: &GenRequest) -> Value {
        let Prompt::Tokens(prompt_ids) = req.prompt;
        // No `model` field: an SGLang server hosts exactly one model. No
        // `return_logprob` either — `output_ids` is a native top-level response
        // field, so recovering ids out of per-token logprobs would only add
        // compute and serialization to the path we are timing.
        serde_json::json!({
            "rid": req.request_id,
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": req.max_tokens,
                "temperature": req.temperature,
                // Always decode to the trace's target length; synthetic prompts
                // otherwise emit EOS immediately and collapse the workload.
                "ignore_eos": true,
            },
            "stream": req.stream,
        })
    }

    fn parse_event(&self, value: &Value) -> StreamEvent {
        let token_ids = value
            .get("output_ids")
            .and_then(Value::as_array)
            .map(|output_ids| {
                output_ids
                    .iter()
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect::<Vec<u32>>()
            });
        let meta_info = value.get("meta_info");
        StreamEvent {
            // Under --skip-tokenizer-init the server never produces text.
            text_delta: None,
            token_ids,
            finish_reason: meta_info.and_then(sglang_finish_reason),
            usage: meta_info.and_then(sglang_usage),
        }
    }
}

/// SGLang reports `finish_reason` as an object (`{"type": "length"}`) rather
/// than the bare string the OpenAI schema uses. Accept both.
fn sglang_finish_reason(meta_info: &Value) -> Option<String> {
    let reason = meta_info.get("finish_reason")?;
    if let Some(reason) = reason.as_str() {
        return Some(reason.to_string());
    }
    reason.get("type")?.as_str().map(str::to_string)
}

/// SGLang carries token accounting in `meta_info`, not in an OpenAI `usage`
/// object. Its schema is fixed, so read the exact keys instead of searching the
/// provider alias list [`usage_cached_prompt_tokens`] needs.
///
/// `meta_info` rides on every streamed chunk with running counts. The shared
/// engine keeps the newest value for each field, so the final chunk's totals win.
fn sglang_usage(meta_info: &Value) -> Option<Usage> {
    let prompt_tokens = usage_usize(meta_info, "prompt_tokens");
    let completion_tokens = usage_usize(meta_info, "completion_tokens")
        .or_else(|| usage_usize(meta_info, "output_tokens"));
    let cached_prompt_tokens = usage_usize(meta_info, "cached_tokens");
    if prompt_tokens.is_none() && completion_tokens.is_none() && cached_prompt_tokens.is_none() {
        return None;
    }
    Some(Usage {
        prompt_tokens,
        completion_tokens,
        // SGLang does not report a combined total; derive it only when both
        // halves are present rather than reporting a partial sum.
        total_tokens: match (prompt_tokens, completion_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt.saturating_add(completion)),
            _ => None,
        },
        cached_prompt_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sglang_backend_sends_input_ids_without_model_or_logprobs() {
        let backend = SglangTokensBackend;
        let payload = backend.build_payload(&GenRequest {
            model: "ignored-by-sglang",
            request_id: "req-1",
            prompt: Prompt::Tokens(&[7, 8, 9]),
            max_tokens: 16,
            temperature: 0.0,
            stream: true,
        });

        assert_eq!(backend.endpoint_suffix(), "/generate");
        assert_eq!(payload["input_ids"], serde_json::json!([7, 8, 9]));
        assert_eq!(payload["sampling_params"]["max_new_tokens"], 16);
        assert_eq!(payload["sampling_params"]["temperature"], 0.0);
        assert_eq!(payload["sampling_params"]["ignore_eos"], true);
        assert_eq!(payload["stream"], true);
        // An SGLang server hosts one model, and output_ids is native: neither a
        // model field nor the logprob recovery path belongs in this payload.
        assert!(payload.get("model").is_none());
        assert!(payload.get("return_logprob").is_none());
        assert!(payload.get("prompt").is_none());
    }

    #[test]
    fn sglang_backend_normalizes_meta_info_into_usage() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({
            "output_ids": [101, 102],
            "meta_info": {
                "prompt_tokens": 512,
                "completion_tokens": 2,
                "cached_tokens": 496,
                "finish_reason": {"type": "length"}
            }
        }));

        assert_eq!(event.token_ids, Some(vec![101, 102]));
        // Under --skip-tokenizer-init there is no text to carry.
        assert!(event.text_delta.is_none());
        assert_eq!(event.finish_reason.as_deref(), Some("length"));
        let usage = event.usage.expect("meta_info must normalize into usage");
        assert_eq!(usage.prompt_tokens, Some(512));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.cached_prompt_tokens, Some(496));
        assert_eq!(usage.total_tokens, Some(514));
    }

    #[test]
    fn sglang_usage_falls_back_to_output_tokens_and_plain_finish_reason() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({
            "output_ids": [5],
            "meta_info": {"prompt_tokens": 4, "output_tokens": 1, "finish_reason": "stop"}
        }));

        let usage = event.usage.expect("usage");
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.cached_prompt_tokens, None);
        assert_eq!(event.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn sglang_chunk_without_meta_info_reports_no_usage() {
        let backend = SglangTokensBackend;
        let event = backend.parse_event(&serde_json::json!({"output_ids": [1, 2]}));

        assert_eq!(event.token_ids, Some(vec![1, 2]));
        assert!(event.usage.is_none());
        assert!(event.finish_reason.is_none());
    }
}
