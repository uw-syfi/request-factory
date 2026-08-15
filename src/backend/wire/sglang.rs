use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use super::super::{Backend, GenRequest, Prompt, StreamEvent, Usage};

/// SGLang native token-in/token-out `/generate` protocol.
pub(crate) struct SglangTokensBackend;

#[derive(Serialize)]
struct Request<'a> {
    rid: &'a str,
    input_ids: &'a [u32],
    sampling_params: SamplingParams,
    stream: bool,
}

#[derive(Serialize)]
struct SamplingParams {
    max_new_tokens: usize,
    temperature: f64,
    ignore_eos: bool,
}

#[derive(Deserialize)]
struct Event {
    output_ids: Option<Vec<u32>>,
    meta_info: Option<MetaInfo>,
}

#[derive(Deserialize)]
struct MetaInfo {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
    output_tokens: Option<usize>,
    cached_tokens: Option<usize>,
    finish_reason: Option<FinishReason>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FinishReason {
    Plain(String),
    Object { r#type: String },
}

impl Backend for SglangTokensBackend {
    fn endpoint_suffix(&self) -> &str {
        "/generate"
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Tokens(input_ids) = req.prompt else {
            bail!("sglang-tokens requires token-id prompts")
        };
        Ok(serde_json::to_vec(&Request {
            rid: req.request_id,
            input_ids,
            sampling_params: SamplingParams {
                max_new_tokens: req.max_tokens,
                temperature: req.temperature,
                ignore_eos: true,
            },
            stream: req.stream,
        })?)
    }

    fn parse_event(&self, data: &[u8]) -> serde_json::Result<StreamEvent> {
        let wire: Event = serde_json::from_slice(data)?;
        let (finish_reason, usage) = wire.meta_info.map_or((None, None), |meta| {
            let completion_tokens = meta.completion_tokens.or(meta.output_tokens);
            let finish_reason = meta.finish_reason.map(|reason| match reason {
                FinishReason::Plain(reason) => reason,
                FinishReason::Object { r#type } => r#type,
            });
            let usage_present = meta.prompt_tokens.is_some()
                || completion_tokens.is_some()
                || meta.cached_tokens.is_some();
            let usage = usage_present.then_some(Usage {
                prompt_tokens: meta.prompt_tokens,
                completion_tokens,
                total_tokens: match (meta.prompt_tokens, completion_tokens) {
                    (Some(prompt), Some(completion)) => Some(prompt.saturating_add(completion)),
                    _ => None,
                },
                cached_prompt_tokens: meta.cached_tokens,
            });
            (finish_reason, usage)
        });
        Ok(StreamEvent {
            text_delta: None,
            token_ids: wire.output_ids,
            finish_reason,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sglang_backend_sends_input_ids_without_model_or_logprobs() {
        let backend = SglangTokensBackend;
        let payload = backend
            .serialize_payload(&GenRequest {
                model: "ignored-by-sglang",
                request_id: "req-1",
                prompt: Prompt::Tokens(&[7, 8, 9]),
                max_tokens: 16,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(backend.endpoint_suffix(), "/generate");
        assert_eq!(payload["input_ids"], serde_json::json!([7, 8, 9]));
        assert_eq!(payload["sampling_params"]["max_new_tokens"], 16);
        assert_eq!(payload["sampling_params"]["temperature"], 0.0);
        assert_eq!(payload["sampling_params"]["ignore_eos"], true);
        assert_eq!(payload["stream"], true);
        assert!(payload.get("model").is_none());
        assert!(payload.get("return_logprob").is_none());
        assert!(payload.get("prompt").is_none());
    }

    #[test]
    fn sglang_backend_normalizes_meta_info_into_usage() {
        let event = SglangTokensBackend
            .parse_event(
                br#"{"output_ids":[101,102],"meta_info":{
            "prompt_tokens":512,"completion_tokens":2,"cached_tokens":496,
            "finish_reason":{"type":"length"}}}"#,
            )
            .unwrap();
        assert_eq!(event.token_ids, Some(vec![101, 102]));
        assert!(event.text_delta.is_none());
        assert_eq!(event.finish_reason.as_deref(), Some("length"));
        let usage = event.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(512));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.cached_prompt_tokens, Some(496));
        assert_eq!(usage.total_tokens, Some(514));
    }

    #[test]
    fn sglang_usage_falls_back_to_output_tokens_and_plain_finish_reason() {
        let event = SglangTokensBackend
            .parse_event(
                br#"{"output_ids":[5],"meta_info":{
            "prompt_tokens":4,"output_tokens":1,"finish_reason":"stop"}}"#,
            )
            .unwrap();
        let usage = event.usage.expect("usage");
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.cached_prompt_tokens, None);
        assert_eq!(event.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn sglang_chunk_without_meta_info_reports_no_usage() {
        let event = SglangTokensBackend
            .parse_event(br#"{"output_ids":[1,2]}"#)
            .unwrap();
        assert_eq!(event.token_ids, Some(vec![1, 2]));
        assert!(event.usage.is_none());
        assert!(event.finish_reason.is_none());
    }
}
