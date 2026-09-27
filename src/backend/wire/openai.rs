use anyhow::{bail, Result};
use serde::Serialize;

use super::super::{Backend, GenRequest, Prompt, StreamEvent};
use super::parse_openai_event;

/// OpenAI-compatible `/completions` protocol. Works against vLLM and SGLang's OpenAI endpoint.
pub(crate) struct OpenAiCompletionsBackend {
    /// Whether the body carries `rid` (`--omit-body-request-id` clears it).
    pub(crate) send_rid: bool,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    /// SGLang's OpenAI layer forwards this straight to GenerateReqInput, so its
    /// records carry the id this replay logged. vLLM has no such field; its base
    /// model allows unknown keys and ignores this one, taking the id from the
    /// x-request-id header instead. Omitted under `--omit-body-request-id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rid: Option<&'a str>,
    /// Raw token ids (OpenAI `prompt` accepts an int array): no client-side
    /// decode, and the server uses the exact ids so prefix-cache keys match what
    /// we constructed.
    prompt: &'a [u32],
    max_tokens: usize,
    temperature: f64,
    stream: bool,
    /// Always run decode to the trace's target length; synthetic prompts
    /// otherwise emit EOS almost immediately and collapse the decode workload.
    ignore_eos: bool,
    /// Echo the generated token ids (recent vLLM) so we carry the real output
    /// forward exactly. Older servers ignore this; we fall back to re-encoding
    /// the output text.
    return_token_ids: bool,
    /// Ask for the trailing usage chunk: server token counts and prefix-cache
    /// details are the whole point of this runner.
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

    fn prefix_cache_remedy(&self) -> &'static str {
        "vLLM reports cached tokens on this route behind --enable-prompt-tokens-details \
         (or ENABLE_PROMPT_TOKENS_DETAILS=1). SGLang's OpenAI layer does not report them \
         on any flag -- use --backend sglang-tokens, whose meta_info carries cached_tokens. \
         See README.md."
    }

    fn serialize_payload(&self, req: &GenRequest) -> Result<Vec<u8>> {
        let Prompt::Tokens(prompt) = req.prompt else {
            bail!("openai completions requires token-id prompts")
        };
        Ok(serde_json::to_vec(&Request {
            model: req.model,
            rid: self.send_rid.then_some(req.request_id),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(send_rid: bool) -> serde_json::Value {
        let bytes = OpenAiCompletionsBackend { send_rid }
            .serialize_payload(&GenRequest {
                model: "llama3-8b",
                request_id: "req-1",
                prompt: Prompt::Tokens(&[11, 22, 33]),
                max_tokens: 4,
                temperature: 0.0,
                stream: true,
            })
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn body_request_id_is_sent_unless_omitted() {
        let with = payload(true);
        assert_eq!(with["rid"], "req-1");
        assert_eq!(with["prompt"], serde_json::json!([11, 22, 33]));
        let without = payload(false);
        assert!(without.get("rid").is_none());
        let mut expected = with.clone();
        expected.as_object_mut().unwrap().remove("rid");
        assert_eq!(without, expected);
    }
}
