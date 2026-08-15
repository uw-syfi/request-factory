//! Per-protocol wire adapters. Pure, synchronous JSON shaping — one file per
//! protocol, plus the accessor helpers they share for reading server usage.

mod openai;
mod openai_chat;
mod sglang;
mod vllm;

use serde::Deserialize;

use crate::cli::BackendKind;

use super::{Backend, StreamEvent, Usage};
pub(super) use openai::OpenAiCompletionsBackend;
pub(super) use openai_chat::OpenAiChatBackend;
pub(super) use sglang::SglangTokensBackend;
pub(super) use vllm::VllmTokensBackend;

/// Build the backend adapter selected on the command line.
pub(crate) fn build_backend(kind: BackendKind) -> Box<dyn Backend> {
    match kind {
        BackendKind::Openai => Box::new(OpenAiCompletionsBackend),
        BackendKind::VllmTokens => Box::new(VllmTokensBackend),
        BackendKind::SglangTokens => Box::new(SglangTokensBackend),
        BackendKind::OpenaiChat => Box::new(OpenAiChatBackend),
        BackendKind::OpenaiImages | BackendKind::OpenaiSpeech => {
            unreachable!("generated-media backends use MediaClient")
        }
    }
}

#[derive(Default, Deserialize)]
struct OpenAiEvent {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Default, Deserialize)]
struct OpenAiChoice {
    text: Option<String>,
    token_ids: Option<Vec<u32>>,
    finish_reason: Option<String>,
    delta: Option<OpenAiDelta>,
}

#[derive(Default, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

#[derive(Default, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
    total_tokens: Option<usize>,
    prompt_tokens_details: Option<PromptTokenDetails>,
    cached_tokens: Option<usize>,
    cached_input_tokens: Option<usize>,
    cache_read_input_tokens: Option<usize>,
    prompt_cached_tokens: Option<usize>,
    num_cached_tokens: Option<usize>,
}

#[derive(Default, Deserialize)]
struct PromptTokenDetails {
    cached_tokens: Option<usize>,
}

impl OpenAiUsage {
    fn normalize(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            cached_prompt_tokens: self
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens)
                .or(self.cached_tokens)
                .or(self.cached_input_tokens)
                .or(self.cache_read_input_tokens)
                .or(self.prompt_cached_tokens)
                .or(self.num_cached_tokens),
        }
    }
}

fn parse_openai_event(
    data: &[u8],
    chat: bool,
    include_text: bool,
) -> serde_json::Result<StreamEvent> {
    let wire: OpenAiEvent = serde_json::from_slice(data)?;
    // OpenAI-compatible schemas can carry several choices, but this client
    // requests one and has always normalized the first if a server sends more.
    let choice = wire.choices.into_iter().next();
    let (text_delta, token_ids, finish_reason) = choice.map_or((None, None, None), |choice| {
        let text_delta = if chat {
            choice.delta.and_then(|delta| delta.content)
        } else if include_text {
            choice.text
        } else {
            None
        };
        (text_delta, choice.token_ids, choice.finish_reason)
    });
    Ok(StreamEvent {
        text_delta,
        token_ids,
        finish_reason,
        usage: wire.usage.map(OpenAiUsage::normalize),
    })
}
