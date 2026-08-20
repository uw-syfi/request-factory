//! Per-protocol wire adapters. Pure, synchronous JSON shaping — one file per
//! protocol, plus the accessor helpers they share for reading server usage.

mod openai;
mod openai_chat;
mod sglang;
mod vllm;

use serde_json::Value;

use crate::backend::Dialect;
use crate::cli::BackendKind;

use super::Backend;
pub(super) use openai::OpenAiCompletionsBackend;
pub(super) use openai_chat::OpenAiChatBackend;
pub(super) use sglang::SglangTokensBackend;
pub(super) use vllm::VllmTokensBackend;

/// Build the backend adapter selected on the command line.
pub(crate) fn build_backend(kind: BackendKind, dialect: &'static Dialect) -> Box<dyn Backend> {
    match kind {
        BackendKind::Openai => Box::new(OpenAiCompletionsBackend),
        BackendKind::VllmTokens => Box::new(VllmTokensBackend),
        BackendKind::SglangTokens => Box::new(SglangTokensBackend),
        BackendKind::OpenaiChat => Box::new(OpenAiChatBackend(dialect)),
        BackendKind::OpenaiImages
        | BackendKind::OpenaiSpeech
        | BackendKind::OpenaiImageEdits
        | BackendKind::OpenaiVideos
        | BackendKind::OpenaiTranscriptions
        | BackendKind::OpenaiTranslations => {
            unreachable!("generated-media backends use MediaClient")
        }
    }
}

fn usage_usize(usage: &Value, key: &str) -> Option<usize> {
    usage
        .get(key)?
        .as_u64()
        .and_then(|value| value.try_into().ok())
}

fn usage_cached_prompt_tokens(usage: &Value) -> Option<usize> {
    [
        &["prompt_tokens_details", "cached_tokens"][..],
        &["cached_tokens"][..],
        &["cached_input_tokens"][..],
        &["cache_read_input_tokens"][..],
        &["prompt_cached_tokens"][..],
        &["num_cached_tokens"][..],
    ]
    .into_iter()
    .find_map(|path| value_at_path(usage, path)?.as_u64()?.try_into().ok())
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    Some(cursor)
}
