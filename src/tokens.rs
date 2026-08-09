use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use tokenizers::Tokenizer;

use tracelab_replay::policy::{ContextChain, RawRound, SessionContextPolicy};
use crate::trace::SessionStep;

/// Cursor over a shared synthetic token pool. Each session seeds at a distinct
/// offset so replayed prompts are not byte-identical across sessions.
pub(crate) struct TokenProvider {
    pool: Arc<Vec<u32>>,
    cursor: usize,
}

impl TokenProvider {
    pub(crate) fn new(pool: Arc<Vec<u32>>, seed_offset: usize) -> Result<Self> {
        if pool.is_empty() {
            return Err(anyhow!("token pool is empty"));
        }
        Ok(Self {
            cursor: seed_offset % pool.len(),
            pool,
        })
    }

    pub(crate) fn take(&mut self, len: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.pool[self.cursor]);
            self.cursor = (self.cursor + 1) % self.pool.len();
        }
        out
    }
}

/// Builds each round's prompt token ids from the materialized prefix/append
/// split: replay `prefix_len` prior-context tokens, then append `input_len`
/// fresh synthetic tokens.
///
/// Length arithmetic lives in [`crate::policy`]; this type owns only the token
/// ids and the one thing the materializer cannot know — whether the replayed
/// conversation really produced the context the plan assumed.
pub(crate) struct PromptBuilder {
    token_provider: TokenProvider,
    context_tokens: Vec<u32>,
    chain: ContextChain,
}

/// One constructed prompt: the planned split, the split actually realized, and
/// the repair between them.
pub(crate) struct PromptBuild {
    pub(crate) prompt_ids: Vec<u32>,
    /// Cache-eligible prefix actually built. Equals the planned `prefix_len`
    /// unless the conversation came up short, in which case see
    /// `prefix_shortfall_len`.
    pub(crate) derived_prefix_len: usize,
    pub(crate) derived_append_len: usize,
    /// Planned prefix tokens the replayed conversation could not supply, filled
    /// with fresh ids and counted as append rather than as a cache hit.
    pub(crate) prefix_shortfall_len: usize,
    /// Raw trace prefix tokens the *policy* moved into fresh input.
    pub(crate) folded_tokens: usize,
    pub(crate) major_compaction: bool,
}

impl PromptBuilder {
    pub(crate) fn new(token_provider: TokenProvider) -> Self {
        Self {
            token_provider,
            context_tokens: Vec::new(),
            chain: ContextChain::new(),
        }
    }

    pub(crate) fn build_prompt(
        &mut self,
        step: &SessionStep,
        policy: SessionContextPolicy,
    ) -> PromptBuild {
        let planned = self.chain.materialize(
            RawRound {
                prefix_len: step.prefix_len,
                input_len: step.input_len,
                output_len: step.output_len,
            },
            policy,
        );

        // Recoverable shortage (the runtime half of the contract). The plan is
        // computed from target output lengths; a real server can return fewer
        // tokens or fail a round, leaving less context than planned. Fill the
        // gap with fresh ids and count it as append — never as a cache hit the
        // server cannot honour — instead of aborting an otherwise runnable
        // session.
        let derived_prefix_len = planned.prefix_len.min(self.context_tokens.len());
        let prefix_shortfall_len = planned.prefix_len - derived_prefix_len;
        let derived_append_len = planned.input_len + prefix_shortfall_len;

        let mut prompt_ids = self.context_tokens[..derived_prefix_len].to_vec();
        prompt_ids.extend(self.token_provider.take(derived_append_len));
        PromptBuild {
            prompt_ids,
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
            folded_tokens: planned.folded_tokens,
            major_compaction: planned.major_compaction,
        }
    }

    /// Carry this round's prompt plus the model's real output tokens forward as the next round's
    /// context. Using the real output (not synthetic) keeps the previous-output region of the next
    /// prefix byte-identical to what the server cached, so it stays prefix-cache-hittable.
    pub(crate) fn commit_output(&mut self, prompt_ids: Vec<u32>, output_ids: Vec<u32>) {
        self.context_tokens = prompt_ids;
        self.context_tokens.extend(output_ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(prefix_len: usize, input_len: usize, output_len: usize) -> SessionStep {
        SessionStep {
            request_id: "session_round_000000".to_string(),
            session_id: "session".to_string(),
            arrival_time: 0.0,
            round_idx: 0,
            prefix_len,
            input_len,
            output_len,
            tool_wait_after_ms: 0.0,
        }
    }

    fn builder(pool_len: usize) -> PromptBuilder {
        let pool = Arc::new((0..pool_len as u32).collect());
        PromptBuilder::new(TokenProvider::new(pool, 0).unwrap())
    }

    // The policy arithmetic itself is covered in `crate::policy`. These cases
    // cover what only this layer can get wrong: which token ids end up in the
    // prompt, and what happens when the live conversation is shorter than the
    // plan assumed.

    #[test]
    fn carries_exact_output_ids_into_the_next_prompt() {
        let mut builder = builder(200_000);
        let first = builder.build_prompt(
            &step(0, 4, 2),
            SessionContextPolicy::PrefixPreserving,
        );
        assert_eq!(first.prompt_ids, vec![0, 1, 2, 3]);
        builder.commit_output(first.prompt_ids, vec![91, 92]);

        let second = builder.build_prompt(
            &step(4, 4, 1),
            SessionContextPolicy::PrefixPreserving,
        );

        assert_eq!(second.prompt_ids, vec![0, 1, 2, 3, 91, 92, 4, 5]);
        assert_eq!(second.derived_prefix_len, 6);
        assert_eq!(second.derived_append_len, 2);
        assert_eq!(second.prefix_shortfall_len, 0);
    }

    #[test]
    fn trace_reported_first_round_prefix_becomes_fresh_tokens_not_a_claimed_hit() {
        let mut builder = builder(200_000);

        let built = builder.build_prompt(
            &step(12_461, 5_875, 124),
            SessionContextPolicy::TraceReported,
        );

        // The session has no context yet, so none of the prompt is cacheable.
        assert_eq!(built.derived_prefix_len, 0);
        assert_eq!(built.derived_append_len, 12_461 + 5_875);
        assert_eq!(built.folded_tokens, 12_461);
        assert_eq!(built.prompt_ids.len(), 12_461 + 5_875);
        assert_eq!(built.prompt_ids[..8], [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn repairs_a_prefix_the_live_conversation_could_not_supply() {
        let mut builder = builder(200_000);
        let first = builder.build_prompt(
            &step(0, 4, 2),
            SessionContextPolicy::PrefixPreserving,
        );
        // The server returned nothing, so the plan's 6 tokens of context are
        // really only the 4 prompt tokens.
        builder.commit_output(first.prompt_ids, Vec::new());

        let second = builder.build_prompt(
            &step(4, 4, 1),
            SessionContextPolicy::PrefixPreserving,
        );

        assert_eq!(second.derived_prefix_len, 4);
        assert_eq!(second.prefix_shortfall_len, 2);
        assert_eq!(second.derived_append_len, 4);
        assert_eq!(second.prompt_ids[..4], [0, 1, 2, 3]);
        assert_eq!(second.prompt_ids.len(), 8);
    }

    #[test]
    fn major_compaction_rebuilds_the_prompt_from_fresh_tokens() {
        let mut builder = builder(400_000);
        let first = builder.build_prompt(
            &step(0, 130_000, 10_000),
            SessionContextPolicy::PrefixPreserving,
        );
        builder.commit_output(first.prompt_ids, vec![7; 10_000]);

        let built = builder.build_prompt(
            &step(50_000, 20_000, 1),
            SessionContextPolicy::PrefixPreserving,
        );

        assert!(built.major_compaction);
        assert_eq!(built.derived_prefix_len, 0);
        assert_eq!(built.derived_append_len, 70_000);
        assert_eq!(built.prefix_shortfall_len, 0);
    }
}

/// Load a tokenizer from a local tokenizer.json / model directory, or download
/// it from the Hugging Face Hub when the path is a repo id.
pub(crate) fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    let path = std::path::Path::new(path);
    let tokenizer = if path.exists() {
        let tokenizer_path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };

        Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer {}: {err}",
                tokenizer_path.display()
            )
        })?
    } else {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|err| anyhow!("failed to create Hugging Face API client: {err}"))?;
        let repo = api.model(path.to_string_lossy().to_string());
        let tokenizer_path = repo.get("tokenizer.json").map_err(|err| {
            anyhow!(
                "failed to download tokenizer.json for {}: {err}",
                path.display()
            )
        })?;
        Tokenizer::from_file(tokenizer_path)
            .map_err(|err| anyhow!("failed to load downloaded tokenizer: {err}"))?
    };
    Ok(tokenizer)
}

/// Tokenize the text corpus into a bounded pool of token ids used as synthetic
/// prompt/input/output content.
pub(crate) fn build_token_pool(
    text_file: &str,
    tokenizer: &Tokenizer,
    limit: usize,
) -> Result<Vec<u32>> {
    let file = File::open(text_file)
        .with_context(|| format!("failed to open text corpus: {text_file}"))?;
    let reader = BufReader::new(file);
    let mut pool = Vec::with_capacity(limit);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let encoding = tokenizer
            .encode(line, false)
            .map_err(|err| anyhow!("tokenizer encode failed: {err}"))?;
        pool.extend(encoding.get_ids());
        if pool.len() >= limit {
            pool.truncate(limit);
            break;
        }
    }

    if pool.is_empty() {
        return Err(anyhow!("text corpus produced an empty token pool"));
    }
    Ok(pool)
}
