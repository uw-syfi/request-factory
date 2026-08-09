use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::cli::SessionContextPolicy;
use crate::trace::SessionStep;

pub(crate) const MAJOR_COMPACTION_MIN_DROP_TOKENS: usize = 64_000;
pub(crate) const MAJOR_COMPACTION_MIN_DROP_RATIO: f64 = 0.5;

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

/// Builds each round's prompt token ids by replaying `prefix_len` prior-context
/// tokens and appending `input_len` fresh synthetic tokens.
pub(crate) struct PromptBuilder {
    token_provider: TokenProvider,
    context_tokens: Vec<u32>,
}

/// One constructed prompt plus the cache-relevant shape implied by the chosen
/// session context policy.
pub(crate) struct PromptBuild {
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) derived_prefix_len: usize,
    pub(crate) derived_append_len: usize,
    pub(crate) major_compaction: bool,
}

impl PromptBuilder {
    pub(crate) fn new(token_provider: TokenProvider) -> Self {
        Self {
            token_provider,
            context_tokens: Vec::new(),
        }
    }

    pub(crate) fn build_prompt(
        &mut self,
        step: &SessionStep,
        policy: SessionContextPolicy,
    ) -> PromptBuild {
        match policy {
            SessionContextPolicy::TraceReported => self.build_trace_reported_prompt(step),
            SessionContextPolicy::Monotonic => self.build_monotonic_prompt(step),
        }
    }

    fn build_trace_reported_prompt(&mut self, step: &SessionStep) -> PromptBuild {
        if self.context_tokens.len() < step.prefix_len {
            let need = step.prefix_len - self.context_tokens.len();
            self.context_tokens.extend(self.token_provider.take(need));
        }

        let mut prompt_ids = self.context_tokens[..step.prefix_len].to_vec();
        prompt_ids.extend(self.token_provider.take(step.input_len));
        PromptBuild {
            prompt_ids,
            derived_prefix_len: step.prefix_len,
            derived_append_len: step.input_len,
            major_compaction: false,
        }
    }

    fn build_monotonic_prompt(&mut self, step: &SessionStep) -> PromptBuild {
        let target_prompt_len = step.prefix_len.saturating_add(step.input_len);
        let previous_context_len = self.context_tokens.len();
        let dropped_tokens = previous_context_len.saturating_sub(target_prompt_len);
        let drop_ratio = if previous_context_len == 0 {
            0.0
        } else {
            dropped_tokens as f64 / previous_context_len as f64
        };
        let major_compaction = dropped_tokens >= MAJOR_COMPACTION_MIN_DROP_TOKENS
            && drop_ratio >= MAJOR_COMPACTION_MIN_DROP_RATIO;

        if major_compaction {
            return PromptBuild {
                prompt_ids: self.token_provider.take(target_prompt_len),
                derived_prefix_len: 0,
                derived_append_len: target_prompt_len,
                major_compaction: true,
            };
        }

        let mut prompt_ids = self.context_tokens.clone();
        let derived_prefix_len = prompt_ids.len();
        let derived_append_len = target_prompt_len.saturating_sub(derived_prefix_len);
        prompt_ids.extend(self.token_provider.take(derived_append_len));
        PromptBuild {
            prompt_ids,
            derived_prefix_len,
            derived_append_len,
            major_compaction: false,
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

    fn step(prefix_len: usize, input_len: usize) -> SessionStep {
        SessionStep {
            session_id: "session".to_string(),
            arrival_time: 0.0,
            round_idx: 0,
            prefix_len,
            input_len,
            output_len: 1,
            tool_wait_after_ms: 0.0,
        }
    }

    fn builder(pool_len: usize) -> PromptBuilder {
        let pool = Arc::new((0..pool_len as u32).collect());
        PromptBuilder::new(TokenProvider::new(pool, 0).unwrap())
    }

    #[test]
    fn monotonic_policy_retains_full_context_across_small_reduction() {
        let mut builder = builder(200_000);
        builder.context_tokens = vec![7; 10_000];

        let built = builder.build_prompt(&step(8_500, 1_000), SessionContextPolicy::Monotonic);

        assert_eq!(built.prompt_ids.len(), 10_000);
        assert_eq!(built.derived_prefix_len, 10_000);
        assert_eq!(built.derived_append_len, 0);
        assert!(!built.major_compaction);
        assert!(built.prompt_ids.iter().all(|token_id| *token_id == 7));
    }

    #[test]
    fn monotonic_policy_appends_only_to_reach_trace_total() {
        let mut builder = builder(200_000);
        builder.context_tokens = vec![7; 10_000];

        let built = builder.build_prompt(&step(10_000, 2_000), SessionContextPolicy::Monotonic);

        assert_eq!(built.prompt_ids.len(), 12_000);
        assert_eq!(built.derived_prefix_len, 10_000);
        assert_eq!(built.derived_append_len, 2_000);
        assert!(!built.major_compaction);
        assert!(built.prompt_ids[..10_000]
            .iter()
            .all(|token_id| *token_id == 7));
    }

    #[test]
    fn monotonic_policy_carries_exact_output_ids_into_next_prompt() {
        let mut builder = builder(200_000);
        let first = builder.build_prompt(&step(0, 4), SessionContextPolicy::Monotonic);
        assert_eq!(first.prompt_ids, vec![0, 1, 2, 3]);
        builder.commit_output(first.prompt_ids, vec![91, 92]);

        let second = builder.build_prompt(&step(4, 4), SessionContextPolicy::Monotonic);

        assert_eq!(second.prompt_ids, vec![0, 1, 2, 3, 91, 92, 4, 5]);
        assert_eq!(second.derived_prefix_len, 6);
        assert_eq!(second.derived_append_len, 2);
    }

    #[test]
    fn monotonic_policy_rebuilds_only_large_half_context_drop() {
        let mut builder = builder(200_000);
        builder.context_tokens = vec![7; 140_000];

        let built = builder.build_prompt(&step(50_000, 20_000), SessionContextPolicy::Monotonic);

        assert_eq!(built.prompt_ids.len(), 70_000);
        assert_eq!(built.derived_prefix_len, 0);
        assert_eq!(built.derived_append_len, 70_000);
        assert!(built.major_compaction);
        assert!(built.prompt_ids.iter().any(|token_id| *token_id != 7));
    }

    #[test]
    fn monotonic_policy_keeps_context_when_only_one_major_threshold_matches() {
        let mut builder = builder(200_000);
        builder.context_tokens = vec![7; 200_000];

        let built = builder.build_prompt(&step(130_000, 0), SessionContextPolicy::Monotonic);

        assert_eq!(built.prompt_ids.len(), 200_000);
        assert_eq!(built.derived_prefix_len, 200_000);
        assert!(!built.major_compaction);
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
