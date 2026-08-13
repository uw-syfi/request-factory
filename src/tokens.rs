use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use tokenizers::Tokenizer;

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
/// The split itself is not decided here — it was resolved when the canonical
/// trace was generated. This type owns only the token ids and the one thing that
/// generation could not know: whether the replayed conversation really produced
/// the context the file assumed.
pub(crate) struct PromptBuilder {
    token_provider: TokenProvider,
    context_tokens: Vec<u32>,
}

/// One constructed prompt: the trace's split, the split actually realized, and
/// the repair between them.
pub(crate) struct PromptBuild {
    pub(crate) prompt_ids: Vec<u32>,
    /// Cache-eligible prefix actually built. Equals the trace's `prefix_len`
    /// unless the conversation came up short, in which case see
    /// `prefix_shortfall_len`.
    pub(crate) derived_prefix_len: usize,
    pub(crate) derived_append_len: usize,
    /// Trace prefix tokens the replayed conversation could not supply, filled
    /// with fresh ids and counted as append rather than as a cache hit. The only
    /// place a live run departs from the file it is replaying.
    pub(crate) prefix_shortfall_len: usize,
}

impl PromptBuilder {
    pub(crate) fn new(token_provider: TokenProvider) -> Self {
        Self {
            token_provider,
            context_tokens: Vec::new(),
        }
    }

    pub(crate) fn build_prompt(&mut self, step: &SessionStep) -> PromptBuild {
        // Recoverable shortage (the runtime half of the contract). The file's
        // split was computed from target output lengths; a real server can
        // return fewer tokens or fail a round, leaving less context than the
        // trace assumed. Fill the gap with fresh ids and count it as append —
        // never as a cache hit the server cannot honour — instead of aborting an
        // otherwise runnable session.
        let derived_prefix_len = step.prefix_len.min(self.context_tokens.len());
        let prefix_shortfall_len = step.prefix_len - derived_prefix_len;
        let derived_append_len = step.input_len + prefix_shortfall_len;

        let mut prompt_ids = self.context_tokens[..derived_prefix_len].to_vec();
        prompt_ids.extend(self.token_provider.take(derived_append_len));
        PromptBuild {
            prompt_ids,
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
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
            request_id: "session_s1_round_000000".to_string(),
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

    // How a trace's split was chosen is covered by `tracegen`'s policy tests.
    // These cases cover what only this layer can get wrong: which token ids end
    // up in the prompt, and what happens when the live conversation turns out
    // shorter than the file assumed.

    #[test]
    fn carries_exact_output_ids_into_the_next_prompt() {
        let mut builder = builder(200_000);
        let first = builder.build_prompt(&step(0, 4, 2));
        assert_eq!(first.prompt_ids, vec![0, 1, 2, 3]);
        builder.commit_output(first.prompt_ids, vec![91, 92]);

        // The trace's round 1 reuses the whole 6-token round 0 — prompt *and*
        // output — so the server's real ids must reappear inside the prefix.
        let second = builder.build_prompt(&step(6, 2, 1));

        assert_eq!(second.prompt_ids, vec![0, 1, 2, 3, 91, 92, 4, 5]);
        assert_eq!(second.derived_prefix_len, 6);
        assert_eq!(second.derived_append_len, 2);
        assert_eq!(second.prefix_shortfall_len, 0);
    }

    #[test]
    fn repairs_a_prefix_the_live_conversation_could_not_supply() {
        let mut builder = builder(200_000);
        let first = builder.build_prompt(&step(0, 4, 2));
        // The server returned nothing, so the 6 tokens of context the trace
        // counted on are really only the 4 prompt tokens.
        builder.commit_output(first.prompt_ids, Vec::new());

        let second = builder.build_prompt(&step(6, 2, 1));

        assert_eq!(second.derived_prefix_len, 4);
        assert_eq!(second.prefix_shortfall_len, 2);
        assert_eq!(second.derived_append_len, 4);
        assert_eq!(second.prompt_ids[..4], [0, 1, 2, 3]);
        assert_eq!(second.prompt_ids.len(), 8);
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
    // `Tokenizer::encode_batch` preserves input order while letting the HF
    // tokenizers backend parallelize independent lines with Rayon. Keep the
    // batch deliberately small: corpus lines can be arbitrarily large, and all
    // batch encodings coexist until their IDs have been copied into the pool.
    const TOKENIZER_BATCH_LINES: usize = 256;

    let file = File::open(text_file)
        .with_context(|| format!("failed to open text corpus: {text_file}"))?;
    let reader = BufReader::new(file);
    let mut pool = Vec::with_capacity(limit);
    let mut line_batch = Vec::with_capacity(TOKENIZER_BATCH_LINES);

    let append_batch = |lines: Vec<String>, pool: &mut Vec<u32>| -> Result<bool> {
        let encodings = tokenizer
            .encode_batch(lines, false)
            .map_err(|err| anyhow!("tokenizer batch encode failed: {err}"))?;
        for encoding in encodings {
            let remaining = limit.saturating_sub(pool.len());
            let token_ids = encoding.get_ids();
            pool.extend(&token_ids[..token_ids.len().min(remaining)]);
            if pool.len() >= limit {
                return Ok(true);
            }
        }
        Ok(false)
    };

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        line_batch.push(line);
        if line_batch.len() == TOKENIZER_BATCH_LINES
            && append_batch(std::mem::take(&mut line_batch), &mut pool)?
        {
            break;
        }
    }
    if pool.len() < limit && !line_batch.is_empty() {
        append_batch(line_batch, &mut pool)?;
    }

    if pool.is_empty() {
        return Err(anyhow!("text corpus produced an empty token pool"));
    }
    Ok(pool)
}
