use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use tokenizers::Tokenizer;

use crate::schema::format::text_generation::session::SessionRound;

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

    pub(crate) fn build_prompt(&mut self, step: &SessionRound) -> PromptBuild {
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
        let repo = path.to_string_lossy().to_string();
        let tokenizer_path = fetch_hub_tokenizer(&repo).with_context(|| {
            format!("failed to download tokenizer.json for {repo}; pass a local tokenizer.json or model directory instead")
        })?;
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("failed to load downloaded tokenizer: {err}"))?
    };
    Ok(tokenizer)
}

/// Fetch `tokenizer.json` for a Hub repo id, caching it under `HF_HOME`.
///
/// The Hub answers `/resolve/main/...` with a *relative* `Location`, so the
/// client has to resolve it against the request URL rather than parse it on its
/// own. That is the one requirement here; everything else is a plain GET.
fn fetch_hub_tokenizer(repo: &str) -> Result<PathBuf> {
    if repo.is_empty() || repo.starts_with('/') || repo.contains("..") {
        bail!("{repo:?} is neither a local path nor a Hub repo id");
    }
    let cache = hub_cache_dir()?.join(repo.replace('/', "--"));
    let cached = cache.join("tokenizer.json");
    if cached.is_file() {
        return Ok(cached);
    }
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let url = format!(
        "{}/{repo}/resolve/main/tokenizer.json",
        endpoint.trim_end_matches('/')
    );
    let mut request = ureq::get(&url);
    // Gated repos need a token; public ones ignore it.
    if let Ok(token) =
        std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        if !token.trim().is_empty() {
            request = request.set("authorization", &format!("Bearer {}", token.trim()));
        }
    }
    let response = request
        .call()
        .map_err(|err| anyhow!("request to {url} failed: {err}"))?;
    let mut body = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|err| anyhow!("reading {url} failed: {err}"))?;
    if body.is_empty() {
        bail!("{url} returned an empty tokenizer.json");
    }
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("create tokenizer cache {}", cache.display()))?;
    // Write-then-rename so a killed download cannot leave a half file that the
    // next run would happily load as a cache hit.
    let staging = cache.join("tokenizer.json.partial");
    std::fs::write(&staging, &body).with_context(|| format!("write {}", staging.display()))?;
    std::fs::rename(&staging, &cached).with_context(|| format!("finalize {}", cached.display()))?;
    Ok(cached)
}

fn hub_cache_dir() -> Result<PathBuf> {
    if let Ok(home) = std::env::var("HF_HOME") {
        if !home.trim().is_empty() {
            return Ok(PathBuf::from(home).join("req-frontend-tokenizers"));
        }
    }
    let home = std::env::var("HOME").map_err(|_| anyhow!("neither HF_HOME nor HOME is set"))?;
    Ok(PathBuf::from(home).join(".cache/req-frontend/tokenizers"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repo_id_that_is_really_a_path_is_rejected_before_any_request() {
        // These reach the Hub branch only because they do not exist on disk;
        // sending them as repo ids would build a URL that escapes the repo.
        for bad in ["", "/etc/passwd", "../../secrets"] {
            let err = fetch_hub_tokenizer(bad).unwrap_err().to_string();
            assert!(
                err.contains("neither a local path nor a Hub repo id"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn the_cache_path_follows_hf_home_when_it_is_set() {
        // Serialized with the other env-reading test by running in one test.
        let previous = std::env::var("HF_HOME").ok();
        std::env::set_var("HF_HOME", "/tmp/hf-home-probe");
        let with_home = hub_cache_dir().unwrap();
        assert!(with_home.starts_with("/tmp/hf-home-probe"), "{with_home:?}");
        // An empty value is not a location; fall back rather than write to "/".
        std::env::set_var("HF_HOME", "  ");
        let blank = hub_cache_dir().unwrap();
        assert!(!blank.starts_with("  "), "{blank:?}");
        match previous {
            Some(value) => std::env::set_var("HF_HOME", value),
            None => std::env::remove_var("HF_HOME"),
        }
    }

    fn step(prefix_len: usize, input_len: usize, output_len: usize) -> SessionRound {
        SessionRound {
            request_id: "session_s1_round_000000".to_string(),
            session_id: "session".to_string(),
            arrival_time: 0.0,
            round_idx: 0,
            prefix_len,
            input_len,
            output_len,
            tool_wait_after_ms: 0.0,
            slo: Default::default(),
            priority: Default::default(),
            speculative: Default::default(),
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
