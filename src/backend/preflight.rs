//! The prefix-cache preflight: a hard gate run once before any workload.
//!
//! Without it every measured hit rate silently reads zero on a server that
//! simply does not report cached tokens, which is indistinguishable from a
//! server whose cache never hit.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use super::client::GenerationClient;
use super::{Backend, GenRequest, Prompt, Usage};

impl GenerationClient {
    /// Abort early unless the server actually reports prefix-cache hits.
    ///
    /// Servers omit cached-token details when nothing is cached, so a single response cannot tell
    /// "feature disabled" apart from "cache cold". We force a guaranteed hit by sending the same
    /// probe prompt twice and require the second response to report cached tokens. This also
    /// confirms prefix caching itself is enabled server-side.
    pub(crate) async fn preflight_cache_check(&self, probe_ids: &[u32]) -> Result<()> {
        let probe = Prompt::Tokens(probe_ids);
        // First request warms the prefix cache; the identical second request must hit it.
        self.post_probe(probe)
            .await
            .context("preflight warm-up request failed")?;
        let usage = self
            .post_probe(probe)
            .await
            .context("preflight cache-hit request failed")?;

        let usage = usage.ok_or_else(|| {
            anyhow!("preflight: server response carried no usage block; cannot verify prefix-cache reporting")
        })?;
        match usage.cached_prompt_tokens {
            Some(cached) if cached > 0 => Ok(()),
            other => Err(anyhow!(
                "preflight: server reported no prefix-cache hit (prompt_tokens={:?}, cached_tokens={:?}). {}",
                usage.prompt_tokens,
                other,
                self.backend.prefix_cache_remedy()
            )),
        }
    }

    /// Send one streaming completion and return its final normalized usage, if present.
    ///
    /// Both supported backends expose prompt-cache details in the final SSE usage
    /// chunk. Keeping preflight on that same wire path also avoids depending on a
    /// backend's optional non-streaming response schema.
    async fn post_probe(&self, probe: Prompt<'_>) -> Result<Option<Usage>> {
        let payload = self.backend.build_payload(&GenRequest {
            model: &self.model,
            // Not a trace request: named so it is obvious in a server log that
            // this row belongs to the prefix-cache preflight, not the workload.
            request_id: "req-frontend-prefix-cache-preflight",
            prompt: probe,
            max_tokens: 1,
            temperature: 0.0,
            stream: true,
        })?;
        let response = self
            .client
            .post(&self.endpoint)
            // vLLM DP ranks own independent prefix caches. Pin both preflight
            // requests to one rank so the second request tests the feature
            // instead of accidentally probing a different cache shard.
            // Servers that do not implement this vLLM routing header ignore it.
            .header("X-data-parallel-rank", "0")
            .json(&payload)
            .send()
            .await
            .map_err(|err| anyhow!("request error: {err}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HTTP {status}: {}",
                body.chars().take(200).collect::<String>()
            ));
        }
        let body = response
            .text()
            .await
            .map_err(|err| anyhow!("invalid streaming response: {err}"))?;
        Ok(final_usage_from_sse(self.backend.as_ref(), &body))
    }
}

fn final_usage_from_sse(backend: &dyn Backend, body: &str) -> Option<Usage> {
    let mut final_usage = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(usage) = backend.parse_event(&value).usage {
            final_usage = Some(usage);
        }
    }
    final_usage
}

#[cfg(test)]
mod tests {
    use super::super::wire::VllmTokensBackend;
    use super::*;

    #[test]
    fn the_remedy_names_the_transport_that_can_actually_answer() {
        use super::super::wire::{OpenAiCompletionsBackend, SglangTokensBackend};
        // Observed against a real SGLang server: its OpenAI layer returns a
        // usage block with no cached-token field on any flag, so telling an
        // operator to set a vLLM flag sends them somewhere that cannot help.
        let openai = OpenAiCompletionsBackend.prefix_cache_remedy();
        assert!(openai.contains("sglang-tokens"), "{openai}");
        assert!(
            openai.contains("--enable-prompt-tokens-details"),
            "{openai}"
        );
        // The native transports do report it, so they keep the generic advice.
        assert!(SglangTokensBackend
            .prefix_cache_remedy()
            .contains("prefix caching"));
    }

    #[test]
    fn vllm_tokens_backend_reads_final_streaming_usage_for_preflight() {
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"token_ids\":[101]}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":512,",
            "\"completion_tokens\":1,\"total_tokens\":513,",
            "\"prompt_tokens_details\":{\"cached_tokens\":496}}}\n\n",
            "data: [DONE]\n\n",
        );

        let usage = final_usage_from_sse(&VllmTokensBackend, body).expect("final usage");
        assert_eq!(usage.prompt_tokens, Some(512));
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.cached_prompt_tokens, Some(496));
    }
}
