//! The prefix-cache preflight: a hard gate run once before any workload.
//!
//! Without it every measured hit rate silently reads zero on a server that
//! simply does not report cached tokens, which is indistinguishable from a
//! server whose cache never hit.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use super::client::GenerationClient;
use super::{Backend, GenRequest, PreparedInputPart, Prompt, Usage};
use crate::schema::Modality;

/// A small valid PNG, used only to ask a server whether it saw an image.
const PROBE_PNG_BASE64: &str = concat!(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAB20lEQVR42tXOhzoQAACF0T9KWtqlpb0XKpXQ0NaS",
    "toZEEy1JUtqlRVvRRor2QDtFU4uW0F7aigZfeY57nuCAoVGxkqXLVTStZla7XsMmzc1bWbW36dilW0+Hfo6Dho4Y",
    "5TJu/CSPqTNm+c6dv3DJ8pVrgjds3hq2c09EVPShoydiT5+7eDnp+q079x88fpr54vW7rM/ffuT+yf9vUMS4hEnZ",
    "CpWr1qhVt0HjZi0t27TrYNe5a4/efQc4DRk+coyr+8QpXtO9Z/vNW7B4WeDqoPWbQkJ37A7fd+DgkeOn4s9eSEi8",
    "dvP2vdRHaRnPX7398Olrds7vvH+FChctXqpM+UpVqtesU79R0xYWrdta26r/OyH+t0f83x3xfy/E/30Q//dH/D8Q",
    "8f9gxP/DEP87I/4fjfh/LOJ/N8T/ExD/T0b874n4fxri/5mI/30Q/89B/O+P+D8A8f8ixP9LEf+vQPy/CvH/WsT/",
    "6xD/b0T8vwXx/zbE/9sR/+9C/L8X8X8k4v/9iP9jEP8fRvx/DPH/ScT/cYj/zyD+P4/4/xLi/yuI/68i/r+B+D8Z",
    "8f9dxP8piP8fIv5/gvg/HfH/M8T/LxH/v0H8/x7x/0fE/18Q/39H/P8T8f8vxP9/CwCxn/Favl2o3gAAAABJRU5E",
    "rkJggg==",
);

impl GenerationClient {
    /// Abort unless the server actually consumed the media we sent it.
    ///
    /// Encoding media is the one thing every serving system spells differently,
    /// and getting it wrong is silent: a server that does not recognize the
    /// field ignores it, answers the text alone, and the run reports success
    /// having measured a text-only workload. Observed directly -- vLLM under the
    /// `sglang-omni` encoding returned 26 prompt tokens where the same trace
    /// under `vllm` returned 37, with every request marked SUCCESS.
    ///
    /// So: send the same text twice, once with an image, and require the
    /// server's own prompt-token count to move. This is the media counterpart of
    /// the prefix-cache preflight, and it fails for the same reason -- a number
    /// the tool cannot verify is worse than no number.
    pub(crate) async fn preflight_media_registered(&self) -> Result<()> {
        let text = PreparedInputPart::Text("Describe this image.".to_string());
        let image = PreparedInputPart::Media {
            modality: Modality::Image,
            data_url: format!("data:image/png;base64,{PROBE_PNG_BASE64}"),
        };
        let text_only = [text.clone()];
        let with_image = [text, image];

        let bare = self
            .post_probe(Prompt::Parts(&text_only))
            .await
            .context("media preflight: text-only probe failed")?;
        let media = self
            .post_probe(Prompt::Parts(&with_image))
            .await
            .context("media preflight: probe carrying an image failed")?;

        let (Some(bare), Some(media)) = (
            bare.and_then(|usage| usage.prompt_tokens),
            media.and_then(|usage| usage.prompt_tokens),
        ) else {
            // Nothing to compare. Say so rather than implying a check ran.
            eprintln!(
                "media preflight | server reported no prompt-token counts; cannot verify that \
                 it consumed the media. Check server.dialect against the server's own docs."
            );
            return Ok(());
        };
        if media > bare {
            return Ok(());
        }
        Err(anyhow!(
            "media preflight: the server's prompt-token count did not change when an image was \
             added ({bare} without, {media} with), so it is ignoring the media this dialect \
             sends and the run would measure a text-only workload. The encoding is almost \
             certainly wrong for this server -- check --dialect."
        ))
    }

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
        if !response.status().is_success() {
            return Err(anyhow!("{}", super::http_failure(response).await));
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
    fn the_probe_image_is_a_real_png() {
        // The whole check rests on a server tokenizing this into visible
        // prompt tokens. A malformed probe would be rejected instead, and the
        // preflight would fail every run for the wrong reason.
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(PROBE_PNG_BASE64)
            .expect("probe must be valid base64");
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
        assert_eq!(&bytes[12..16], b"IHDR");
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        // Large enough that a vision model does not discard it as a thumbnail.
        assert_eq!((width, height), (64, 64));
        assert_eq!(&bytes[bytes.len() - 8..bytes.len() - 4], b"IEND");
    }

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
