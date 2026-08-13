//! The fold over one response stream.
//!
//! Split out of the engine so it can be tested without a server. [`Self::absorb`]
//! is a pure function of one parsed event and the instant it arrived, which is
//! the whole reason this type exists: the fold decides every latency number the
//! run reports, and until now nothing exercised it below the HTTP layer.
//!
//! Everything here records *observations*. What they mean — whether the ids can
//! be trusted, which token count is authoritative — is decided afterwards, in
//! the engine.

use std::ops::ControlFlow;

use super::integrity::restates_accumulated_output;
use super::StreamEvent;

/// Server-reported token counts, folded across chunks.
///
/// Fields arrive on different chunks and each is independently optional, so the
/// newest present value wins per field rather than the newest chunk winning
/// wholesale. SGLang restates running counts on every chunk; vLLM sends one
/// terminal usage object. Both land correctly here.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct UsageFold {
    /// Whether any usage object was seen at all. Distinct from "every field is
    /// `None`": a server that reports an empty usage block has still reported.
    pub(super) seen: bool,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) completion_tokens: Option<usize>,
    pub(super) total_tokens: Option<usize>,
    pub(super) cached_prompt_tokens: Option<usize>,
}

/// What one response stream produced, and when.
#[derive(Debug)]
pub(super) struct StreamAccumulator {
    /// First non-empty *text* delta. The legacy TTFT, kept because a server that
    /// returns no token ids has nothing else to offer.
    pub(super) first_text_ms: Option<f64>,
    /// First event carrying generated token ids — the TTFT this runner reports.
    pub(super) first_token_id_ms: Option<f64>,
    pub(super) last_token_id_ms: Option<f64>,
    /// Tokens delivered in that first timed event. They share one observable
    /// arrival boundary, so they are excluded from TPOT's denominator.
    pub(super) first_token_event_tokens: usize,
    pub(super) token_event_count: usize,
    pub(super) usage_event_count: usize,
    /// Every parsed JSON object, usage-only ones included. Not a token count.
    pub(super) chunk_count: usize,
    pub(super) output_text: String,
    pub(super) output_token_ids: Vec<u32>,
    pub(super) usage: UsageFold,
    pub(super) finish_reason: Option<String>,
    /// Set when the stream itself is unusable. `None` means nothing went wrong
    /// *during* the fold; later checks may still fail the round.
    pub(super) failure: Option<String>,
}

impl StreamAccumulator {
    /// `expected_tokens` only sizes the id buffer. A server may return fewer or
    /// more; the capacity is a hint, never a bound.
    pub(super) fn new(expected_tokens: usize) -> Self {
        Self {
            first_text_ms: None,
            first_token_id_ms: None,
            last_token_id_ms: None,
            first_token_event_tokens: 0,
            token_event_count: 0,
            usage_event_count: 0,
            chunk_count: 0,
            output_text: String::new(),
            output_token_ids: Vec::with_capacity(expected_tokens),
            usage: UsageFold::default(),
            finish_reason: None,
            failure: None,
        }
    }

    /// Fold one parsed response object that arrived `at_ms` after the send.
    ///
    /// Returns `Break` when the stream must stop being read at all — reserved
    /// for a response that cannot be interpreted, not for an ordinary error
    /// status, which the engine records separately.
    pub(super) fn absorb(&mut self, event: StreamEvent, at_ms: f64) -> ControlFlow<()> {
        let token_ids = event.token_ids.unwrap_or_default();

        // Checked before the ids are folded in: appending a cumulative chunk
        // would multiply the output and there would be no way back.
        if restates_accumulated_output(&self.output_token_ids, &token_ids) {
            self.fail(format!(
                "server streamed cumulative output: a chunk repeated all {} tokens delivered so \
                 far. Launch SGLang with --stream-output (renamed --incremental-streaming-output \
                 in newer builds) so chunks are disjoint deltas.",
                self.output_token_ids.len(),
            ));
            return ControlFlow::Break(());
        }

        if !token_ids.is_empty() {
            if self.first_token_id_ms.is_none() {
                self.first_token_id_ms = Some(at_ms);
                self.first_token_event_tokens = token_ids.len();
            }
            self.last_token_id_ms = Some(at_ms);
            self.token_event_count += 1;
            self.output_token_ids.extend(token_ids);
        }

        if let Some(delta) = event.text_delta {
            if !delta.is_empty() {
                if self.first_text_ms.is_none() {
                    self.first_text_ms = Some(at_ms);
                }
                self.output_text.push_str(&delta);
            }
        }

        if let Some(reason) = event.finish_reason {
            self.finish_reason = Some(reason);
        }

        if let Some(usage) = event.usage {
            self.usage_event_count += 1;
            self.usage.seen = true;
            self.usage.prompt_tokens = usage.prompt_tokens.or(self.usage.prompt_tokens);
            self.usage.completion_tokens = usage.completion_tokens.or(self.usage.completion_tokens);
            self.usage.total_tokens = usage.total_tokens.or(self.usage.total_tokens);
            self.usage.cached_prompt_tokens = usage
                .cached_prompt_tokens
                .or(self.usage.cached_prompt_tokens);
        }

        self.chunk_count += 1;
        ControlFlow::Continue(())
    }

    /// Record a transport- or protocol-level failure. The first one wins: later
    /// errors are usually consequences of it.
    pub(super) fn fail(&mut self, message: String) {
        if self.failure.is_none() {
            self.failure = Some(message);
        }
    }

    /// Client-observed delivery time per token *after* the first timed event.
    ///
    /// The first event's tokens are excluded from the denominator because they
    /// share one arrival instant; counting them would report a per-token time
    /// nothing measured.
    pub(super) fn token_delivery_tpot_ms(&self) -> Option<f64> {
        let first = self.first_token_id_ms?;
        let last = self.last_token_id_ms?;
        let after_first = self
            .output_token_ids
            .len()
            .checked_sub(self.first_token_event_tokens)?;
        (after_first > 0 && last >= first).then(|| (last - first) / after_first as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Usage;

    fn tokens(ids: &[u32]) -> StreamEvent {
        StreamEvent {
            text_delta: None,
            token_ids: Some(ids.to_vec()),
            finish_reason: None,
            usage: None,
        }
    }

    fn usage(
        prompt: Option<usize>,
        completion: Option<usize>,
        cached: Option<usize>,
    ) -> StreamEvent {
        StreamEvent {
            text_delta: None,
            token_ids: None,
            finish_reason: None,
            usage: Some(Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: None,
                cached_prompt_tokens: cached,
            }),
        }
    }

    /// Fold an event the test only needs as setup, discarding the control flow
    /// it already expects to be `Continue`.
    fn feed(fold: &mut StreamAccumulator, event: StreamEvent, at_ms: f64) {
        assert_eq!(fold.absorb(event, at_ms), ControlFlow::Continue(()));
    }

    #[test]
    fn the_first_timed_event_anchors_ttft_and_is_kept_out_of_the_tpot_denominator() {
        let mut fold = StreamAccumulator::new(8);
        // A first chunk of three tokens: one observable instant, three tokens.
        feed(&mut fold, tokens(&[1, 2, 3]), 100.0);
        feed(&mut fold, tokens(&[4]), 120.0);
        feed(&mut fold, tokens(&[5]), 140.0);

        assert_eq!(fold.first_token_id_ms, Some(100.0));
        assert_eq!(fold.first_token_event_tokens, 3);
        assert_eq!(fold.last_token_id_ms, Some(140.0));
        assert_eq!(fold.token_event_count, 3);
        // 40 ms spanning the 2 tokens that arrived after the first event, not
        // the 5 tokens total — those first three were never timed apart.
        assert_eq!(fold.token_delivery_tpot_ms(), Some(20.0));
    }

    #[test]
    fn a_single_event_response_has_no_measurable_tpot() {
        let mut fold = StreamAccumulator::new(4);
        feed(&mut fold, tokens(&[1, 2]), 50.0);

        assert_eq!(fold.first_token_id_ms, Some(50.0));
        assert_eq!(fold.token_delivery_tpot_ms(), None);
    }

    #[test]
    fn empty_token_events_do_not_start_the_clock_or_count() {
        let mut fold = StreamAccumulator::new(4);
        feed(&mut fold, usage(Some(10), None, None), 5.0);
        feed(&mut fold, tokens(&[]), 6.0);

        assert_eq!(fold.first_token_id_ms, None);
        assert_eq!(fold.token_event_count, 0);
        // Both were still parsed objects.
        assert_eq!(fold.chunk_count, 2);
    }

    #[test]
    fn a_cumulative_chunk_breaks_the_stream_before_it_is_folded_in() {
        let mut fold = StreamAccumulator::new(8);
        feed(&mut fold, tokens(&[1, 2, 3]), 10.0);

        assert_eq!(
            fold.absorb(tokens(&[1, 2, 3, 4]), 20.0),
            ControlFlow::Break(())
        );
        // The offending chunk is not appended: the accumulator still holds
        // exactly what was legitimately delivered.
        assert_eq!(fold.output_token_ids, vec![1, 2, 3]);
        assert!(fold.failure.unwrap().contains("cumulative output"));
    }

    #[test]
    fn usage_folds_per_field_so_a_later_partial_chunk_cannot_erase_an_earlier_one() {
        let mut fold = StreamAccumulator::new(4);
        feed(&mut fold, usage(Some(512), Some(1), Some(496)), 10.0);
        // A terminal chunk that restates only the completion count.
        feed(&mut fold, usage(None, Some(2), None), 20.0);

        assert_eq!(
            fold.usage,
            UsageFold {
                seen: true,
                prompt_tokens: Some(512),
                completion_tokens: Some(2),
                total_tokens: None,
                cached_prompt_tokens: Some(496),
            }
        );
        assert_eq!(fold.usage_event_count, 2);
    }

    #[test]
    fn the_first_failure_wins() {
        let mut fold = StreamAccumulator::new(4);
        fold.fail("stream error".to_string());
        fold.fail("idle timeout".to_string());

        assert_eq!(fold.failure.as_deref(), Some("stream error"));
    }

    #[test]
    fn text_and_token_id_clocks_are_independent() {
        let mut fold = StreamAccumulator::new(4);
        feed(
            &mut fold,
            StreamEvent {
                text_delta: Some(String::new()),
                token_ids: None,
                finish_reason: None,
                usage: None,
            },
            10.0,
        );
        feed(
            &mut fold,
            StreamEvent {
                text_delta: Some("hi".to_string()),
                token_ids: Some(vec![7]),
                finish_reason: Some("length".to_string()),
                usage: None,
            },
            30.0,
        );

        // The empty delta must not start the text clock.
        assert_eq!(fold.first_text_ms, Some(30.0));
        assert_eq!(fold.first_token_id_ms, Some(30.0));
        assert_eq!(fold.output_text, "hi");
        assert_eq!(fold.finish_reason.as_deref(), Some("length"));
    }
}
