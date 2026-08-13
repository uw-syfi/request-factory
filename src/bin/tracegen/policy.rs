//! Session context policy as pure length arithmetic.
//!
//! A raw coding-agent trace reports, per round, how many prompt tokens were
//! cache-eligible prefix and how many were freshly appended. Those numbers
//! describe the conversation the agent actually had; they are not directly
//! replayable, because a replay starts with no context at all and because a
//! later round may ask for a prefix the replayed conversation never produced.
//!
//! Resolving that gap is *materialization*: turning a raw round into a round
//! whose `prefix_len` is guaranteed to exist by the time it runs. This module
//! owns that arithmetic and nothing else — no token ids, no tokenizer, no I/O.
//!
//! It lives inside `tracegen` because materialization happens exactly once, when
//! a canonical trace is generated. A replay runtime never performs it: it reads
//! a file whose split was already resolved, and the policy that resolved it is
//! recorded in that file's manifest.
//!
//! The two policies differ only in what they preserve when raw and replayed
//! context disagree:
//!
//! - [`SessionContextPolicy::TraceReported`] preserves the *reported split*. It
//!   never claims more prefix than exists, and moves the shortfall into fresh
//!   input so the total prompt length still matches the trace.
//! - [`SessionContextPolicy::Monotonic`] preserves the *longest real
//!   prefix*. It reuses as much of the replayed conversation as the trace's
//!   total prompt length allows, and only starts over on a major compaction.
//!
//! Both keep `prefix_len + input_len` equal to the trace's total prompt length,
//! so neither changes the shape of the workload — only the cache assumption.

use clap::ValueEnum;

/// How raw trace lengths become a round's replayable prefix/append split.
///
/// This is a property of the trace, not of a runtime: it is chosen here, when
/// the canonical execution trace is generated, and recorded in that trace's
/// manifest. Nothing downstream gets to reinterpret it.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionContextPolicy {
    /// Preserve the trace-reported split, folding any prefix the replayed
    /// conversation cannot supply into fresh input.
    TraceReported,
    /// Preserve the longest prefix the replayed conversation can supply, up to
    /// the trace's total prompt length, resetting only on a major compaction.
    Monotonic,
}

impl SessionContextPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::TraceReported => "trace_reported",
            Self::Monotonic => "monotonic",
        }
    }
}

/// Drop size, in tokens, above which a context reduction may count as a major
/// compaction. Both this and [`MAJOR_COMPACTION_MIN_DROP_RATIO`] must hold: a
/// large absolute drop out of a much larger context is ordinary trimming, and a
/// large relative drop out of a small context is noise.
pub const MAJOR_COMPACTION_MIN_DROP_TOKENS: usize = 64_000;

/// Fraction of the previous context that must disappear for a reduction to
/// count as a major compaction.
pub const MAJOR_COMPACTION_MIN_DROP_RATIO: f64 = 0.5;

/// One round as the raw trace reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawRound {
    pub prefix_len: usize,
    pub input_len: usize,
    pub output_len: usize,
}

/// One round after policy materialization, with the accounting that explains
/// how it differs from the raw round it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializedRound {
    /// Prefix guaranteed to exist in the replayed conversation when this round
    /// runs. Cache-*eligible*, never a guaranteed hit.
    pub prefix_len: usize,
    /// Freshly appended tokens, including anything folded out of `prefix_len`.
    pub input_len: usize,
    pub output_len: usize,
    /// Raw prefix tokens that did not exist in the replayed conversation and
    /// were therefore moved into `input_len`. Zero on a faithful round.
    pub folded_tokens: usize,
    /// This round restarts the conversation rather than extending it.
    pub major_compaction: bool,
}

impl MaterializedRound {
    pub fn total_prompt_len(&self) -> usize {
        self.prefix_len + self.input_len
    }
}

/// Walks one session's rounds in order, tracking how much context the replayed
/// conversation owns at each step.
///
/// Context grows by the whole round — prompt plus generated output — because a
/// server's prefix cache holds both. Constructed per session; never reused
/// across sessions, since cross-session prefix sharing is never fabricated.
#[derive(Debug, Default)]
pub struct ContextChain {
    context_len: usize,
}

impl ContextChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Context length available to the *next* round. Read only by the tests
    /// that pin how a chain grows; generation itself never inspects it.
    #[cfg(test)]
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    pub fn materialize(
        &mut self,
        raw: RawRound,
        policy: SessionContextPolicy,
    ) -> MaterializedRound {
        let round = match policy {
            SessionContextPolicy::TraceReported => self.materialize_trace_reported(raw),
            SessionContextPolicy::Monotonic => self.materialize_monotonic(raw),
        };
        self.context_len = round.prefix_len + round.input_len + round.output_len;
        round
    }

    /// Preserve the reported split, and turn an unavailable prefix into fresh
    /// work rather than into a cache hit that cannot happen.
    ///
    /// This is the common case on real traces, not a fallback: a coding agent
    /// usually resumes from a cached system prompt and prior context that the
    /// published trace does not contain, so a session's very first round
    /// typically reports a large prefix against an empty conversation.
    fn materialize_trace_reported(&self, raw: RawRound) -> MaterializedRound {
        let available_prefix = raw.prefix_len.min(self.context_len);
        let folded_tokens = raw.prefix_len - available_prefix;
        MaterializedRound {
            prefix_len: available_prefix,
            input_len: raw.input_len + folded_tokens,
            output_len: raw.output_len,
            folded_tokens,
            major_compaction: false,
        }
    }

    /// Preserve the longest prefix the replayed conversation can actually offer,
    /// bounded by the trace's total prompt length.
    ///
    /// A reduction that is not a major compaction is still the same
    /// conversation, so the prefix is truncated to the trace target rather than
    /// retained in full — the live prompt must never grow past the recorded
    /// workload shape.
    fn materialize_monotonic(&self, raw: RawRound) -> MaterializedRound {
        let target_prompt_len = raw.prefix_len + raw.input_len;
        let dropped = self.context_len.saturating_sub(target_prompt_len);
        let drop_ratio = if self.context_len == 0 {
            0.0
        } else {
            dropped as f64 / self.context_len as f64
        };
        let major_compaction = dropped >= MAJOR_COMPACTION_MIN_DROP_TOKENS
            && drop_ratio >= MAJOR_COMPACTION_MIN_DROP_RATIO;

        if major_compaction {
            return MaterializedRound {
                prefix_len: 0,
                input_len: target_prompt_len,
                output_len: raw.output_len,
                folded_tokens: raw.prefix_len,
                major_compaction: true,
            };
        }

        let prefix_len = self.context_len.min(target_prompt_len);
        MaterializedRound {
            prefix_len,
            input_len: target_prompt_len - prefix_len,
            output_len: raw.output_len,
            folded_tokens: raw.prefix_len.saturating_sub(prefix_len),
            major_compaction: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(prefix_len: usize, input_len: usize, output_len: usize) -> RawRound {
        RawRound {
            prefix_len,
            input_len,
            output_len,
        }
    }

    #[test]
    fn trace_reported_folds_an_unavailable_first_round_prefix_into_fresh_input() {
        let mut chain = ContextChain::new();

        let round = chain.materialize(raw(12_461, 5_875, 124), SessionContextPolicy::TraceReported);

        // Total prompt shape is preserved; none of it is claimed as cacheable.
        assert_eq!(round.prefix_len, 0);
        assert_eq!(round.input_len, 12_461 + 5_875);
        assert_eq!(round.folded_tokens, 12_461);
        assert_eq!(round.total_prompt_len(), 12_461 + 5_875);
        assert_eq!(chain.context_len(), 12_461 + 5_875 + 124);
    }

    #[test]
    fn trace_reported_keeps_the_reported_split_once_context_covers_it() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 512, 64), SessionContextPolicy::TraceReported);

        let round = chain.materialize(raw(576, 128, 64), SessionContextPolicy::TraceReported);

        assert_eq!(round.prefix_len, 576);
        assert_eq!(round.input_len, 128);
        assert_eq!(round.folded_tokens, 0);
    }

    #[test]
    fn trace_reported_folds_only_the_missing_part_of_a_partial_prefix() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 100, 10), SessionContextPolicy::TraceReported);
        assert_eq!(chain.context_len(), 110);

        let round = chain.materialize(raw(200, 50, 10), SessionContextPolicy::TraceReported);

        assert_eq!(round.prefix_len, 110);
        assert_eq!(round.input_len, 50 + 90);
        assert_eq!(round.folded_tokens, 90);
        assert_eq!(round.total_prompt_len(), 250);
    }

    #[test]
    fn monotonic_grows_the_prefix_to_the_trace_target() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 512, 64), SessionContextPolicy::Monotonic);
        assert_eq!(chain.context_len(), 576);

        let round = chain.materialize(raw(576, 128, 64), SessionContextPolicy::Monotonic);

        assert_eq!(round.prefix_len, 576);
        assert_eq!(round.input_len, 128);
        assert!(!round.major_compaction);
    }

    #[test]
    fn monotonic_reuses_more_than_the_trace_reported() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 1_000, 100), SessionContextPolicy::Monotonic);

        // The trace reports a cold-ish round, but the conversation has 1,100
        // tokens of real context and the target prompt is 900 tokens.
        let round = chain.materialize(raw(400, 500, 50), SessionContextPolicy::Monotonic);

        assert_eq!(round.prefix_len, 900);
        assert_eq!(round.input_len, 0);
        assert_eq!(round.total_prompt_len(), 900);
    }

    #[test]
    fn monotonic_truncates_a_small_reduction_without_resetting() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 9_000, 1_000), SessionContextPolicy::Monotonic);
        assert_eq!(chain.context_len(), 10_000);

        let round = chain.materialize(raw(8_500, 1_000, 100), SessionContextPolicy::Monotonic);

        assert_eq!(round.prefix_len, 9_500);
        assert_eq!(round.input_len, 0);
        assert!(!round.major_compaction);
    }

    #[test]
    fn monotonic_resets_only_when_both_thresholds_hold() {
        let mut chain = ContextChain::new();
        chain.materialize(raw(0, 190_000, 10_000), SessionContextPolicy::Monotonic);
        assert_eq!(chain.context_len(), 200_000);

        // 70,000 tokens dropped clears the absolute floor but is only 35% of the
        // context, so this is trimming, not compaction.
        let trimmed = chain.materialize(raw(130_000, 0, 0), SessionContextPolicy::Monotonic);
        assert!(!trimmed.major_compaction);
        assert_eq!(trimmed.prefix_len, 130_000);

        // Now 122,000 tokens go, which is 93% of the remaining context.
        let reset = chain.materialize(raw(8_000, 0, 0), SessionContextPolicy::Monotonic);
        assert!(reset.major_compaction);
        assert_eq!(reset.prefix_len, 0);
        assert_eq!(reset.input_len, 8_000);
    }

    #[test]
    fn both_policies_preserve_total_prompt_length() {
        for policy in [
            SessionContextPolicy::TraceReported,
            SessionContextPolicy::Monotonic,
        ] {
            let mut chain = ContextChain::new();
            for raw_round in [raw(9_000, 500, 64), raw(600, 128, 64), raw(0, 32, 8)] {
                let round = chain.materialize(raw_round, policy);
                assert_eq!(
                    round.total_prompt_len(),
                    raw_round.prefix_len + raw_round.input_len,
                    "{policy:?} changed the total prompt length"
                );
            }
        }
    }
}
