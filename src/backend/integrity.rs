//! Checks that decide whether a streamed response can be believed.
//!
//! These are policy, not parsing: each one names a specific way a server can
//! hand back something that looks fine and would silently corrupt a measurement.

/// Whether a streamed chunk restates the whole output so far instead of
/// carrying only new tokens.
///
/// Every supported backend is expected to stream disjoint deltas. Folding a
/// cumulative chunk as if it were a delta would multiply the output and wreck
/// the TPOT denominator, so the shared engine treats this as a hard failure.
/// The first chunk is exempt because an empty accumulator prefixes anything.
pub(super) fn restates_accumulated_output(accumulated: &[u32], incoming: &[u32]) -> bool {
    !accumulated.is_empty()
        && incoming.len() > accumulated.len()
        && incoming.starts_with(accumulated)
}

/// Verdict on generated token ids that outnumber the server's completion count.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PromptEcho {
    /// The counts agree; nothing was echoed.
    None,
    /// The leading `n` ids provably repeat the prompt's tail and can be dropped.
    Leading(usize),
    /// There are more ids than completion tokens, but the excess does not match
    /// the prompt tail, so what it represents is unknown.
    Unexplained,
}

/// Classify a leading prompt echo in the collected generated token ids.
///
/// SGLang has been reported to prepend a suffix of `input_ids` to `output_ids`
/// (sgl-project/sglang#10896). That was filed against the offline Engine API
/// rather than this streaming HTTP path, so the guard is cheap insurance:
/// it trims only what it can prove came from the prompt, and refuses to guess
/// otherwise. Carrying an echoed prefix forward would corrupt the next round's
/// context and every prefix-cache number derived from it.
pub(super) fn classify_prompt_echo(
    output_ids: &[u32],
    prompt_ids: &[u32],
    completion_tokens: usize,
) -> PromptEcho {
    let Some(echoed) = output_ids.len().checked_sub(completion_tokens) else {
        return PromptEcho::None;
    };
    if echoed == 0 {
        return PromptEcho::None;
    }
    let matches_prompt_tail = prompt_ids
        .len()
        .checked_sub(echoed)
        .is_some_and(|start| prompt_ids[start..] == output_ids[..echoed]);
    if matches_prompt_tail {
        PromptEcho::Leading(echoed)
    } else {
        PromptEcho::Unexplained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_streaming_is_detected_but_real_deltas_are_not() {
        // A cumulative chunk repeats everything delivered so far and then adds.
        assert!(restates_accumulated_output(&[1, 2, 3], &[1, 2, 3, 4]));
        // A disjoint delta does not, even when it happens to start with the
        // same id the accumulator did.
        assert!(!restates_accumulated_output(&[1, 2, 3], &[4, 5]));
        assert!(!restates_accumulated_output(&[1, 2, 3], &[1, 9]));
        // Re-sending the identical array without growing is not the cumulative
        // pattern this guard is for, and must not be misread as one.
        assert!(!restates_accumulated_output(&[1, 2, 3], &[1, 2, 3]));
        // The first chunk is exempt: an empty accumulator prefixes anything.
        assert!(!restates_accumulated_output(&[], &[1, 2, 3]));
    }

    #[test]
    fn prompt_echo_is_trimmed_only_when_it_matches_the_prompt_tail() {
        let prompt_ids = [10, 11, 12, 13];

        // Counts agree: nothing echoed.
        assert_eq!(
            classify_prompt_echo(&[90, 91], &prompt_ids, 2),
            PromptEcho::None
        );
        // Two extra leading ids that are exactly the prompt's last two tokens.
        assert_eq!(
            classify_prompt_echo(&[12, 13, 90, 91], &prompt_ids, 2),
            PromptEcho::Leading(2)
        );
        // Extra ids that are not the prompt tail: meaning unknown, never guess.
        assert_eq!(
            classify_prompt_echo(&[77, 78, 90, 91], &prompt_ids, 2),
            PromptEcho::Unexplained
        );
        // More excess than the prompt has tokens cannot be a prompt echo.
        assert_eq!(
            classify_prompt_echo(&[1, 2, 3, 4, 5, 90], &[10, 11], 1),
            PromptEcho::Unexplained
        );
        // Fewer ids than completion tokens is a different problem, and is left
        // to the existing output_ids_exact check rather than handled here.
        assert_eq!(
            classify_prompt_echo(&[90], &prompt_ids, 4),
            PromptEcho::None
        );
    }
}
