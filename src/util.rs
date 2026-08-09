use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Planned prefix-cache hit rate: prior-context tokens over total prompt tokens.
pub(crate) fn prefix_hit_rate(prefix_tokens: usize, prompt_tokens: usize) -> f64 {
    ratio(prefix_tokens, prompt_tokens).unwrap_or(0.0)
}

/// Safe ratio that returns `None` instead of dividing by zero.
pub(crate) fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

/// Whether a request would consume the model's full context budget.
///
/// Equality skips deliberately: a submitted request must leave at least one
/// token below `max_model_len` rather than relying on a boundary-exact server
/// interpretation.
pub(crate) fn reaches_context_limit(
    prompt_len: usize,
    output_len_target: usize,
    max_model_len: usize,
) -> bool {
    prompt_len.saturating_add(output_len_target) >= max_model_len
}

pub(crate) fn unix_seconds_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub(crate) fn elapsed_ms(start: Instant) -> f64 {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (ms * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_limit_guard_reserves_one_token_of_headroom() {
        assert!(!reaches_context_limit(90, 9, 100));
        assert!(reaches_context_limit(90, 10, 100));
        assert!(reaches_context_limit(90, 11, 100));
    }
}
