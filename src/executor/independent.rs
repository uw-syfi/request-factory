use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::backend::{context_limit_skip_result, Prompt};
use crate::cli::ArrivalMode;
use crate::executor::AppState;
use crate::record::StepLog;
use crate::timeline::{RequestTimeline, TimelineSink};
use crate::tokens::TokenProvider;
use crate::trace::IndependentRequest;

/// Replay one independent request. This is deliberately separate from the session
/// executor: independent requests have no round ordering, prefix carry-forward,
/// or tool wait semantics.
pub(crate) async fn run_independent_request(
    state: Arc<AppState>,
    log_tx: mpsc::Sender<StepLog>,
    // Travels with the task rather than living on `AppState`; see `run_session`.
    timeline_sink: Option<TimelineSink>,
    request_ordinal: usize,
    request: IndependentRequest,
) {
    let arrival_release_lag_ms = wait_for_arrival(&state, request.arrival_time).await;
    let _concurrency_permit = state.acquire_capacity_slot(request_ordinal).await;
    let mut token_provider = match TokenProvider::new(
        state.token_pool.clone(),
        request_ordinal.wrapping_mul(9_973),
    ) {
        Ok(provider) => provider,
        Err(err) => {
            eprintln!("request {}: {err}", request.id);
            state.stats.record_unit_done();
            return;
        }
    };
    let prompt_ids = token_provider.take(request.input_len);
    let prompt = Prompt::Tokens(&prompt_ids);
    let request_id = format!("independent_{}", request.id);
    state.stats.record_submit();
    let result = if state
        .policy
        .skips_at_context_limit(prompt.token_len(), request.output_len)
    {
        context_limit_skip_result(
            request_id,
            prompt.token_len(),
            request.output_len,
            state.policy.max_model_len(),
        )
    } else {
        state
            .client
            .run_step(request_id, prompt, request.output_len)
            .await
    };
    if let Some(sink) = &timeline_sink {
        sink.offer(RequestTimeline {
            request_id: result.outcome.request_id.clone(),
            events: result.timeline,
        });
    }
    let log = StepLog::independent_request(
        &request,
        prompt.token_len(),
        arrival_release_lag_ms,
        result.outcome,
    );
    let success = log.outcome.is_success();
    let _ = log_tx.send(log).await;
    state.stats.record_result(success);
    state.stats.record_unit_done();
}

async fn wait_for_arrival(state: &AppState, arrival_time_ms: f64) -> f64 {
    if state.policy.arrival_mode == ArrivalMode::Saturated {
        return 0.0;
    }
    let arrival_time_ms = arrival_time_ms.max(0.0);
    let target = state.run_start + Duration::from_secs_f64(arrival_time_ms / 1000.0);
    let now = Instant::now();
    if target > now {
        tokio::time::sleep_until(tokio::time::Instant::from_std(target)).await;
    }
    Instant::now()
        .saturating_duration_since(target)
        .as_secs_f64()
        * 1000.0
}
