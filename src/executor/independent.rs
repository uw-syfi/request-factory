use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::backend::context_overflow_result;
use crate::executor::AppState;
use crate::record::StepLog;
use crate::tokens::TokenProvider;
use crate::trace::VibeSimRequest;

/// Replay one VibeSim request. This is deliberately separate from the session
/// executor: independent requests have no round ordering, prefix carry-forward,
/// or tool wait semantics.
pub(crate) async fn run_independent_request(
    state: Arc<AppState>,
    log_tx: mpsc::Sender<StepLog>,
    request_ordinal: usize,
    request: VibeSimRequest,
) {
    wait_for_arrival(&state, request.arrival_time).await;
    let _concurrency_permit = match &state.concurrency_semaphore {
        Some(semaphore) => semaphore.clone().acquire_owned().await.ok(),
        None => None,
    };
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
    let request_id = format!("vibesim_{}", request.id);
    state.stats.record_submit();
    let result = if state.args.fail_on_context_overflow
        && state
            .args
            .max_model_len
            .is_some_and(|limit| prompt_ids.len() > limit)
    {
        context_overflow_result(request_id, prompt_ids.len(), state.args.max_model_len)
    } else {
        state
            .client
            .run_step(request_id, &prompt_ids, request.output_len)
            .await
    };
    let log = StepLog::vibesim_request(&request, prompt_ids.len(), result.outcome);
    let success = log.outcome.is_success();
    let _ = log_tx.send(log).await;
    state.stats.record_result(success);
    state.stats.record_unit_done();
}

async fn wait_for_arrival(state: &AppState, arrival_time_ms: f64) {
    let arrival_time_ms = arrival_time_ms.max(0.0);
    if arrival_time_ms <= 0.0 {
        return;
    }
    let target = state.run_start + Duration::from_secs_f64(arrival_time_ms / 1000.0);
    let now = Instant::now();
    if target > now {
        tokio::time::sleep_until(tokio::time::Instant::from_std(target)).await;
    }
}
