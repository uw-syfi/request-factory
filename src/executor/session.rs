use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::backend::{context_overflow_result, GenerationResult};
use crate::cli::Args;
use crate::executor::AppState;
use crate::record::StepLog;
use crate::tokens::{PromptBuilder, TokenProvider};
use crate::trace::SessionStep;

/// Replay one session as an ordered, closed-loop chain of rounds.
pub(crate) async fn run_session(
    state: Arc<AppState>,
    log_tx: mpsc::Sender<StepLog>,
    session_ordinal: usize,
    session_id: String,
    steps: Vec<SessionStep>,
) {
    wait_for_session_arrival(&state, &steps).await;
    let _concurrency_permit = match &state.concurrency_semaphore {
        Some(semaphore) => semaphore.clone().acquire_owned().await.ok(),
        None => None,
    };

    let token_provider = match TokenProvider::new(
        state.token_pool.clone(),
        session_ordinal.wrapping_mul(9_973),
    ) {
        Ok(provider) => provider,
        Err(err) => {
            eprintln!("session {session_id}: {err}");
            state.stats.record_unit_done();
            return;
        }
    };
    let mut prompt_builder = PromptBuilder::new(token_provider);

    for step in steps {
        let prompt_ids = prompt_builder.build_prompt(&step);
        let request_id = format!("{}_round_{:06}", session_id, step.round_idx);
        state.stats.record_submit();
        let result = if should_skip_context_overflow(&state.args, prompt_ids.len()) {
            context_overflow_result(request_id, prompt_ids.len(), state.args.max_model_len)
        } else {
            state
                .client
                .run_step(request_id, &prompt_ids, step.output_len)
                .await
        };
        let GenerationResult {
            outcome,
            output_ids,
        } = result;
        let log = StepLog::session_round(&step, prompt_ids.len(), outcome);
        let success = log.outcome.is_success();
        let _ = log_tx.send(log).await;

        state.stats.record_result(success);
        if !success && state.args.stop_session_on_error {
            break;
        }

        // Carry the model's real output tokens forward (not synthetic) so the previous-output
        // region of the next prefix matches what the server cached and stays cache-hittable.
        prompt_builder.commit_output(prompt_ids, output_ids);

        if step.tool_wait_after_ms > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(step.tool_wait_after_ms / 1000.0)).await;
        }
    }

    state.stats.record_unit_done();
}

fn should_skip_context_overflow(args: &Args, prompt_len: usize) -> bool {
    args.fail_on_context_overflow
        && args
            .max_model_len
            .map(|limit| prompt_len > limit)
            .unwrap_or(false)
}

async fn wait_for_session_arrival(state: &AppState, steps: &[SessionStep]) {
    let arrival_ms = steps
        .first()
        .map(|step| step.arrival_time.max(0.0))
        .unwrap_or(0.0);
    if arrival_ms <= 0.0 {
        return;
    }

    let target = state.run_start + Duration::from_secs_f64(arrival_ms / 1000.0);
    let now = Instant::now();
    if target > now {
        tokio::time::sleep_until(tokio::time::Instant::from_std(target)).await;
    }
}
