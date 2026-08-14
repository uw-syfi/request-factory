use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::backend::{context_limit_skip_result, GenerationResult, Prompt};
use crate::executor::AppState;
use crate::record::StepLog;
use crate::release::ArrivalMode;
use crate::timeline::{RequestTimeline, TimelineSink};
use crate::tokens::{PromptBuild, PromptBuilder, TokenProvider};
use crate::trace::SessionStep;

/// Replay one session as an ordered, closed-loop chain of rounds.
pub(crate) async fn run_session(
    state: Arc<AppState>,
    log_tx: mpsc::Sender<StepLog>,
    // Travels with the task rather than living on `AppState`, for the same
    // reason `log_tx` does: the run closes the writer's channel by dropping
    // every sender, and a sender parked inside shared state is one nobody can
    // drop on time.
    timeline_sink: Option<TimelineSink>,
    session_ordinal: usize,
    session_id: String,
    steps: Vec<SessionStep>,
) {
    wait_for_session_arrival(&state, &steps).await;
    // Bound to this scope on purpose: the session owns its slot for every round
    // and every tool wait below, and gives it up only when the whole
    // conversation ends. That is the contract VibeSim's session ledger mirrors.
    let _concurrency_permit = state.acquire_capacity_slot(session_ordinal).await;

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
        let PromptBuild {
            prompt_ids,
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
        } = prompt_builder.build_prompt(&step);
        let prompt = Prompt::Tokens(&prompt_ids);
        let request_id = step.request_id.clone();
        state.stats.record_submit();
        let context_limit_skipped = state
            .policy
            .skips_at_context_limit(prompt.token_len(), step.output_len);
        let result = if context_limit_skipped {
            context_limit_skip_result(
                request_id,
                prompt.token_len(),
                step.output_len,
                state.policy.max_model_len(),
            )
        } else {
            state
                .client
                .run_step(request_id, prompt, step.output_len)
                .await
        };
        let GenerationResult {
            outcome,
            output_ids,
            timeline,
        } = result;
        if let Some(sink) = &timeline_sink {
            sink.offer(RequestTimeline {
                request_id: outcome.request_id.clone(),
                events: timeline,
            });
        }
        let log = StepLog::session_round(
            &step,
            prompt.token_len(),
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
            outcome,
        );
        let success = log.outcome.is_success();
        let _ = log_tx.send(log).await;

        state.stats.record_result(success);
        if context_limit_skipped || (!success && state.policy.stop_session_on_error) {
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

async fn wait_for_session_arrival(state: &AppState, steps: &[SessionStep]) {
    if state.policy.arrival_mode == ArrivalMode::Saturated {
        return;
    }
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
