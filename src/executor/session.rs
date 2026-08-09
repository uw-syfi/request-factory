use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::backend::{context_limit_skip_result, GenerationResult};
use crate::cli::Args;
use tracelab_replay::policy::SessionContextPolicy;
use crate::executor::AppState;
use crate::record::StepLog;
use crate::tokens::{PromptBuild, PromptBuilder, TokenProvider};
use crate::trace::SessionStep;
use crate::util::reaches_context_limit;

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
        let PromptBuild {
            prompt_ids,
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
            folded_tokens,
            major_compaction,
        } = prompt_builder.build_prompt(&step, state.args.session_context_policy);
        let request_id = step.request_id.clone();
        state.stats.record_submit();
        let context_limit_skipped =
            should_skip_at_context_limit(&state.args, prompt_ids.len(), step.output_len);
        let result = if context_limit_skipped {
            context_limit_skip_result(
                request_id,
                prompt_ids.len(),
                step.output_len,
                state.args.max_model_len,
            )
        } else {
            state
                .client
                .run_step(request_id, &prompt_ids, step.output_len)
                .await
        };
        let GenerationResult {
            mut outcome,
            output_ids,
            output_ids_exact,
        } = result;
        let exact_context_failure = outcome.is_success()
            && state.args.session_context_policy == SessionContextPolicy::PrefixPreserving
            && !output_ids_exact;
        if exact_context_failure {
            outcome.status = "FAILED".to_string();
            outcome.error = Some(
                "monotonic session context requires exact generated token IDs, but the server response supplied none or a count inconsistent with completion_tokens"
                    .to_string(),
            );
        }
        let log = StepLog::session_round(
            &step,
            prompt_ids.len(),
            state.args.session_context_policy.label(),
            derived_prefix_len,
            derived_append_len,
            prefix_shortfall_len,
            folded_tokens,
            major_compaction,
            outcome,
        );
        let success = log.outcome.is_success();
        let _ = log_tx.send(log).await;

        state.stats.record_result(success);
        if context_limit_skipped
            || exact_context_failure
            || (!success && state.args.stop_session_on_error)
        {
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

fn should_skip_at_context_limit(args: &Args, prompt_len: usize, output_len_target: usize) -> bool {
    args.skip_when_reaching_limit
        && args
            .max_model_len
            .is_some_and(|limit| reaches_context_limit(prompt_len, output_len_target, limit))
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
