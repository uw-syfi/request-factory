use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use tokio::sync::mpsc;

use crate::assets::AssetStore;
use crate::backend::{GenerationClient, PreparedInputPart, Prompt};
use crate::executor::independent::wait_for_common_arrival;
use crate::executor::CommonState;
use crate::record::StepLog;
use crate::schema::{CapabilityProfile, InputPart, Modality, OutputSpec, RequestSpec};
use crate::timeline::{RequestTimeline, TimelineSink};

pub(crate) struct MultimodalState {
    pub(crate) common: CommonState,
    pub(crate) client: Arc<GenerationClient>,
}

pub(crate) struct PreparedMultimodalRequest {
    source: RequestSpec,
    parts: Vec<PreparedInputPart>,
    max_tokens: usize,
    asset_bytes: usize,
}

pub(crate) fn prepare_multimodal_requests(
    artifact_path: &str,
    requests: Vec<RequestSpec>,
) -> Result<Vec<PreparedMultimodalRequest>> {
    let capabilities = CapabilityProfile {
        accepted_inputs: [
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        produced_outputs: [Modality::Text].into_iter().collect(),
        supports_mixed_inputs: true,
        supports_multiple_outputs: false,
    };
    let store = AssetStore::new(artifact_path)?;
    let mut prepared = Vec::with_capacity(requests.len());
    for request in requests {
        capabilities
            .validate(&request)
            .with_context(|| format!("request {:?} is not supported by openai-chat", request.id))?;
        let max_tokens = match request.outputs.as_slice() {
            [OutputSpec::Text { max_tokens }] => *max_tokens,
            _ => bail!(
                "request {:?}: openai-chat requires one text output",
                request.id
            ),
        };
        let mut asset_bytes = 0usize;
        let mut parts = Vec::with_capacity(request.inputs.len());
        for input in &request.inputs {
            match input {
                InputPart::Text { text } => parts.push(PreparedInputPart::Text(text.clone())),
                InputPart::Image { asset }
                | InputPart::Audio { asset }
                | InputPart::Video { asset } => {
                    let loaded = store.load(asset).with_context(|| {
                        format!(
                            "prepare asset {:?} for request {:?}",
                            asset.path, request.id
                        )
                    })?;
                    asset_bytes = asset_bytes.saturating_add(loaded.bytes.len());
                    parts.push(PreparedInputPart::Media {
                        modality: input.modality(),
                        // Encode before the replay clock starts; CPU-side media
                        // reads, hashing, and base64 must not inflate TTFT.
                        data_url: loaded.data_url(),
                    });
                }
                InputPart::Tensor { .. } => {
                    bail!(
                        "request {:?}: openai-chat does not accept tensor input",
                        request.id
                    )
                }
            }
        }
        prepared.push(PreparedMultimodalRequest {
            source: request,
            parts,
            max_tokens,
            asset_bytes,
        });
    }
    Ok(prepared)
}

pub(crate) async fn run_multimodal_request(
    state: Arc<MultimodalState>,
    log_tx: mpsc::Sender<StepLog>,
    timeline_sink: Option<TimelineSink>,
    request_ordinal: usize,
    request: PreparedMultimodalRequest,
) {
    let arrival_release_lag_ms =
        wait_for_common_arrival(&state.common, request.source.arrival_time_ms).await;
    let _concurrency_permit = state.common.acquire_capacity_slot(request_ordinal).await;
    state.common.stats.record_submit();
    let result = state
        .client
        .run_step(
            request.source.id.clone(),
            Prompt::Parts(&request.parts),
            request.max_tokens,
        )
        .await;
    if let Some(sink) = &timeline_sink {
        sink.offer(RequestTimeline {
            request_id: result.outcome.request_id.clone(),
            events: result.timeline,
        });
    }
    let log = StepLog::multimodal_request(
        &request.source,
        request.asset_bytes,
        request.max_tokens,
        arrival_release_lag_ms,
        result.outcome,
    );
    let success = log.outcome.is_success();
    let _ = log_tx.send(log).await;
    state.common.stats.record_result(success);
    state.common.stats.record_unit_done();
}
