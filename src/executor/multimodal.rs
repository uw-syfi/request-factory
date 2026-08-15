use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use tokio::sync::mpsc;

use crate::assets::AssetStore;
use crate::backend::{GenerationClient, MediaClient, PreparedInputPart, Prompt};
use crate::cli::BackendKind;
use crate::executor::independent::wait_for_common_arrival;
use crate::executor::CommonState;
use crate::record::StepLog;
use crate::schema::{CapabilityProfile, InputPart, Modality, OutputSpec, RequestSpec};
use crate::timeline::{RequestTimeline, TimelineSink};

pub(crate) struct MultimodalState {
    pub(crate) common: CommonState,
    pub(crate) text_client: Option<Arc<GenerationClient>>,
    pub(crate) media_client: Option<Arc<MediaClient>>,
}

pub(crate) struct PreparedMultimodalRequest {
    source: RequestSpec,
    parts: Vec<PreparedInputPart>,
    output: OutputSpec,
    asset_bytes: usize,
}

pub(crate) fn prepare_multimodal_requests(
    artifact_path: &str,
    backend: BackendKind,
    requests: Vec<RequestSpec>,
) -> Result<Vec<PreparedMultimodalRequest>> {
    let (accepted_inputs, produced_outputs) = match backend {
        BackendKind::OpenaiChat => (
            [
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
            ]
            .into_iter()
            .collect::<BTreeSet<_>>(),
            [Modality::Text, Modality::Image, Modality::Audio]
                .into_iter()
                .collect(),
        ),
        BackendKind::OpenaiImages => (
            [Modality::Text].into_iter().collect(),
            [Modality::Image].into_iter().collect(),
        ),
        BackendKind::OpenaiSpeech => (
            [Modality::Text].into_iter().collect(),
            [Modality::Audio].into_iter().collect(),
        ),
        _ => bail!("backend {backend:?} does not support multimodal-independent-v1"),
    };
    let capabilities = CapabilityProfile {
        accepted_inputs,
        produced_outputs,
        supports_mixed_inputs: true,
        supports_multiple_outputs: false,
    };
    let store = AssetStore::new(artifact_path)?;
    let mut prepared = Vec::with_capacity(requests.len());
    for request in requests {
        capabilities
            .validate(&request)
            .with_context(|| format!("request {:?} is not supported by {backend:?}", request.id))?;
        let [output] = request.outputs.as_slice() else {
            bail!(
                "request {:?}: backend requires exactly one output",
                request.id
            )
        };
        let output = output.clone();
        let mut asset_bytes = 0usize;
        let mut parts = Vec::with_capacity(request.inputs.len());
        for input in &request.inputs {
            match input {
                InputPart::System { text } => parts.push(PreparedInputPart::System(text.clone())),
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
            output,
            asset_bytes,
        });
    }
    Ok(prepared)
}

pub(crate) async fn run_multimodal_request(
    state: Arc<MultimodalState>,
    log_tx: mpsc::Sender<StepLog>,
    timeline_sink: Option<TimelineSink>,
    _request_ordinal: usize,
    request: PreparedMultimodalRequest,
) {
    let arrival_release_lag_ms =
        wait_for_common_arrival(&state.common, request.source.arrival_time_ms).await;
    state.common.stats.record_submit();
    let result = match &request.output {
        OutputSpec::Text { max_tokens } => {
            state
                .text_client
                .as_ref()
                .expect("validated text request has a text client")
                .run_step(
                    request.source.id.clone(),
                    Prompt::Parts(&request.parts),
                    *max_tokens,
                )
                .await
        }
        OutputSpec::Image { .. } | OutputSpec::Audio { .. } => {
            state
                .media_client
                .as_ref()
                .expect("validated media request has a media client")
                .run_step(request.source.id.clone(), &request.parts, &request.output)
                .await
        }
        OutputSpec::Video { .. } | OutputSpec::Tensor { .. } => {
            unreachable!("capability validation rejects unsupported generated media")
        }
    };
    if let Some(sink) = &timeline_sink {
        sink.offer(RequestTimeline {
            request_id: result.outcome.request_id.clone(),
            events: result.timeline,
        });
    }
    let log = StepLog::multimodal_request(
        &request.source,
        request.asset_bytes,
        match request.output {
            OutputSpec::Text { max_tokens } => max_tokens,
            _ => 0,
        },
        arrival_release_lag_ms,
        result.outcome,
    );
    let success = log.outcome.is_success();
    let _ = log_tx.send(log).await;
    state.common.stats.record_result(success);
    state.common.stats.record_unit_done();
}
