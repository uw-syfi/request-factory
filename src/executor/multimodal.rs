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
use crate::schema::{CapabilityProfile, InputPart, MediaSource, Modality, OutputSpec, RequestSpec};
use crate::synthetic::SyntheticStore;
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
    dialect: &crate::backend::Dialect,
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
        BackendKind::OpenaiImageEdits => (
            [Modality::Text, Modality::Image].into_iter().collect(),
            [Modality::Image].into_iter().collect(),
        ),
        BackendKind::OpenaiVideos => (
            [Modality::Text, Modality::Image, Modality::Video]
                .into_iter()
                .collect(),
            [Modality::Video].into_iter().collect(),
        ),
        BackendKind::OpenaiTranscriptions | BackendKind::OpenaiTranslations => (
            [Modality::Text, Modality::Audio].into_iter().collect(),
            [Modality::Text].into_iter().collect(),
        ),
        _ => bail!("backend {backend:?} does not support multimodal-independent-v1"),
    };
    // The surface decides what the model can do; the dialect decides what can be
    // said to it. A trace needs both, and needs to be told so before the run
    // starts rather than one failed request at a time.
    let accepted_inputs = accepted_inputs
        .intersection(&dialect.accepted_input_modalities())
        .copied()
        .collect();
    let capabilities = CapabilityProfile {
        accepted_inputs,
        produced_outputs,
        supports_mixed_inputs: true,
        supports_multiple_outputs: false,
    };
    let store = AssetStore::new(artifact_path)?;
    let synthetic = SyntheticStore::default();
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
                InputPart::Image { source }
                | InputPart::Audio { source }
                | InputPart::Video { source } => {
                    let loaded = match source {
                        MediaSource::Asset(asset) => store.load(asset).with_context(|| {
                            format!(
                                "prepare asset {:?} for request {:?}",
                                asset.path, request.id
                            )
                        })?,
                        // Generated here, alongside asset loading: both finish
                        // before the replay clock starts, so neither shows up in
                        // a measured latency.
                        MediaSource::Synthetic(spec) => synthetic
                            .build(spec, input.modality(), &request.id)
                            .with_context(|| {
                                format!("generate synthetic input for request {:?}", request.id)
                            })?,
                    };
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
    request_ordinal: usize,
    request: PreparedMultimodalRequest,
) {
    let arrival_release_lag_ms =
        wait_for_common_arrival(&state.common, request.source.arrival_time_ms).await;
    let _concurrency_permit = state.common.acquire_capacity_slot(request_ordinal).await;
    state.common.stats.record_submit();
    // Text output does not imply the token-streaming client: transcription and
    // translation also answer with text, but over a one-shot multipart surface
    // with no token stream to fold. The surface decides the client, not the
    // output modality alone.
    let result = match (&request.output, &state.text_client) {
        (OutputSpec::Text { max_tokens }, Some(text_client)) => {
            text_client
                .run_step(
                    request.source.id.clone(),
                    Prompt::Parts(&request.parts),
                    *max_tokens,
                )
                .await
        }
        (OutputSpec::Tensor { .. }, _) => {
            unreachable!("capability validation rejects tensor output")
        }
        _ => {
            state
                .media_client
                .as_ref()
                .expect("validated media request has a media client")
                .run_step(request.source.id.clone(), &request.parts, &request.output)
                .await
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
