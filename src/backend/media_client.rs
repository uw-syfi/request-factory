//! Generated-media transport and measurement.
//!
//! Text generation retains its token-aware SSE engine. This client handles the
//! orthogonal response shapes used by current multimodal servers: JSON-carried
//! image/audio data and raw PCM audio streams. Both produce the same
//! modality-neutral outcome fields and timeline events.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use bytes::BytesMut;
use futures::StreamExt;
use serde_json::Value;
use tokio::time::timeout;

use crate::cli::{Args, BackendKind};
use crate::record::GenerationOutcome;
use crate::schema::{Modality, OutputSpec};
use crate::timeline::{EventKind, TimelineEvent};
use crate::util::{elapsed_ms, unix_seconds_now};

use super::dialect::{AudioDeltas, MediaRead, VoiceSlot};
use super::{dialect_for, sse_data_line, ChatInputs, Dialect, GenerationResult, PreparedInputPart};

pub(crate) struct MediaClient {
    endpoint: String,
    client: reqwest::Client,
    model: String,
    backend: BackendKind,
    dialect: &'static Dialect,
    stream_idle_timeout_secs: u64,
    record_timeline: bool,
    artifact_dir: Option<PathBuf>,
    temperature: f64,
}

/// One request as this dialect renders it.
///
/// Two of the seven surfaces are `multipart/form-data` rather than JSON, so the
/// body and the upload are separate: the fields stay inspectable (and unit
/// testable) instead of disappearing into a `Form` that cannot be read back.
#[derive(Debug)]
struct Rendered {
    body: Value,
    file: Option<FilePart>,
    /// Whether the body goes out as `multipart/form-data`.
    ///
    /// Declared rather than inferred from `file.is_some()`: a text-only video
    /// request carries no upload and still has to be a form, and a live
    /// vLLM-Omni rejects the JSON version with "Field required: prompt".
    multipart: bool,
}

#[derive(Debug)]
struct FilePart {
    field: &'static str,
    filename: String,
    mime: String,
    bytes: Vec<u8>,
}

impl Rendered {
    fn json(body: Value) -> Self {
        Self {
            body,
            file: None,
            multipart: false,
        }
    }

    fn form(body: Value, file: Option<FilePart>) -> Self {
        Self {
            body,
            file,
            multipart: true,
        }
    }
}

pub(super) struct MediaFold {
    pub(super) modality: Modality,
    pub(super) bytes: Vec<u8>,
    pub(super) first_output_ms: Option<f64>,
    pub(super) last_output_ms: Option<f64>,
    pub(super) output_chunk_count: usize,
    pub(super) duration_ms: f64,
    pub(super) timeline: Vec<TimelineEvent>,
    pub(super) error: Option<String>,
    /// The server answered in the shape this surface expects, whether or not
    /// that answer carried any bytes. Transcribing silence yields
    /// `{"text": ""}` -- a correct result, and one that must not be counted as
    /// a failed request.
    pub(super) answered: bool,
}

impl MediaFold {
    pub(super) fn new(modality: Modality, record_timeline: bool) -> Self {
        Self {
            modality,
            bytes: Vec::new(),
            first_output_ms: None,
            last_output_ms: None,
            output_chunk_count: 0,
            duration_ms: 0.0,
            timeline: if record_timeline {
                Vec::with_capacity(32)
            } else {
                Vec::new()
            },
            error: None,
            answered: false,
        }
    }

    pub(super) fn absorb(&mut self, payload: &[u8], at_ms: f64, duration_ms: Option<f64>) {
        if payload.is_empty() {
            return;
        }
        self.first_output_ms.get_or_insert(at_ms);
        self.last_output_ms = Some(at_ms);
        self.output_chunk_count += 1;
        self.duration_ms += duration_ms.unwrap_or(0.0);
        self.bytes.extend_from_slice(payload);
        if self.timeline.capacity() > 0 {
            self.timeline.push(TimelineEvent {
                elapsed_ms: at_ms as f32,
                kind: EventKind::Media,
                tokens: 0,
                cumulative_tokens: 0,
                bytes: u32::try_from(payload.len()).unwrap_or(u32::MAX),
                cumulative_bytes: u64::try_from(self.bytes.len()).unwrap_or(u64::MAX),
            });
        }
    }

    pub(super) fn fail(&mut self, error: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(error.into());
        }
    }
}

impl MediaClient {
    pub(crate) fn new(args: &Args) -> Result<Self> {
        // The backend picks which surface to talk to; the dialect decides what
        // that surface is called and how to speak it. Keeping the two apart is
        // what lets a new server be a table entry instead of a new backend.
        let dialect = dialect_for(&args.dialect)?;
        let suffix = dialect.surface(args.backend)?;
        let artifact_dir = args.output_artifact_dir.as_deref().map(PathBuf::from);
        if let Some(directory) = &artifact_dir {
            std::fs::create_dir_all(directory).with_context(|| {
                format!(
                    "failed to create output artifact directory {}",
                    directory.display()
                )
            })?;
        }
        Ok(Self {
            endpoint: format!("{}{}", args.base_url.trim_end_matches('/'), suffix),
            client: reqwest::Client::builder()
                .pool_max_idle_per_host(20_000)
                .tcp_nodelay(true)
                .timeout(Duration::from_secs(3600))
                .build()?,
            model: args.model.clone(),
            backend: args.backend,
            dialect,
            stream_idle_timeout_secs: args.stream_idle_timeout_secs,
            record_timeline: args.timeline,
            artifact_dir,
            temperature: args.temperature,
        })
    }

    pub(crate) async fn run_step(
        &self,
        request_id: String,
        parts: &[PreparedInputPart],
        output: &OutputSpec,
    ) -> GenerationResult {
        let submit_timestamp = unix_seconds_now();
        let start = Instant::now();
        let payload = match self.build_payload(parts, output) {
            Ok(payload) => payload,
            Err(error) => {
                return media_failure(request_id, output.modality(), submit_timestamp, error)
            }
        };
        let post_timestamp = Some(unix_seconds_now());
        let send_instant = Instant::now();
        let mut fold = MediaFold::new(output.modality(), self.record_timeline);
        self.fold_response(&request_id, &payload, output, send_instant, &mut fold)
            .await;
        let response_complete_ms_value = elapsed_ms(send_instant);
        let response_complete_ms = Some(response_complete_ms_value);
        let measured_duration_ms = (fold.duration_ms > 0.0).then_some(fold.duration_ms);
        let real_time_factor = measured_duration_ms
            .filter(|duration| *duration > 0.0)
            .map(|duration| response_complete_ms_value / duration);
        let complete_timestamp = unix_seconds_now();
        let total_duration_ms = elapsed_ms(start);
        let mut status = if fold.error.is_none() && (fold.answered || !fold.bytes.is_empty()) {
            "SUCCESS".to_string()
        } else {
            if fold.error.is_none() {
                let what = match fold.modality {
                    Modality::Text => "transcript".to_string(),
                    other => format!("{other:?}").to_lowercase(),
                };
                fold.fail(format!("server returned no {what} in its response"));
            }
            "FAILED".to_string()
        };
        // The response clock stops above. Artifact I/O is deliberately outside
        // request latency and cannot alter first-output or completion metrics.
        let artifact_path = if status == "SUCCESS" {
            match self.persist(&request_id, output, &fold.bytes).await {
                Ok(path) => path,
                Err(error) => {
                    fold.fail(format!("artifact persistence failed: {error:#}"));
                    status = "FAILED".into();
                    None
                }
            }
        } else {
            None
        };
        GenerationResult {
            outcome: GenerationOutcome {
                request_id,
                output_modality: fold.modality,
                output_len_actual: 0,
                output_len_text_tokens: 0,
                echoed_prompt_tokens: 0,
                server_usage: None,
                finish_reason: None,
                submit_timestamp,
                post_timestamp,
                complete_timestamp,
                first_output_ms: fold.first_output_ms,
                last_output_ms: fold.last_output_ms,
                output_bytes: fold.bytes.len(),
                output_chunk_count: fold.output_chunk_count,
                output_duration_ms: measured_duration_ms,
                real_time_factor,
                artifact_path,
                first_token_ms: None,
                first_token_id_ms: None,
                last_token_id_ms: None,
                first_token_event_tokens: 0,
                token_event_count: 0,
                usage_event_count: 0,
                token_delivery_tpot_ms: None,
                response_complete_ms,
                terminal_tail_ms: match (fold.last_output_ms, response_complete_ms) {
                    (Some(last), Some(done)) if done >= last => Some(done - last),
                    _ => None,
                },
                total_duration_ms,
                chunk_count: fold.output_chunk_count,
                status,
                output_preview: String::new(),
                error: fold.error,
            },
            output_ids: Vec::new(),
            timeline: fold.timeline,
        }
    }

    fn build_payload(&self, parts: &[PreparedInputPart], output: &OutputSpec) -> Result<Rendered> {
        let dialect = self.dialect;
        match (self.backend, output) {
            (BackendKind::OpenaiImages, OutputSpec::Image { .. }) => {
                let mut payload = serde_json::json!({
                    "model": self.model,
                    "prompt": text_prompt(parts)?,
                    "response_format": "b64_json",
                });
                self.put_image_controls(&mut payload, output)?;
                Ok(Rendered::json(payload))
            }
            (BackendKind::OpenaiChat, OutputSpec::Image { .. }) => {
                dialect.check_chat_image_generation()?;
                let mut payload =
                    Value::Object(ChatInputs::plan(parts, dialect.media_input)?.to_object()?);
                payload["model"] = self.model.clone().into();
                payload["stream"] = false.into();
                if let Some(modalities) = dialect.image_modalities {
                    payload["modalities"] = serde_json::json!(modalities);
                }
                self.put_image_controls(&mut payload, output)?;
                Ok(Rendered::json(payload))
            }
            (
                BackendKind::OpenaiChat,
                OutputSpec::Audio {
                    voice,
                    max_tokens,
                    model_params,
                    ..
                },
            ) => {
                let mut payload =
                    Value::Object(ChatInputs::plan(parts, dialect.media_input)?.to_object()?);
                payload["model"] = self.model.clone().into();
                payload["modalities"] = serde_json::json!(dialect.audio_modalities);
                payload["stream"] = true.into();
                payload["temperature"] = self.temperature.into();
                let mut audio = serde_json::json!({"format": "pcm16"});
                match (&dialect.voice, voice) {
                    (VoiceSlot::AudioObject, Some(name)) => {
                        audio["voice"] = name.clone().into();
                    }
                    (VoiceSlot::TopLevel(key), Some(name)) => {
                        payload[*key] = name.clone().into();
                    }
                    (_, None) => {}
                }
                payload["audio"] = audio;
                for key in dialect.knob_names.max_tokens {
                    if let Some(limit) = max_tokens {
                        payload[*key] = (*limit).into();
                    }
                }
                self.put_model_params(&mut payload, model_params);
                Ok(Rendered::json(payload))
            }
            (
                BackendKind::OpenaiSpeech,
                OutputSpec::Audio {
                    voice,
                    max_tokens,
                    model_params,
                    ..
                },
            ) => {
                let mut payload = serde_json::json!({
                    "model": self.model,
                    "input": text_prompt(parts)?,
                    "response_format": dialect.speech.response_format,
                });
                if let Some(name) = voice {
                    payload["voice"] = name.clone().into();
                }
                if dialect.speech.stream_bool {
                    payload["stream"] = true.into();
                }
                if let Some(key) = dialect.speech.stream_format {
                    payload[key] = dialect.speech.stream_format_value.into();
                }
                if let (Some(key), Some(limit)) = (dialect.speech.max_tokens_key, max_tokens) {
                    payload[key] = (*limit).into();
                }
                self.put_model_params(&mut payload, model_params);
                Ok(Rendered::json(payload))
            }
            (BackendKind::OpenaiImageEdits, OutputSpec::Image { .. }) => {
                // Multipart, and the knobs ride as form fields rather than JSON:
                // M* parses the form manually and json-decodes each value, so a
                // non-scalar must be sent as its JSON text.
                // The prompt and the image arrive together here, so the strict
                // text-only reader would reject a valid edit request.
                let prompt = text_prompt_lenient(parts);
                if prompt.is_empty() {
                    bail!("image editing requires an instruction prompt")
                }
                let mut payload = serde_json::json!({
                    "model": self.model,
                    "prompt": prompt,
                    "response_format": "b64_json",
                });
                self.put_image_controls(&mut payload, output)?;
                let file = upload_from(parts, Modality::Image, "image")?;
                Ok(Rendered::form(payload, Some(file)))
            }
            (
                BackendKind::OpenaiVideos,
                OutputSpec::Video {
                    width,
                    height,
                    frames,
                    steps,
                    fps,
                    guidance,
                    seed,
                    model_params,
                },
            ) => {
                let names = dialect.knob_names;
                let mut payload = serde_json::json!({
                    "model": self.model,
                    "prompt": text_prompt_lenient(parts),
                    "size": format!("{width}x{height}"),
                    "response_format": "b64_json",
                });
                if dialect.video.nested_controls {
                    dialect.put_knob(&mut payload, "num_frames", (*frames).into());
                    if let Some(rate) = fps {
                        if rate.fract() != 0.0 {
                            bail!(
                                "dialect {} requires video fps to be a whole number",
                                dialect.name
                            );
                        }
                        dialect.put_knob(&mut payload, "fps", (*rate as u64).into());
                    }
                } else {
                    payload["num_frames"] = (*frames).into();
                    if let Some(rate) = fps {
                        payload["fps"] = (*rate).into();
                    }
                }
                // Image- and video-conditioned generation are first-class fields
                // on this surface rather than message content.
                for part in parts {
                    if let PreparedInputPart::Media { modality, data_url } = part {
                        match modality {
                            Modality::Image if dialect.video.input_reference => {
                                payload["input_reference"] = data_url.clone().into()
                            }
                            Modality::Image => payload["image"] = data_url.clone().into(),
                            Modality::Video if dialect.video.input_reference => {
                                bail!("dialect {} video generation accepts image conditioning through input_reference, not video conditioning", dialect.name)
                            }
                            Modality::Video => payload["video"] = data_url.clone().into(),
                            _ => bail!("video generation accepts image or video conditioning only"),
                        }
                    }
                }
                if let Some(key) = names.steps {
                    dialect.put_knob(&mut payload, key, (*steps).into());
                }
                if let (Some(key), Some(value)) = (names.guidance, guidance) {
                    dialect.put_knob(&mut payload, key, (*value).into());
                }
                if let (Some(key), Some(value)) = (names.seed, seed) {
                    dialect.put_knob(&mut payload, key, (*value).into());
                }
                self.put_model_params(&mut payload, model_params);
                if dialect.video.multipart {
                    // vLLM-Omni's video routes take multipart/form-data, so the
                    // same declared shape goes out as form fields instead.
                    return Ok(Rendered::form(
                        payload,
                        upload_from(parts, Modality::Image, "input_reference_image").ok(),
                    ));
                }
                Ok(Rendered::json(payload))
            }
            (
                BackendKind::OpenaiTranscriptions | BackendKind::OpenaiTranslations,
                OutputSpec::Text { .. },
            ) => {
                let mut payload = serde_json::json!({
                    "model": self.model,
                    "response_format": "json",
                    "temperature": self.temperature,
                });
                // A prompt is optional here and ignored by some models; send it
                // only when the trace actually carries text.
                let prompt = text_prompt_lenient(parts);
                if !prompt.is_empty() {
                    payload["prompt"] = prompt.into();
                }
                let file = upload_from(parts, Modality::Audio, "file")?;
                Ok(Rendered::form(payload, Some(file)))
            }
            (backend, requested) => bail!(
                "backend {backend:?} cannot produce {:?} output",
                requested.modality()
            ),
        }
    }

    /// Render the image knobs a trace declares semantically into whatever this
    /// dialect calls them, dropping any it has no name for rather than guessing.
    fn put_image_controls(&self, payload: &mut Value, output: &OutputSpec) -> Result<()> {
        let OutputSpec::Image {
            width,
            height,
            steps,
            count,
            guidance,
            seed,
            model_params,
        } = output
        else {
            bail!("image controls requested for a non-image output")
        };
        let names = self.dialect.knob_names;
        match (names.width, names.height) {
            (Some(w), Some(h)) => {
                payload[w] = (*width).into();
                payload[h] = (*height).into();
            }
            // OpenAI and the servers that copied it take one `WxH` string.
            _ => payload["size"] = format!("{width}x{height}").into(),
        }
        if let Some(key) = names.count {
            payload[key] = (*count).into();
        }
        if let Some(key) = names.steps {
            self.dialect.put_knob(payload, key, (*steps).into());
        }
        if let (Some(key), Some(value)) = (names.guidance, guidance) {
            self.dialect.put_knob(payload, key, (*value).into());
        }
        if let (Some(key), Some(value)) = (names.seed, seed) {
            self.dialect.put_knob(payload, key, (*value).into());
        }
        self.put_model_params(payload, model_params);
        Ok(())
    }

    /// Apply only this dialect's own namespace. A trace may carry knobs for
    /// several servers; sending another server's knobs is how a run ends up
    /// measuring something the operator did not ask for.
    fn put_model_params(&self, payload: &mut Value, params: &crate::schema::ModelParams) {
        let Some(knobs) = params.get(self.dialect.name) else {
            return;
        };
        for (key, value) in knobs {
            self.dialect.put_knob(payload, key, value.clone());
        }
    }

    async fn fold_response(
        &self,
        request_id: &str,
        rendered: &Rendered,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) {
        let request = self
            .client
            .post(&self.endpoint)
            .header("x-request-id", request_id);
        let request = if !rendered.multipart {
            request.json(&rendered.body)
        } else {
            {
                let mut form = reqwest::multipart::Form::new();
                for (key, value) in rendered.body.as_object().into_iter().flatten() {
                    // Scalars go as their bare text; anything structured goes as
                    // JSON text, which is what a manual form parser decodes.
                    let rendered_value = match value {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                    form = form.text(key.clone(), rendered_value);
                }
                if let Some(file) = &rendered.file {
                    let part = match reqwest::multipart::Part::bytes(file.bytes.clone())
                        .file_name(file.filename.clone())
                        .mime_str(&file.mime)
                    {
                        Ok(part) => part,
                        Err(error) => return fold.fail(format!("invalid upload mime: {error}")),
                    };
                    form = form.part(file.field, part);
                }
                request.multipart(form)
            }
        };
        let response = request.send().await;
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => return fold.fail(super::http_failure(response).await),
            Err(error) => return fold.fail(format!("request error: {error}")),
        };
        if self.backend == BackendKind::OpenaiSpeech {
            match self.dialect.speech.deltas {
                AudioDeltas::RawBytes => {
                    self.fold_raw_audio(response, output, send_instant, fold)
                        .await
                }
                _ => {
                    self.fold_sse_audio(response, output, send_instant, fold)
                        .await
                }
            }
        } else if self.backend == BackendKind::OpenaiChat
            && matches!(output, OutputSpec::Audio { .. })
        {
            self.fold_chat_audio(response, output, send_instant, fold)
                .await;
        } else if self.backend == BackendKind::OpenaiVideos && self.dialect.video.raw_response {
            // `/v1/videos/sync` writes the MP4 back as the response body; there
            // is no envelope to parse and nothing to base64-decode.
            match response.bytes().await {
                Ok(bytes) if bytes.is_empty() => fold.fail("video response body was empty"),
                Ok(bytes) => {
                    fold.answered = true;
                    fold.absorb(&bytes, elapsed_ms(send_instant), None);
                }
                Err(error) => fold.fail(format!("response read error: {error}")),
            }
        } else {
            match response.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                    Ok(value) => {
                        self.absorb_json_media(&value, output, elapsed_ms(send_instant), fold)
                    }
                    Err(error) => fold.fail(format!("invalid JSON response: {error}")),
                },
                Err(error) => fold.fail(format!("response read error: {error}")),
            }
        }
    }

    async fn fold_raw_audio(
        &self,
        response: reqwest::Response,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) {
        let OutputSpec::Audio { sample_rate_hz, .. } = output else {
            return;
        };
        let mut stream = response.bytes_stream();
        loop {
            match timeout(
                Duration::from_secs(self.stream_idle_timeout_secs),
                stream.next(),
            )
            .await
            {
                Ok(Some(Ok(chunk))) => {
                    let duration = pcm16_duration_ms(chunk.len(), *sample_rate_hz, 1);
                    fold.absorb(&chunk, elapsed_ms(send_instant), Some(duration));
                }
                Ok(Some(Err(error))) => {
                    return fold.fail(format!("response stream error: {error}"))
                }
                Ok(None) => return,
                Err(_) => {
                    return fold.fail(format!(
                        "stream idle timeout after {}s",
                        self.stream_idle_timeout_secs
                    ))
                }
            }
        }
    }

    /// The speech surface's SSE variant. Same line framing as chat, different
    /// event vocabulary, so the delta locator is what varies -- not the loop.
    async fn fold_sse_audio(
        &self,
        response: reqwest::Response,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) {
        self.fold_sse(
            response,
            output,
            send_instant,
            fold,
            self.dialect.speech.deltas,
        )
        .await
    }

    async fn fold_chat_audio(
        &self,
        response: reqwest::Response,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) {
        self.fold_sse(
            response,
            output,
            send_instant,
            fold,
            self.dialect.audio_deltas,
        )
        .await
    }

    async fn fold_sse(
        &self,
        response: reqwest::Response,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
        deltas: AudioDeltas,
    ) {
        let OutputSpec::Audio { sample_rate_hz, .. } = output else {
            return;
        };
        let mut stream = response.bytes_stream();
        let mut buffer = BytesMut::with_capacity(8192);
        loop {
            match timeout(
                Duration::from_secs(self.stream_idle_timeout_secs),
                stream.next(),
            )
            .await
            {
                Ok(Some(Ok(chunk))) => {
                    buffer.extend_from_slice(&chunk);
                    while let Some(index) = buffer.iter().position(|byte| *byte == b'\n') {
                        let line = buffer.split_to(index + 1);
                        let Some(data) = sse_data_line(&line) else {
                            continue;
                        };
                        if data == b"[DONE]" {
                            return;
                        }
                        let Ok(value) = serde_json::from_slice::<Value>(data) else {
                            continue;
                        };
                        let Some(encoded) = audio_delta(deltas, &value) else {
                            continue;
                        };
                        match STANDARD.decode(encoded) {
                            Ok(bytes) => {
                                let duration =
                                    Some(pcm16_duration_ms(bytes.len(), *sample_rate_hz, 1));
                                fold.absorb(&bytes, elapsed_ms(send_instant), duration);
                            }
                            Err(error) => {
                                return fold.fail(format!("invalid base64 audio delta: {error}"))
                            }
                        }
                    }
                }
                Ok(Some(Err(error))) => {
                    return fold.fail(format!("response stream error: {error}"))
                }
                Ok(None) => return,
                Err(_) => {
                    return fold.fail(format!(
                        "stream idle timeout after {}s",
                        self.stream_idle_timeout_secs
                    ))
                }
            }
        }
    }

    fn absorb_json_media(
        &self,
        value: &Value,
        output: &OutputSpec,
        at_ms: f64,
        fold: &mut MediaFold,
    ) {
        // A 2xx is not the same as a success: vLLM answers an unsupported
        // request with HTTP 200 and an error envelope in the body. Reporting
        // "no media in the response" there would hide the one sentence that
        // says why, so the server's own message wins.
        if let Some(message) = server_error(value) {
            return fold.fail(message);
        }
        match output {
            OutputSpec::Image { .. } => {
                let encoded = read_images(self.dialect.image_reads, value);
                if encoded.is_empty() {
                    return fold
                        .fail("image response matched none of this dialect's declared shapes");
                }
                // Every returned image. A request that asked for `n` and folded
                // one would under-report bytes by a factor of `n` while still
                // reporting SUCCESS.
                for payload in encoded {
                    match STANDARD.decode(payload) {
                        Ok(bytes) => fold.absorb(&bytes, at_ms, None),
                        Err(error) => return fold.fail(format!("invalid base64 image: {error}")),
                    }
                }
            }
            OutputSpec::Audio { sample_rate_hz, .. } => {
                let encoded = value
                    .pointer("/choices/0/message/audio/data")
                    .and_then(Value::as_str);
                match encoded.map(|value| STANDARD.decode(value)) {
                    Some(Ok(bytes)) => {
                        let duration = wav_duration_ms(&bytes)
                            .or_else(|| Some(pcm16_duration_ms(bytes.len(), *sample_rate_hz, 1)));
                        fold.absorb(&bytes, at_ms, duration);
                    }
                    Some(Err(error)) => fold.fail(format!("invalid base64 audio: {error}")),
                    None => fold.fail("audio response contained no audio data"),
                }
            }
            OutputSpec::Video { .. } => {
                let encoded = read_images(self.dialect.video_reads, value);
                if encoded.is_empty() {
                    return fold
                        .fail("video response matched none of this dialect's declared shapes");
                }
                for payload in encoded {
                    match STANDARD.decode(payload) {
                        Ok(bytes) => fold.absorb(&bytes, at_ms, None),
                        Err(error) => return fold.fail(format!("invalid base64 video: {error}")),
                    }
                }
            }
            OutputSpec::Text { .. } => {
                // Transcription and translation both answer `{"text": ...}`;
                // `verbose_json` nests the same field alongside segments.
                match value.get("text").and_then(Value::as_str) {
                    Some(text) => {
                        // Present-but-empty is a real answer: a live SGLang-Omni
                        // transcribing noise returns `{"text": ""}` with 200.
                        fold.answered = true;
                        fold.absorb(text.as_bytes(), at_ms, None);
                    }
                    None => fold.fail("transcription response contained no text field"),
                }
            }
            _ => fold.fail("media client received a non-media output"),
        }
    }

    async fn persist(
        &self,
        request_id: &str,
        output: &OutputSpec,
        bytes: &[u8],
    ) -> Result<Option<String>> {
        let Some(directory) = &self.artifact_dir else {
            return Ok(None);
        };
        let extension = match output {
            OutputSpec::Image { .. } => "png",
            OutputSpec::Audio { .. } => "pcm",
            OutputSpec::Video { .. } => "mp4",
            OutputSpec::Text { .. } => "txt",
            _ => "bin",
        };
        let safe_id: String = request_id
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        let path = directory.join(format!("{safe_id}.{extension}"));
        tokio::fs::write(&path, bytes)
            .await
            .with_context(|| format!("failed to write generated artifact {}", path.display()))?;
        Ok(Some(path.to_string_lossy().into_owned()))
    }
}

/// The server's own error message, when the body carries one despite the status.
///
/// Both envelopes are in use: OpenAI nests `{"error": {"message": ...}}`, and
/// some servers return a bare `{"message": ...}`.
fn server_error(value: &Value) -> Option<String> {
    let detail = value
        .pointer("/error/message")
        .or_else(|| value.get("error").filter(|error| error.is_string()))
        .or_else(|| value.get("message"))?;
    let text = detail.as_str()?;
    if text.trim().is_empty() {
        return None;
    }
    let code = value
        .pointer("/error/code")
        .map(|code| format!(" (code {code})"))
        .unwrap_or_default();
    Some(format!("server reported: {text}{code}"))
}

/// Locate a streamed audio payload according to how this dialect frames one.
fn audio_delta(shape: AudioDeltas, value: &Value) -> Option<&str> {
    match shape {
        AudioDeltas::OpenAiDelta => value
            .pointer("/choices/0/delta/audio/data")
            .and_then(Value::as_str),
        // vLLM-Omni tags the chunk instead of nesting, and puts base64 where
        // text normally goes -- so the gate is load-bearing, not a filter.
        AudioDeltas::ModalityGated => {
            if value.get("modality").and_then(Value::as_str) != Some("audio") {
                return None;
            }
            value
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        }
        AudioDeltas::SpeechSse => {
            if value.get("type").and_then(Value::as_str) != Some("speech.audio.delta") {
                return None;
            }
            value.get("audio").and_then(Value::as_str)
        }
        AudioDeltas::RawBytes => None,
    }
}

/// Collect every base64 image payload the dialect's declared shapes can find.
fn read_images<'a>(shapes: &[MediaRead], value: &'a Value) -> Vec<&'a str> {
    let mut found = Vec::new();
    for shape in shapes {
        match shape {
            MediaRead::DataB64Json => {
                if let Some(items) = value.get("data").and_then(Value::as_array) {
                    found.extend(
                        items
                            .iter()
                            .filter_map(|item| item.get("b64_json").and_then(Value::as_str)),
                    );
                }
            }
            MediaRead::ChatContentDataUrl => {
                if let Some(parts) = value
                    .pointer("/choices/0/message/content")
                    .and_then(Value::as_array)
                {
                    found.extend(parts.iter().filter_map(|part| {
                        part.pointer("/image_url/url")
                            .and_then(Value::as_str)
                            .and_then(data_url_payload)
                    }));
                }
            }
            MediaRead::ChatDeltaDataUrl => {
                if let Some(payload) = value
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                    .and_then(data_url_payload)
                {
                    found.push(payload);
                }
            }
        }
        if !found.is_empty() {
            break;
        }
    }
    found
}

pub(super) fn data_url_payload(url: &str) -> Option<&str> {
    url.split_once(',').map(|(_, payload)| payload)
}

/// Pull one media part back out as raw bytes for a multipart upload.
///
/// Inputs are prepared as data URLs so the JSON surfaces can embed them
/// directly; the form surfaces need the bytes, so they are decoded back here.
/// The decode happens while building the request, before the response clock
/// starts, so it cannot inflate a measured latency.
fn upload_from(
    parts: &[PreparedInputPart],
    want: Modality,
    field: &'static str,
) -> Result<FilePart> {
    for part in parts {
        if let PreparedInputPart::Media { modality, data_url } = part {
            if *modality != want {
                continue;
            }
            let meta = data_url
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(','))
                .map(|(meta, _)| meta.split(';').next().unwrap_or("").to_string())
                .unwrap_or_default();
            let payload = data_url_payload(data_url)
                .ok_or_else(|| anyhow::anyhow!("{field} upload is not a data URL"))?;
            let bytes = STANDARD
                .decode(payload)
                .map_err(|error| anyhow::anyhow!("invalid base64 {field} upload: {error}"))?;
            let extension = meta.rsplit('/').next().unwrap_or("bin");
            return Ok(FilePart {
                field,
                filename: format!("{field}.{extension}"),
                mime: if meta.is_empty() {
                    "application/octet-stream".into()
                } else {
                    meta
                },
                bytes,
            });
        }
    }
    bail!("this surface requires one {want:?} input to upload")
}

/// Video generation may be conditioned entirely by an image, with no text at
/// all, so an empty prompt is legal here where the speech surface requires one.
fn text_prompt_lenient(parts: &[PreparedInputPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            PreparedInputPart::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_prompt(parts: &[PreparedInputPart]) -> Result<String> {
    let mut texts = Vec::new();
    for part in parts {
        match part {
            PreparedInputPart::System(_) => {}
            PreparedInputPart::Text(text) => texts.push(text.as_str()),
            PreparedInputPart::Media { .. } => bail!("selected backend accepts text input only"),
        }
    }
    if texts.is_empty() {
        bail!("request has no text prompt")
    }
    Ok(texts.join("\n"))
}

pub(super) fn pcm16_duration_ms(bytes: usize, sample_rate_hz: u32, channels: u16) -> f64 {
    bytes as f64 * 1_000.0 / (sample_rate_hz as f64 * channels as f64 * 2.0)
}

fn wav_duration_ms(bytes: &[u8]) -> Option<f64> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12usize;
    let mut byte_rate = None;
    let mut data_len = None;
    while offset.checked_add(8)? <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        let body = offset + 8;
        if body.checked_add(length)? > bytes.len() {
            return None;
        }
        if id == b"fmt " && length >= 12 {
            byte_rate =
                Some(u32::from_le_bytes(bytes[body + 8..body + 12].try_into().ok()?) as f64);
        }
        if id == b"data" {
            data_len = Some(length as f64);
        }
        offset = body + length + (length % 2);
    }
    Some(data_len? * 1_000.0 / byte_rate?)
}

pub(super) fn media_failure(
    request_id: String,
    modality: Modality,
    timestamp: f64,
    error: impl std::fmt::Display,
) -> GenerationResult {
    GenerationResult {
        outcome: GenerationOutcome {
            request_id,
            output_modality: modality,
            output_len_actual: 0,
            output_len_text_tokens: 0,
            echoed_prompt_tokens: 0,
            server_usage: None,
            finish_reason: None,
            submit_timestamp: timestamp,
            post_timestamp: None,
            complete_timestamp: timestamp,
            first_output_ms: None,
            last_output_ms: None,
            output_bytes: 0,
            output_chunk_count: 0,
            output_duration_ms: None,
            real_time_factor: None,
            artifact_path: None,
            first_token_ms: None,
            first_token_id_ms: None,
            last_token_id_ms: None,
            first_token_event_tokens: 0,
            token_event_count: 0,
            usage_event_count: 0,
            token_delivery_tpot_ms: None,
            response_complete_ms: None,
            terminal_tail_ms: None,
            total_duration_ms: 0.0,
            chunk_count: 0,
            status: "FAILED".into(),
            output_preview: String::new(),
            error: Some(format!("request build error: {error}")),
        },
        output_ids: Vec::new(),
        timeline: Vec::new(),
    }
}

#[cfg(test)]
impl MediaClient {
    /// A client with no network behind it, for exercising payload rendering.
    fn for_test(dialect_name: &str, backend: BackendKind) -> Self {
        Self {
            endpoint: String::new(),
            client: reqwest::Client::new(),
            model: "m".into(),
            backend,
            dialect: dialect_for(dialect_name).unwrap(),
            stream_idle_timeout_secs: 1,
            record_timeline: false,
            artifact_dir: None,
            temperature: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ModelParams;

    fn image_output() -> OutputSpec {
        let mut model_params = ModelParams::new();
        model_params.insert(
            "mstar".into(),
            [("cfg_renorm_type".to_string(), Value::from("text_channel"))]
                .into_iter()
                .collect(),
        );
        model_params.insert(
            "vllm-omni".into(),
            [("cfg_img_scale".to_string(), Value::from(2.0))]
                .into_iter()
                .collect(),
        );
        OutputSpec::Image {
            width: 1024,
            height: 1024,
            steps: 50,
            count: 2,
            guidance: Some(4.0),
            seed: Some(7),
            model_params,
        }
    }

    fn text_part() -> Vec<PreparedInputPart> {
        vec![PreparedInputPart::Text("a cat".into())]
    }

    #[test]
    fn one_trace_renders_differently_per_dialect() {
        let mstar = MediaClient::for_test("mstar", BackendKind::OpenaiImages)
            .build_payload(&text_part(), &image_output())
            .unwrap()
            .body;
        let omni = MediaClient::for_test("vllm-omni", BackendKind::OpenaiImages)
            .build_payload(&text_part(), &image_output())
            .unwrap()
            .body;
        let dynamo = MediaClient::for_test("dynamo", BackendKind::OpenaiImages)
            .build_payload(&text_part(), &image_output())
            .unwrap()
            .body;

        // M* reads pydantic `model_extra`: knobs must be flat or they are lost.
        assert_eq!(mstar["num_inference_steps"], 50);
        assert_eq!(mstar["cfg_text_scale"], 4.0);
        assert_eq!(mstar["size"], "1024x1024");

        // vLLM-Omni documents a real `extra_body` envelope, and names the same
        // guidance knob the diffusers way.
        assert_eq!(omni["extra_body"]["num_inference_steps"], 50);
        assert_eq!(omni["extra_body"]["guidance_scale"], 4.0);
        assert_eq!(omni["width"], 1024);
        assert!(omni.get("size").is_none());

        assert_eq!(dynamo["nvext"]["num_inference_steps"], 50);
    }

    #[test]
    fn a_dialect_reads_only_its_own_model_params() {
        let mstar = MediaClient::for_test("mstar", BackendKind::OpenaiImages)
            .build_payload(&text_part(), &image_output())
            .unwrap()
            .body;
        assert_eq!(mstar["cfg_renorm_type"], "text_channel");
        // The trace also carries a vLLM-Omni knob; sending it here would change
        // what the run measures without the operator asking for it.
        assert!(mstar.get("cfg_img_scale").is_none());
    }

    #[test]
    fn openai_drops_knobs_it_has_no_name_for() {
        let openai = MediaClient::for_test("openai", BackendKind::OpenaiImages)
            .build_payload(&text_part(), &image_output())
            .unwrap()
            .body;
        // OpenAI 400s on unrecognized params, so an unmappable knob is dropped
        // rather than guessed at.
        assert!(openai.get("num_inference_steps").is_none());
        assert!(openai.get("cfg_text_scale").is_none());
        assert!(openai.get("seed").is_none());
        assert_eq!(openai["size"], "1024x1024");
        assert_eq!(openai["n"], 2);
    }

    #[test]
    fn audio_voice_goes_where_each_server_looks_for_it() {
        let output = OutputSpec::Audio {
            sample_rate_hz: 24_000,
            max_tokens: Some(256),
            max_samples: None,
            voice: Some("Ethan".into()),
            model_params: ModelParams::new(),
        };
        let mstar = MediaClient::for_test("mstar", BackendKind::OpenaiChat)
            .build_payload(&text_part(), &output)
            .unwrap()
            .body;
        // Every server surveyed reads `audio.voice`; a top-level `speaker` is
        // read by none of them and would silently yield the default voice.
        assert_eq!(mstar["audio"]["voice"], "Ethan");
        assert!(mstar.get("speaker").is_none());
        // M* ignores the cap unless its own second name is present.
        assert_eq!(mstar["max_tokens"], 256);
        assert_eq!(mstar["max_output_tokens"], 256);

        let openai = MediaClient::for_test("openai", BackendKind::OpenaiChat)
            .build_payload(&text_part(), &output)
            .unwrap()
            .body;
        // OpenAI rejects audio-only output.
        assert_eq!(openai["modalities"], serde_json::json!(["text", "audio"]));
        assert_eq!(
            openai["max_completion_tokens"], 256,
            "openai deprecated max_tokens"
        );
    }

    #[test]
    fn image_generation_is_refused_where_the_dialect_cannot_express_it() {
        let err = MediaClient::for_test("openai", BackendKind::OpenaiChat)
            .build_payload(&text_part(), &image_output())
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot generate images over chat"), "{err}");
    }

    #[test]
    fn an_empty_but_present_transcript_is_a_successful_answer() {
        // A live SGLang-Omni transcribing noise returns {"text": ""} with HTTP
        // 200. Counting that as a failure would report a healthy ASR server as
        // 100% failed whenever the audio contains no words.
        let client = MediaClient::for_test("sglang-omni", BackendKind::OpenaiTranscriptions);
        let mut fold = MediaFold::new(Modality::Text, false);
        client.absorb_json_media(
            &serde_json::json!({"text": "", "usage": {"type": "duration", "seconds": 1}}),
            &OutputSpec::Text { max_tokens: 8 },
            1.0,
            &mut fold,
        );
        assert!(fold.error.is_none());
        assert!(
            fold.answered,
            "a present field is an answer even when empty"
        );
        assert!(fold.bytes.is_empty());
    }

    #[test]
    fn a_missing_transcript_field_is_still_a_failure() {
        let client = MediaClient::for_test("sglang-omni", BackendKind::OpenaiTranscriptions);
        let mut fold = MediaFold::new(Modality::Text, false);
        client.absorb_json_media(
            &serde_json::json!({"usage": {"seconds": 1}}),
            &OutputSpec::Text { max_tokens: 8 },
            1.0,
            &mut fold,
        );
        assert!(!fold.answered);
        assert!(fold.error.unwrap().contains("no text field"));
    }

    #[test]
    fn a_two_hundred_carrying_an_error_envelope_reports_the_server_message() {
        // Observed against vLLM 0.11: an unsupported surface answers HTTP 200
        // with a 400-coded error in the body. Without this the operator sees
        // "response contained no text field" and has to go find the real reason.
        let client = MediaClient::for_test("vllm", BackendKind::OpenaiTranscriptions);
        let mut fold = MediaFold::new(Modality::Text, false);
        client.absorb_json_media(
            &serde_json::json!({
                "error": {
                    "message": "The model does not support Transcriptions API",
                    "type": "BadRequestError",
                    "code": 400
                }
            }),
            &OutputSpec::Text { max_tokens: 8 },
            1.0,
            &mut fold,
        );
        let error = fold.error.expect("an error envelope must fail the request");
        assert!(
            error.contains("does not support Transcriptions API"),
            "{error}"
        );
        assert!(error.contains("400"), "{error}");
    }

    #[test]
    fn a_normal_response_is_not_mistaken_for_an_error() {
        let client = MediaClient::for_test("vllm", BackendKind::OpenaiTranscriptions);
        let mut fold = MediaFold::new(Modality::Text, false);
        client.absorb_json_media(
            &serde_json::json!({"text": "hello", "error": null}),
            &OutputSpec::Text { max_tokens: 8 },
            1.0,
            &mut fold,
        );
        assert!(fold.error.is_none());
        assert_eq!(fold.bytes, b"hello");
    }

    #[test]
    fn every_returned_image_is_measured() {
        let client = MediaClient::for_test("openai", BackendKind::OpenaiImages);
        let mut fold = MediaFold::new(Modality::Image, false);
        let response = serde_json::json!({
            "data": [{"b64_json": "AAAA"}, {"b64_json": "AAAA"}, {"b64_json": "AAAA"}]
        });
        client.absorb_json_media(&response, &image_output(), 1.0, &mut fold);
        // Folding only data[0] would report a third of the generated bytes and
        // still call the request a success.
        assert_eq!(fold.output_chunk_count, 3);
        assert_eq!(fold.bytes.len(), 9);
        assert!(fold.error.is_none());
    }

    fn media_part(modality: Modality, data_url: &str) -> PreparedInputPart {
        PreparedInputPart::Media {
            modality,
            data_url: data_url.into(),
        }
    }

    fn video_output() -> OutputSpec {
        OutputSpec::Video {
            width: 512,
            height: 512,
            frames: 16,
            steps: 30,
            fps: Some(8.0),
            guidance: Some(3.0),
            seed: Some(1),
            model_params: ModelParams::new(),
        }
    }

    #[test]
    fn image_edits_uploads_the_image_and_sends_knobs_as_form_fields() {
        let parts = vec![
            PreparedInputPart::Text("make it bluer".into()),
            media_part(Modality::Image, "data:image/png;base64,QUJD"),
        ];
        let rendered = MediaClient::for_test("mstar", BackendKind::OpenaiImageEdits)
            .build_payload(&parts, &image_output())
            .unwrap();
        let file = rendered.file.expect("edits must upload the source image");
        assert_eq!(file.field, "image");
        assert_eq!(file.mime, "image/png");
        assert_eq!(file.bytes, b"ABC");
        assert_eq!(rendered.body["prompt"], "make it bluer");
        // Flat for M*, exactly as on the JSON surface.
        assert_eq!(rendered.body["num_inference_steps"], 50);
        assert_eq!(rendered.body["cfg_renorm_type"], "text_channel");
    }

    #[test]
    fn a_form_surface_stays_a_form_even_with_nothing_to_upload() {
        // A live vLLM-Omni rejects the JSON version of a text-only video
        // request with "Field required: prompt" -- it reads a form, and an
        // absent upload is not a reason to stop sending one.
        let rendered = MediaClient::for_test("vllm-omni", BackendKind::OpenaiVideos)
            .build_payload(&text_part(), &video_output())
            .unwrap();
        assert!(rendered.multipart, "vllm-omni's video route takes a form");
        assert!(rendered.file.is_none(), "text-only: nothing to upload");
        assert_eq!(rendered.body["prompt"], "a cat");
    }

    #[test]
    fn a_json_video_surface_is_unaffected() {
        let rendered = MediaClient::for_test("mstar", BackendKind::OpenaiVideos)
            .build_payload(&text_part(), &video_output())
            .unwrap();
        assert!(!rendered.multipart);
    }

    #[test]
    fn video_generation_conditions_on_media_as_first_class_fields() {
        let parts = vec![
            PreparedInputPart::Text("pan left".into()),
            media_part(Modality::Image, "data:image/png;base64,QUJD"),
        ];
        let body = MediaClient::for_test("mstar", BackendKind::OpenaiVideos)
            .build_payload(&parts, &video_output())
            .unwrap()
            .body;
        assert_eq!(body["num_frames"], 16);
        assert_eq!(body["fps"], 8.0);
        assert_eq!(body["size"], "512x512");
        // Conditioning is a request field on this surface, not message content.
        assert_eq!(body["image"], "data:image/png;base64,QUJD");
        assert_eq!(body["num_inference_steps"], 30);
    }

    #[test]
    fn dynamo_video_uses_the_documented_surface_shape() {
        let parts = vec![
            PreparedInputPart::Text("pan left".into()),
            media_part(Modality::Image, "data:image/png;base64,QUJD"),
        ];
        let body = MediaClient::for_test("dynamo", BackendKind::OpenaiVideos)
            .build_payload(&parts, &video_output())
            .unwrap()
            .body;
        assert_eq!(body["input_reference"], "data:image/png;base64,QUJD");
        assert_eq!(body["nvext"]["num_frames"], 16);
        assert_eq!(body["nvext"]["fps"], 8);
        assert_eq!(body["nvext"]["num_inference_steps"], 30);
        assert_eq!(body["nvext"]["guidance_scale"], 3.0);
        assert_eq!(body["nvext"]["seed"], 1);
        assert!(body.get("num_frames").is_none());
        assert!(body.get("fps").is_none());
        assert!(body.get("image").is_none());
    }

    #[test]
    fn dynamo_speech_is_one_shot_and_honors_the_output_cap() {
        let output = OutputSpec::Audio {
            sample_rate_hz: 24_000,
            max_tokens: Some(128),
            max_samples: None,
            voice: Some("Vivian".into()),
            model_params: ModelParams::new(),
        };
        let body = MediaClient::for_test("dynamo", BackendKind::OpenaiSpeech)
            .build_payload(&text_part(), &output)
            .unwrap()
            .body;
        assert_eq!(body["voice"], "Vivian");
        assert_eq!(body["max_new_tokens"], 128);
        assert_eq!(body["response_format"], "pcm");
        assert!(body.get("stream").is_none());
        assert!(body.get("stream_format").is_none());
    }

    #[test]
    fn dynamo_rejects_fractional_video_fps_before_sending() {
        let mut output = video_output();
        if let OutputSpec::Video { fps, .. } = &mut output {
            *fps = Some(7.5);
        }
        let error = MediaClient::for_test("dynamo", BackendKind::OpenaiVideos)
            .build_payload(&text_part(), &output)
            .unwrap_err()
            .to_string();
        assert!(error.contains("whole number"), "{error}");
    }

    #[test]
    fn transcription_uploads_audio_and_asks_for_json() {
        let parts = vec![media_part(Modality::Audio, "data:audio/wav;base64,QUJD")];
        let rendered = MediaClient::for_test("sglang-omni", BackendKind::OpenaiTranscriptions)
            .build_payload(&parts, &OutputSpec::Text { max_tokens: 16 })
            .unwrap();
        let file = rendered.file.expect("transcription must upload audio");
        assert_eq!(file.field, "file");
        assert_eq!(file.mime, "audio/wav");
        assert_eq!(rendered.body["response_format"], "json");
    }

    #[test]
    fn transcription_reads_the_text_field() {
        let client = MediaClient::for_test("sglang-omni", BackendKind::OpenaiTranscriptions);
        let mut fold = MediaFold::new(Modality::Text, false);
        client.absorb_json_media(
            &serde_json::json!({"text": "hello there"}),
            &OutputSpec::Text { max_tokens: 4 },
            1.0,
            &mut fold,
        );
        assert_eq!(fold.bytes, b"hello there");
        assert!(fold.error.is_none());
    }

    #[test]
    fn a_surface_a_system_does_not_serve_is_named_as_such() {
        let dialect = dialect_for("vllm").unwrap();
        let err = dialect
            .surface(BackendKind::OpenaiVideos)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not serve video generation"), "{err}");
        assert!(dialect_for("dynamo")
            .unwrap()
            .surface(BackendKind::OpenaiVideos)
            .is_ok());
        // M* has no ASR surface at all; SGLang-Omni does.
        assert!(dialect_for("mstar")
            .unwrap()
            .surface(BackendKind::OpenaiTranscriptions)
            .is_err());
        assert!(dialect_for("sglang-omni")
            .unwrap()
            .surface(BackendKind::OpenaiTranscriptions)
            .is_ok());
    }

    #[test]
    fn streamed_audio_is_located_by_dialect_not_by_guessing() {
        let openai_shaped = serde_json::json!({
            "choices": [{"delta": {"audio": {"data": "QUJD"}}}]
        });
        let modality_gated = serde_json::json!({
            "modality": "audio",
            "choices": [{"delta": {"content": "QUJD"}}]
        });
        assert_eq!(
            audio_delta(AudioDeltas::OpenAiDelta, &openai_shaped),
            Some("QUJD")
        );
        assert_eq!(audio_delta(AudioDeltas::OpenAiDelta, &modality_gated), None);
        assert_eq!(
            audio_delta(AudioDeltas::ModalityGated, &modality_gated),
            Some("QUJD")
        );
        assert_eq!(
            audio_delta(
                AudioDeltas::SpeechSse,
                &serde_json::json!({"type": "speech.audio.delta", "audio": "QUJD"})
            ),
            Some("QUJD")
        );
    }

    #[test]
    fn wav_duration_uses_the_data_chunk_and_byte_rate() {
        let mut wav = b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\xc0\x5d\0\0\x80\xbb\0\0\x02\0\x10\0data\x04\0\0\0\0\0\0\0".to_vec();
        assert_eq!(wav_duration_ms(&wav), Some(4.0 * 1000.0 / 48_000.0));
        wav[0] = b'X';
        assert_eq!(wav_duration_ms(&wav), None);
    }

    #[test]
    fn media_fold_records_bytes_and_first_output() {
        let mut fold = MediaFold::new(Modality::Audio, true);
        fold.absorb(&[1, 2], 10.0, Some(1.0));
        fold.absorb(&[3], 20.0, Some(0.5));
        assert_eq!(fold.first_output_ms, Some(10.0));
        assert_eq!(fold.last_output_ms, Some(20.0));
        assert_eq!(fold.bytes, vec![1, 2, 3]);
        assert_eq!(fold.duration_ms, 1.5);
        assert_eq!(fold.timeline[1].cumulative_bytes, 3);
    }
}
