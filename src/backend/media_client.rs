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
use serde_json::{json, Value};
use tokio::time::timeout;

use crate::cli::{Args, BackendKind};
use crate::record::GenerationOutcome;
use crate::schema::{Modality, OutputSpec};
use crate::timeline::{EventKind, TimelineEvent};
use crate::util::{elapsed_ms, unix_seconds_now};

use super::{chat_messages, GenerationResult, PreparedInputPart};

pub(crate) struct MediaClient {
    endpoint: String,
    client: reqwest::Client,
    model: String,
    backend: BackendKind,
    stream_idle_timeout_secs: u64,
    record_timeline: bool,
    artifact_dir: Option<PathBuf>,
    temperature: f64,
}

struct MediaFold {
    modality: Modality,
    bytes: Vec<u8>,
    first_output_ms: Option<f64>,
    last_output_ms: Option<f64>,
    output_chunk_count: usize,
    duration_ms: f64,
    timeline: Vec<TimelineEvent>,
    error: Option<String>,
}

impl MediaFold {
    fn new(modality: Modality, record_timeline: bool) -> Self {
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
        }
    }

    fn absorb(&mut self, payload: &[u8], at_ms: f64, duration_ms: Option<f64>) {
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

    fn fail(&mut self, error: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(error.into());
        }
    }
}

impl MediaClient {
    pub(crate) fn new(args: &Args) -> Result<Self> {
        let suffix = match args.backend {
            BackendKind::OpenaiChat => "/chat/completions",
            BackendKind::OpenaiImages => "/images/generations",
            BackendKind::OpenaiSpeech => "/audio/speech",
            _ => bail!("backend {:?} cannot generate media", args.backend),
        };
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
        let mut status = if fold.error.is_none() && !fold.bytes.is_empty() {
            "SUCCESS".to_string()
        } else {
            if fold.error.is_none() {
                fold.fail("backend returned no generated media");
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

    fn build_payload(&self, parts: &[PreparedInputPart], output: &OutputSpec) -> Result<Value> {
        match (self.backend, output) {
            (
                BackendKind::OpenaiImages,
                OutputSpec::Image {
                    width,
                    height,
                    steps,
                    count,
                    cfg_scale,
                    cfg_img_scale,
                    cfg_renorm_type,
                    cfg_interval,
                    seed,
                },
            ) => {
                let prompt = text_prompt(parts)?;
                let mut payload = json!({
                    "model": self.model,
                    "prompt": prompt,
                    "size": format!("{width}x{height}"),
                    "n": count,
                    "num_inference_steps": steps,
                    "response_format": "b64_json",
                });
                let mut extra_params = json!({});
                insert_optional(
                    &mut extra_params,
                    "guidance_scale",
                    cfg_scale.map(Value::from),
                );
                insert_optional(
                    &mut extra_params,
                    "cfg_img_scale",
                    cfg_img_scale.map(Value::from),
                );
                insert_optional(
                    &mut extra_params,
                    "cfg_renorm_type",
                    cfg_renorm_type.clone().map(Value::from),
                );
                insert_optional(
                    &mut extra_params,
                    "cfg_interval",
                    cfg_interval.map(|value| json!(value)),
                );
                insert_optional(&mut payload, "seed", seed.map(Value::from));
                if extra_params
                    .as_object()
                    .is_some_and(|values| !values.is_empty())
                {
                    payload["extra_params"] = extra_params;
                }
                Ok(payload)
            }
            (
                BackendKind::OpenaiChat,
                OutputSpec::Image {
                    width,
                    height,
                    steps,
                    count,
                    cfg_scale,
                    cfg_img_scale,
                    cfg_renorm_type,
                    cfg_interval,
                    seed,
                },
            ) => {
                let mut extra = json!({
                    "height": height,
                    "width": width,
                    "num_inference_steps": steps,
                    "num_outputs_per_prompt": count,
                });
                insert_optional(&mut extra, "guidance_scale", cfg_scale.map(Value::from));
                insert_optional(&mut extra, "cfg_img_scale", cfg_img_scale.map(Value::from));
                insert_optional(
                    &mut extra,
                    "cfg_renorm_type",
                    cfg_renorm_type.clone().map(Value::from),
                );
                insert_optional(
                    &mut extra,
                    "cfg_interval",
                    cfg_interval.map(|value| json!(value)),
                );
                insert_optional(&mut extra, "seed", seed.map(Value::from));
                Ok(json!({
                    "model": self.model,
                    "messages": chat_messages(parts)?,
                    "modalities": ["image"],
                    "stream": false,
                    "extra_body": extra,
                }))
            }
            (
                BackendKind::OpenaiChat,
                OutputSpec::Audio {
                    voice, max_tokens, ..
                },
            ) => {
                let mut payload = json!({
                    "model": self.model,
                    "messages": chat_messages(parts)?,
                    "modalities": ["audio"],
                    "audio": {"format": "pcm16"},
                    "temperature": self.temperature,
                    "thinker_temperature": self.temperature,
                    "stream": true,
                });
                insert_optional(&mut payload, "speaker", voice.clone().map(Value::from));
                insert_optional(&mut payload, "max_tokens", max_tokens.map(Value::from));
                insert_optional(
                    &mut payload,
                    "max_output_tokens",
                    max_tokens.map(Value::from),
                );
                Ok(payload)
            }
            (
                BackendKind::OpenaiSpeech,
                OutputSpec::Audio {
                    voice, max_tokens, ..
                },
            ) => {
                let mut payload = json!({
                    "model": self.model,
                    "input": text_prompt(parts)?,
                    "voice": voice.as_deref().unwrap_or("default"),
                    "response_format": "pcm",
                    "stream": true,
                    "stream_format": "audio",
                });
                insert_optional(&mut payload, "max_new_tokens", max_tokens.map(Value::from));
                Ok(payload)
            }
            (backend, requested) => bail!(
                "backend {backend:?} cannot produce {:?} output",
                requested.modality()
            ),
        }
    }

    async fn fold_response(
        &self,
        request_id: &str,
        payload: &Value,
        output: &OutputSpec,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) {
        let response = self
            .client
            .post(&self.endpoint)
            .header("x-request-id", request_id)
            .json(payload)
            .send()
            .await;
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => return fold.fail(format!("HTTP {}", response.status())),
            Err(error) => return fold.fail(format!("request error: {error}")),
        };
        if self.backend == BackendKind::OpenaiSpeech {
            self.fold_raw_audio(response, output, send_instant, fold)
                .await;
        } else if self.backend == BackendKind::OpenaiChat
            && matches!(output, OutputSpec::Audio { .. })
        {
            self.fold_chat_audio(response, output, send_instant, fold)
                .await;
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

    async fn fold_chat_audio(
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
                        let line = String::from_utf8_lossy(&line);
                        let Some(data) = line.trim().strip_prefix("data: ") else {
                            continue;
                        };
                        if data == "[DONE]" {
                            return;
                        }
                        let Ok(value) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };
                        if value.get("modality").and_then(Value::as_str) != Some("audio") {
                            continue;
                        }
                        let Some(encoded) = value
                            .pointer("/choices/0/delta/content")
                            .and_then(Value::as_str)
                        else {
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
        match output {
            OutputSpec::Image { .. } => {
                let encoded = value
                    .pointer("/data/0/b64_json")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        value
                            .pointer("/choices/0/message/content/0/image_url/url")
                            .and_then(Value::as_str)
                            .and_then(|url| url.split_once(',').map(|(_, payload)| payload))
                    });
                match encoded.map(|value| STANDARD.decode(value)) {
                    Some(Ok(bytes)) => fold.absorb(&bytes, at_ms, None),
                    Some(Err(error)) => fold.fail(format!("invalid base64 image: {error}")),
                    None => fold.fail("image response contained no b64_json or image data URL"),
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

fn insert_optional(object: &mut Value, key: &str, value: Option<Value>) {
    if let (Some(map), Some(value)) = (object.as_object_mut(), value) {
        map.insert(key.into(), value);
    }
}

fn pcm16_duration_ms(bytes: usize, sample_rate_hz: u32, channels: u16) -> f64 {
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

fn media_failure(
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
mod tests {
    use super::*;

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
