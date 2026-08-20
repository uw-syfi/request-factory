//! The `/realtime` WebSocket surface.
//!
//! Every other surface here is one HTTP request. This one is a socket: the
//! client configures a session, describes a turn, and reads server events until
//! the turn ends. Three systems serve it -- OpenAI, vLLM-Omni and SGLang-Omni --
//! and they disagree about both the event names and how a turn is driven, so the
//! vocabulary comes from the dialect table rather than from this file.
//!
//! One request is one session. That is a deliberate narrowing: a realtime
//! connection can host many turns, but a replay unit has to be a thing that can
//! be timed and compared, and multi-turn sessions are what `executor/session.rs`
//! already models for text. Reusing the request path keeps the measurement
//! identical to the other media surfaces -- same fold, same outcome fields.

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message;

use crate::cli::Args;
use crate::record::GenerationOutcome;
use crate::schema::{Modality, OutputSpec};
use crate::util::{elapsed_ms, unix_seconds_now};

use super::dialect::{RealtimeShape, RealtimeTurn};
use super::media_client::{data_url_payload, media_failure, pcm16_duration_ms, MediaFold};
use super::{dialect_for, Dialect, GenerationResult, PreparedInputPart};

/// Input audio is appended in chunks rather than one frame, because a realtime
/// server's first-output latency depends on when it saw enough input -- sending
/// a whole utterance as one frame measures a different thing than streaming it.
const APPEND_CHUNK_BYTES: usize = 16 * 1024;

pub(crate) struct RealtimeClient {
    url: String,
    model: String,
    dialect: &'static Dialect,
    stream_idle_timeout_secs: u64,
    record_timeline: bool,
    temperature: f64,
}

impl RealtimeClient {
    pub(crate) fn new(args: &Args) -> Result<Self> {
        let dialect = dialect_for(&args.dialect)?;
        let suffix = dialect.surface(args.backend)?;
        let base = args.base_url.trim_end_matches('/');
        // The socket lives at the same authority as the HTTP surfaces; only the
        // scheme changes, so an operator configures one base_url for both.
        let ws_base = match base.split_once("://") {
            Some(("http", rest)) => format!("ws://{rest}"),
            Some(("https", rest)) => format!("wss://{rest}"),
            Some(("ws" | "wss", _)) => base.to_string(),
            _ => bail!("base_url {base:?} has no scheme the realtime socket can use"),
        };
        Ok(Self {
            url: format!("{ws_base}{suffix}?model={}", args.model),
            model: args.model.clone(),
            dialect,
            stream_idle_timeout_secs: args.stream_idle_timeout_secs,
            record_timeline: args.timeline,
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
        let shape = self.dialect.realtime;
        let mut fold = MediaFold::new(output.modality(), self.record_timeline);

        let events = match self.turn_events(parts, output, &shape) {
            Ok(events) => events,
            Err(error) => {
                return media_failure(request_id, output.modality(), submit_timestamp, error)
            }
        };

        let connected = match timeout(
            Duration::from_secs(self.stream_idle_timeout_secs),
            tokio_tungstenite::connect_async(&self.url),
        )
        .await
        {
            Ok(Ok((socket, _response))) => socket,
            Ok(Err(error)) => {
                return media_failure(
                    request_id,
                    output.modality(),
                    submit_timestamp,
                    format!("realtime connect failed: {error}"),
                )
            }
            Err(_) => {
                return media_failure(
                    request_id,
                    output.modality(),
                    submit_timestamp,
                    "realtime connect timed out",
                )
            }
        };
        let mut socket = connected;
        let post_timestamp = Some(unix_seconds_now());

        for event in &events {
            if let Err(error) = socket.send(Message::Text(event.to_string())).await {
                fold.fail(format!("realtime send failed: {error}"));
                break;
            }
        }
        // The turn is only in flight once every client event is on the wire, so
        // the response clock starts here rather than at connect.
        let send_instant = Instant::now();

        if fold.error.is_none() {
            self.read_turn(&mut socket, output, &shape, send_instant, &mut fold)
                .await;
        }
        let _ = socket.close(None).await;

        let response_complete_ms_value = elapsed_ms(send_instant);
        let measured_duration_ms = (fold.duration_ms > 0.0).then_some(fold.duration_ms);
        let real_time_factor = measured_duration_ms
            .filter(|duration| *duration > 0.0)
            .map(|duration| response_complete_ms_value / duration);
        let status = if fold.error.is_none() && (fold.answered || !fold.bytes.is_empty()) {
            "SUCCESS".to_string()
        } else {
            if fold.error.is_none() {
                fold.fail("realtime session produced no output");
            }
            "FAILED".to_string()
        };
        let preview = if fold.modality == Modality::Text {
            String::from_utf8_lossy(&fold.bytes)
                .chars()
                .take(120)
                .collect()
        } else {
            String::new()
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
                complete_timestamp: unix_seconds_now(),
                first_output_ms: fold.first_output_ms,
                last_output_ms: fold.last_output_ms,
                output_bytes: fold.bytes.len(),
                output_chunk_count: fold.output_chunk_count,
                output_duration_ms: measured_duration_ms,
                real_time_factor,
                artifact_path: None,
                first_token_ms: None,
                first_token_id_ms: None,
                last_token_id_ms: None,
                first_token_event_tokens: 0,
                token_event_count: 0,
                usage_event_count: 0,
                token_delivery_tpot_ms: None,
                response_complete_ms: Some(response_complete_ms_value),
                terminal_tail_ms: match (fold.last_output_ms, response_complete_ms_value) {
                    (Some(last), done) if done >= last => Some(done - last),
                    _ => None,
                },
                total_duration_ms: elapsed_ms(start),
                chunk_count: fold.output_chunk_count,
                status,
                output_preview: preview,
                error: fold.error,
            },
            output_ids: Vec::new(),
            timeline: fold.timeline,
        }
    }

    /// The client events that drive one turn, in order.
    ///
    /// Returned as a list rather than sent inline so the whole turn is a value a
    /// test can inspect -- the shaping is where dialects differ, and it should
    /// not need a socket to check.
    pub(super) fn turn_events(
        &self,
        parts: &[PreparedInputPart],
        output: &OutputSpec,
        shape: &RealtimeShape,
    ) -> Result<Vec<Value>> {
        let mut events = Vec::new();
        match shape.turn {
            RealtimeTurn::ItemThenResponse => {
                let mut session = json!({
                    "modalities": shape.modalities,
                    "output_audio_format": "pcm16",
                    "temperature": self.temperature,
                });
                if let OutputSpec::Audio {
                    voice: Some(name), ..
                } = output
                {
                    session["voice"] = name.clone().into();
                }
                events.push(json!({"type": "session.update", "session": session}));

                let mut content = Vec::new();
                for part in parts {
                    match part {
                        PreparedInputPart::System(text) | PreparedInputPart::Text(text) => {
                            content.push(json!({"type": "input_text", "text": text}));
                        }
                        PreparedInputPart::Media { modality, data_url } => {
                            if *modality != Modality::Audio {
                                bail!("the realtime surface accepts text and audio input only");
                            }
                            let payload = data_url_payload(data_url)
                                .ok_or_else(|| anyhow!("realtime audio input is not a data URL"))?;
                            content.push(json!({"type": "input_audio", "audio": payload}));
                        }
                    }
                }
                if content.is_empty() {
                    bail!("realtime turn has no input")
                }
                events.push(json!({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": content}
                }));
                events.push(json!({
                    "type": "response.create",
                    "response": {"modalities": shape.modalities}
                }));
            }
            RealtimeTurn::AudioBufferCommit => {
                // No item and no explicit response request: generation starts on
                // the opening commit and the turn closes on the final one.
                events.push(json!({"type": "session.update", "model": self.model}));
                events.push(json!({"type": "input_audio_buffer.commit", "final": false}));
                let mut appended = false;
                for part in parts {
                    let PreparedInputPart::Media { modality, data_url } = part else {
                        continue;
                    };
                    if *modality != Modality::Audio {
                        bail!("this dialect's realtime turn accepts audio input only");
                    }
                    let payload = data_url_payload(data_url)
                        .ok_or_else(|| anyhow!("realtime audio input is not a data URL"))?;
                    let bytes = STANDARD
                        .decode(payload)
                        .map_err(|error| anyhow!("invalid base64 realtime audio: {error}"))?;
                    for chunk in bytes.chunks(APPEND_CHUNK_BYTES) {
                        events.push(json!({
                            "type": "input_audio_buffer.append",
                            "audio": STANDARD.encode(chunk)
                        }));
                    }
                    appended = true;
                }
                if !appended {
                    bail!("this dialect's realtime turn requires an audio input")
                }
                events.push(json!({"type": "input_audio_buffer.commit", "final": true}));
            }
        }
        Ok(events)
    }

    async fn read_turn<S>(
        &self,
        socket: &mut S,
        output: &OutputSpec,
        shape: &RealtimeShape,
        send_instant: Instant,
        fold: &mut MediaFold,
    ) where
        S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
    {
        let sample_rate_hz = match output {
            OutputSpec::Audio { sample_rate_hz, .. } => Some(*sample_rate_hz),
            _ => None,
        };
        loop {
            match timeout(
                Duration::from_secs(self.stream_idle_timeout_secs),
                socket.next(),
            )
            .await
            {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let Ok(event) = serde_json::from_str::<Value>(&text) else {
                        continue;
                    };
                    if let Some(payload) = realtime_delta(shape, &event, fold.modality) {
                        let decoded = match fold.modality {
                            Modality::Text => Some(payload.as_bytes().to_vec()),
                            _ => match STANDARD.decode(payload) {
                                Ok(bytes) => Some(bytes),
                                Err(error) => {
                                    return fold
                                        .fail(format!("invalid base64 realtime audio: {error}"))
                                }
                            },
                        };
                        if let Some(bytes) = decoded {
                            let duration =
                                sample_rate_hz.map(|rate| pcm16_duration_ms(bytes.len(), rate, 1));
                            fold.absorb(&bytes, elapsed_ms(send_instant), duration);
                        }
                    }
                    match event.get("type").and_then(Value::as_str) {
                        Some("error") => {
                            let detail = event
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .unwrap_or("unspecified");
                            return fold.fail(format!("realtime server error: {detail}"));
                        }
                        Some(kind) if shape.done_types.contains(&kind) => return,
                        _ => {}
                    }
                }
                // A server is free to frame audio as binary rather than base64.
                Ok(Some(Ok(Message::Binary(bytes)))) => {
                    let duration =
                        sample_rate_hz.map(|rate| pcm16_duration_ms(bytes.len(), rate, 1));
                    fold.absorb(&bytes, elapsed_ms(send_instant), duration);
                }
                Ok(Some(Ok(Message::Close(_)))) | Ok(None) => return,
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(error))) => {
                    return fold.fail(format!("realtime stream error: {error}"))
                }
                Err(_) => {
                    return fold.fail(format!(
                        "realtime idle timeout after {}s",
                        self.stream_idle_timeout_secs
                    ))
                }
            }
        }
    }
}

/// Locate the payload of one output delta, if this event carries one.
///
/// The event name and the field holding the payload vary independently: OpenAI
/// renamed the event and kept `delta`; vLLM-Omni kept the name and moved the
/// payload to `audio`. Matching on one without the other silently reads nothing.
pub(super) fn realtime_delta<'a>(
    shape: &RealtimeShape,
    event: &'a Value,
    modality: Modality,
) -> Option<&'a str> {
    let kind = event.get("type").and_then(Value::as_str)?;
    let (want, field) = match modality {
        Modality::Text => (shape.text_delta_type, shape.text_delta_field),
        _ => (shape.audio_delta_type, shape.audio_delta_field),
    };
    if kind != want {
        return None;
    }
    event.get(field).and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(dialect: &str) -> RealtimeClient {
        RealtimeClient {
            url: String::new(),
            model: "m".into(),
            dialect: dialect_for(dialect).unwrap(),
            stream_idle_timeout_secs: 1,
            record_timeline: false,
            temperature: 0.0,
        }
    }

    fn audio_output() -> OutputSpec {
        OutputSpec::Audio {
            sample_rate_hz: 24_000,
            max_tokens: None,
            max_samples: None,
            voice: Some("Ethan".into()),
            model_params: crate::schema::ModelParams::new(),
        }
    }

    #[test]
    fn item_style_turns_describe_the_item_then_ask_for_a_response() {
        let client = client("sglang-omni");
        let parts = vec![PreparedInputPart::Text("say hello".into())];
        let events = client
            .turn_events(&parts, &audio_output(), &client.dialect.realtime)
            .unwrap();
        let types: Vec<_> = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            types,
            vec![
                "session.update",
                "conversation.item.create",
                "response.create"
            ]
        );
        assert_eq!(events[0]["session"]["voice"], "Ethan");
        // Audio-only is rejected by every system serving this surface.
        assert_eq!(events[0]["session"]["modalities"], json!(["text", "audio"]));
        assert_eq!(events[1]["item"]["content"][0]["type"], "input_text");
    }

    #[test]
    fn buffer_style_turns_stream_input_and_close_with_a_final_commit() {
        let client = client("vllm-omni");
        // 40 KB of audio, so the append must chunk rather than send one frame.
        let raw = vec![0u8; 40 * 1024];
        let parts = vec![PreparedInputPart::Media {
            modality: Modality::Audio,
            data_url: format!("data:audio/pcm;base64,{}", STANDARD.encode(&raw)),
        }];
        let events = client
            .turn_events(&parts, &audio_output(), &client.dialect.realtime)
            .unwrap();
        let types: Vec<_> = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect();
        assert_eq!(types[0], "session.update");
        assert_eq!(types[1], "input_audio_buffer.commit");
        assert_eq!(events[1]["final"], false);
        assert_eq!(*types.last().unwrap(), "input_audio_buffer.commit");
        assert_eq!(events.last().unwrap()["final"], true);
        let appends = types
            .iter()
            .filter(|kind| **kind == "input_audio_buffer.append")
            .count();
        assert_eq!(appends, 40 * 1024 / APPEND_CHUNK_BYTES + 1);
        // No item and no response.create: this dialect drives the turn purely
        // from the input buffer.
        assert!(!types.contains(&"conversation.item.create"));
        assert!(!types.contains(&"response.create"));
    }

    #[test]
    fn a_delta_needs_both_the_right_event_name_and_the_right_field() {
        let openai = dialect_for("openai").unwrap().realtime;
        let sglang = dialect_for("sglang-omni").unwrap().realtime;
        let omni = dialect_for("vllm-omni").unwrap().realtime;

        let openai_event = json!({"type": "response.output_audio.delta", "delta": "QUJD"});
        let sglang_event = json!({"type": "response.audio.delta", "delta": "QUJD"});
        let omni_event = json!({"type": "response.audio.delta", "audio": "QUJD"});

        assert_eq!(
            realtime_delta(&openai, &openai_event, Modality::Audio),
            Some("QUJD")
        );
        assert_eq!(
            realtime_delta(&sglang, &sglang_event, Modality::Audio),
            Some("QUJD")
        );
        assert_eq!(
            realtime_delta(&omni, &omni_event, Modality::Audio),
            Some("QUJD")
        );

        // The name matches but the payload lives elsewhere: reading the wrong
        // field yields nothing rather than an error, which is the failure this
        // pairing exists to prevent.
        assert_eq!(realtime_delta(&sglang, &omni_event, Modality::Audio), None);
        assert_eq!(realtime_delta(&omni, &sglang_event, Modality::Audio), None);
        assert_eq!(
            realtime_delta(&openai, &sglang_event, Modality::Audio),
            None
        );
    }

    #[test]
    fn only_the_systems_that_serve_realtime_expose_it() {
        for name in ["openai", "vllm-omni", "sglang-omni"] {
            assert!(
                dialect_for(name).unwrap().realtime_suffix.is_some(),
                "{name}"
            );
        }
        for name in ["vllm", "mstar", "dynamo"] {
            assert!(
                dialect_for(name).unwrap().realtime_suffix.is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_realtime_turn_refuses_input_it_cannot_encode() {
        let client = client("sglang-omni");
        let parts = vec![PreparedInputPart::Media {
            modality: Modality::Image,
            data_url: "data:image/png;base64,QUJD".into(),
        }];
        let err = client
            .turn_events(&parts, &audio_output(), &client.dialect.realtime)
            .unwrap_err()
            .to_string();
        assert!(err.contains("text and audio input only"), "{err}");
    }
}
