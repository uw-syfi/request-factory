//! When each thing arrived on each request's stream.
//!
//! The run summary reports distributions; the JSONL log reports one row per
//! request. Neither can answer "what was this request doing between its first
//! token and its last", which is the question a stalled decode, a bursty server,
//! or a multimodal pipeline's per-stage progress all reduce to.
//!
//! # Events, not tokens
//!
//! A chunk carrying four token ids is **one** observable instant. Writing four
//! rows that share a timestamp would invite a reader to average them as four
//! measurements — the same mistake this codebase already refuses to make when it
//! excludes `first_token_event_tokens` from TPOT's denominator. So one row per
//! parsed response object, and `tokens` says how many tokens that one arrival
//! carried.
//!
//! # Not at the cost of submission
//!
//! The measurement is free: the fold already computes each event's elapsed time.
//! Only the recording is new, and it is arranged so nothing on the request path
//! can be slowed by it:
//!
//! - events are pushed into a `Vec` preallocated to the expected output length,
//!   so there is no allocation, no formatting and no syscall while a response is
//!   streaming;
//! - the whole `Vec` is handed off **once per request**, not once per token;
//! - the handoff is a `try_send` on a bounded channel. A full channel drops that
//!   request's timeline and increments a counter — a slow disk must never become
//!   backpressure on the thing being measured. The counter is reported, so a
//!   lossy run says so instead of quietly under-reporting.
//! - the writer runs on its **own** thread, not on an async worker. Arrow
//!   encoding and zstd are CPU-bound blocking work, and leaving them on the
//!   runtime that drives the requests made them compete with it — measurably, in
//!   the arrival-release lag this client records.
//!
//! The channel is therefore `std::sync::mpsc::sync_channel`, whose producer half
//! is non-blocking by construction and whose consumer half is happy to block.

mod writer;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

use serde::Serialize;

pub(crate) use writer::write_timeline;

/// What one arrival carried.
///
/// New variants are additive: a multimodal pipeline reporting per-stage progress
/// adds its own kinds here rather than a second file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EventKind {
    /// Generated token ids. `tokens` says how many arrived in this one event.
    Tokens,
    /// Generated image/audio/video bytes. `bytes` says how much this arrival carried.
    Media,
    /// A server usage report and no tokens.
    Usage,
    /// A finish reason and no tokens.
    Finish,
    /// Parsed, but carrying nothing this schema names. Recorded rather than
    /// discarded: an unexplained arrival is still an arrival, and when it
    /// happened is evidence.
    Other,
}

impl EventKind {
    /// Written as text, not as an integer code, so a reader opening the file in
    /// DataFusion or pandas needs no legend. Parquet dictionary-encodes a column
    /// of a handful of repeated strings down to about what the integer would
    /// have cost.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Tokens => "tokens",
            Self::Media => "media",
            Self::Usage => "usage",
            Self::Finish => "finish",
            Self::Other => "other",
        }
    }

    /// Classify one parsed response object.
    ///
    /// Precedence, not a set: the row describes one instant, and an arrival that
    /// delivered tokens is a token arrival even if it also restated usage. What
    /// it carried besides is still visible — `tokens` is the count, and the
    /// finish reason is on the request's own log row.
    pub(crate) fn classify(tokens: usize, has_usage: bool, has_finish_reason: bool) -> Self {
        if tokens > 0 {
            Self::Tokens
        } else if has_usage {
            Self::Usage
        } else if has_finish_reason {
            Self::Finish
        } else {
            Self::Other
        }
    }
}

/// One observed arrival on one request's stream.
///
/// Deliberately small and `Copy`: this is pushed inside the streaming fold, and
/// a `Vec` of them is the only per-request allocation the timeline adds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimelineEvent {
    /// Milliseconds since the request was sent. `f32` because this is a wall
    /// clock reading of a few seconds at millisecond scale, where f32 has room
    /// to spare, and the file is one row per event.
    pub(crate) elapsed_ms: f32,
    pub(crate) kind: EventKind,
    /// Tokens delivered by this one arrival. Zero for every non-token kind.
    pub(crate) tokens: u16,
    /// Tokens delivered by this arrival and every earlier one, so a reader can
    /// plot progress without a running fold.
    pub(crate) cumulative_tokens: u32,
    /// Decoded generated-media bytes delivered by this arrival.
    pub(crate) bytes: u32,
    /// Generated-media bytes delivered by this and all earlier arrivals.
    pub(crate) cumulative_bytes: u64,
}

/// One request's stream, start to finish.
pub(crate) struct RequestTimeline {
    pub(crate) request_id: String,
    pub(crate) events: Vec<TimelineEvent>,
}

/// The handoff from the request path to the writer.
///
/// Cloned into every executor. `offer` is the only thing a request-path caller
/// may do with it, and `offer` cannot block.
#[derive(Clone)]
pub(crate) struct TimelineSink {
    sender: SyncSender<RequestTimeline>,
    dropped_requests: Arc<AtomicUsize>,
}

impl TimelineSink {
    /// Hand off one request's timeline, or drop it.
    ///
    /// Never blocks and never awaits. When the writer has fallen behind far
    /// enough to fill the channel, this request's timeline is lost and counted,
    /// because the alternative — making the client wait on a disk — would
    /// corrupt the very latencies the file exists to record.
    pub(crate) fn offer(&self, timeline: RequestTimeline) {
        if timeline.events.is_empty() {
            return;
        }
        if self.sender.try_send(timeline).is_err() {
            self.dropped_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The counter itself, for the writer task to read once at the end.
    ///
    /// Handed over rather than reported through the channel, because the whole
    /// point of the counter is to survive a channel that is already full.
    pub(crate) fn dropped_handle(&self) -> Arc<AtomicUsize> {
        self.dropped_requests.clone()
    }
}

/// What the timeline recorded, for the run summary.
///
/// `dropped_requests` is here rather than only in a log line because a run whose
/// timeline is incomplete must be able to say so to whatever reads it later.
#[derive(Debug, Default, Serialize)]
pub(crate) struct TimelineSummary {
    pub(crate) path: Option<String>,
    pub(crate) requests_written: usize,
    pub(crate) events_written: usize,
    /// Requests whose timeline was discarded because the writer could not keep
    /// up. Nonzero means this file is a sample, not a record.
    pub(crate) dropped_requests: usize,
}

/// Build the sink and the receiving end of its channel.
///
/// `capacity` is in requests, not events. It bounds how far the writer may fall
/// behind before timelines start being dropped instead of queued.
pub(crate) fn channel(capacity: usize) -> (TimelineSink, Receiver<RequestTimeline>) {
    let (sender, receiver) = sync_channel(capacity);
    (
        TimelineSink {
            sender,
            dropped_requests: Arc::new(AtomicUsize::new(0)),
        },
        receiver,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dropped(sink: &TimelineSink) -> usize {
        sink.dropped_handle().load(Ordering::Relaxed)
    }

    #[test]
    fn a_multi_token_arrival_is_one_event_that_counts_its_tokens() {
        // The failure this whole schema exists to prevent: four ids arriving
        // together are one measurement of one instant, not four measurements.
        let kind = EventKind::classify(4, false, false);

        assert_eq!(kind, EventKind::Tokens);
        assert_eq!(kind.name(), "tokens");
    }

    #[test]
    fn an_arrival_is_classified_by_what_it_delivered_not_by_everything_it_mentioned() {
        // A final chunk often carries the last token, the usage block and the
        // finish reason at once. It is still one token arrival.
        assert_eq!(EventKind::classify(1, true, true), EventKind::Tokens);
        assert_eq!(EventKind::classify(0, true, true), EventKind::Usage);
        assert_eq!(EventKind::classify(0, false, true), EventKind::Finish);
        assert_eq!(EventKind::classify(0, false, false), EventKind::Other);
    }

    #[test]
    fn a_full_channel_drops_the_timeline_instead_of_blocking() {
        let (sink, _receiver) = channel(1);
        let timeline = || RequestTimeline {
            request_id: "req".to_string(),
            events: vec![TimelineEvent {
                elapsed_ms: 1.0,
                kind: EventKind::Tokens,
                tokens: 1,
                cumulative_tokens: 1,
                bytes: 0,
                cumulative_bytes: 0,
            }],
        };

        sink.offer(timeline());
        assert_eq!(dropped(&sink), 0);
        // The channel now holds its one permitted request; the next offer must
        // return rather than wait for the writer.
        sink.offer(timeline());
        assert_eq!(dropped(&sink), 1);
    }

    #[test]
    fn an_empty_timeline_is_not_worth_a_channel_slot() {
        let (sink, receiver) = channel(1);

        // A request skipped at the context limit never reached the wire.
        sink.offer(RequestTimeline {
            request_id: "skipped".to_string(),
            events: Vec::new(),
        });

        assert_eq!(dropped(&sink), 0);
        assert!(receiver.try_recv().is_err());
    }
}
