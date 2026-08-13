//! The Arrow/Parquet side of the timeline, which runs off the request path.
//!
//! Everything expensive is here: building arrays, compressing, and touching the
//! disk. The request path's entire contribution is one `try_send` of a `Vec`.
//!
//! Synchronous and blocking on purpose. Run this on an async worker and its zstd
//! compression competes with the very requests it is recording; the caller puts
//! it on a thread of its own.

use std::fs::File;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::builder::{
    ArrayBuilder, Float32Builder, StringBuilder, UInt16Builder, UInt32Builder,
};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::sync::mpsc::Receiver;

use super::{RequestTimeline, TimelineSummary};

/// Rows buffered before a row group is flushed.
///
/// Row groups are the unit a reader can skip, so they want to be large enough to
/// compress well and small enough that a run killed mid-flight still leaves a
/// readable file.
const ROWS_PER_ROW_GROUP: usize = 65_536;

/// The file's columns.
///
/// `request_id` and `kind` are plain strings rather than integer codes: both are
/// low-cardinality and repeat on every row, so Parquet's dictionary encoding
/// squashes them to near nothing, and a reader opening the file needs no legend
/// to know what it is looking at.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("request_id", DataType::Utf8, false),
        // Position within this request's stream. Written rather than left to a
        // reader's row order, because a Parquet reader may return row groups in
        // any order it likes.
        Field::new("seq", DataType::UInt32, false),
        Field::new("elapsed_ms", DataType::Float32, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("tokens", DataType::UInt16, false),
        Field::new("cumulative_tokens", DataType::UInt32, false),
    ]))
}

#[derive(Default)]
struct Columns {
    request_id: StringBuilder,
    seq: UInt32Builder,
    elapsed_ms: Float32Builder,
    kind: StringBuilder,
    tokens: UInt16Builder,
    cumulative_tokens: UInt32Builder,
}

impl Columns {
    fn push(&mut self, request_id: &str, timeline: &RequestTimeline) {
        for (seq, event) in timeline.events.iter().enumerate() {
            self.request_id.append_value(request_id);
            self.seq.append_value(seq as u32);
            self.elapsed_ms.append_value(event.elapsed_ms);
            self.kind.append_value(event.kind.name());
            self.tokens.append_value(event.tokens);
            self.cumulative_tokens.append_value(event.cumulative_tokens);
        }
    }

    fn len(&self) -> usize {
        self.seq.len()
    }

    fn finish(&mut self, schema: &Arc<Schema>) -> Result<RecordBatch> {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(self.request_id.finish()),
                Arc::new(self.seq.finish()),
                Arc::new(self.elapsed_ms.finish()),
                Arc::new(self.kind.finish()),
                Arc::new(self.tokens.finish()),
                Arc::new(self.cumulative_tokens.finish()),
            ],
        )
        .context("failed to assemble a timeline row group")
    }
}

/// Drain per-request timelines to Parquet until every sender is gone.
///
/// A write failure ends the file rather than the run: the timeline is
/// observability, and losing it must not cost the measurement it describes. What
/// was written stays readable, and the summary reports how much that was.
pub(crate) fn write_timeline(
    path: String,
    receiver: Receiver<RequestTimeline>,
    dropped_requests: Arc<std::sync::atomic::AtomicUsize>,
) -> TimelineSummary {
    let mut summary = TimelineSummary {
        path: Some(path.clone()),
        ..TimelineSummary::default()
    };

    let schema = schema();
    let mut writer = match open(&path, &schema) {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!("warning: timeline disabled: {err:#}");
            summary.path = None;
            // Keep draining: the sink must not start dropping requests (and
            // reporting a lossy run) merely because there is nowhere to write.
            while receiver.recv().is_ok() {}
            summary.dropped_requests = dropped_requests.load(std::sync::atomic::Ordering::Relaxed);
            return summary;
        }
    };

    let mut columns = Columns::default();
    while let Ok(timeline) = receiver.recv() {
        columns.push(&timeline.request_id, &timeline);
        summary.requests_written += 1;
        summary.events_written += timeline.events.len();
        if columns.len() >= ROWS_PER_ROW_GROUP {
            if let Err(err) = flush(&mut writer, &mut columns, &schema) {
                eprintln!("warning: timeline write failed, stopping: {err:#}");
                while receiver.recv().is_ok() {}
                summary.dropped_requests =
                    dropped_requests.load(std::sync::atomic::Ordering::Relaxed);
                return summary;
            }
        }
    }

    if let Err(err) = flush(&mut writer, &mut columns, &schema) {
        eprintln!("warning: timeline final row group failed: {err:#}");
    }
    if let Err(err) = writer.close().context("failed to close the timeline file") {
        eprintln!("warning: {err:#}");
    }

    summary.dropped_requests = dropped_requests.load(std::sync::atomic::Ordering::Relaxed);
    if summary.dropped_requests > 0 {
        eprintln!(
            "warning: {} request timelines were dropped because the timeline writer could not \
             keep up; {path} is a sample of this run, not a record of it",
            summary.dropped_requests,
        );
    }
    summary
}

fn open(path: &str, schema: &Arc<Schema>) -> Result<ArrowWriter<File>> {
    let file =
        File::create(path).with_context(|| format!("failed to create timeline file: {path}"))?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    ArrowWriter::try_new(file, schema.clone(), Some(properties))
        .with_context(|| format!("failed to open timeline writer: {path}"))
}

fn flush(
    writer: &mut ArrowWriter<File>,
    columns: &mut Columns,
    schema: &Arc<Schema>,
) -> Result<()> {
    if columns.len() == 0 {
        return Ok(());
    }
    let batch = columns.finish(schema)?;
    writer
        .write(&batch)
        .context("failed to write a timeline row group")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::{EventKind, TimelineEvent};

    fn event(elapsed_ms: f32, kind: EventKind, tokens: u16, cumulative: u32) -> TimelineEvent {
        TimelineEvent {
            elapsed_ms,
            kind,
            tokens,
            cumulative_tokens: cumulative,
        }
    }

    #[test]
    fn a_four_token_arrival_produces_one_row_that_says_four() {
        // The regression this file exists to prevent: expanding one arrival into
        // one row per token would let a reader treat a single observed instant
        // as four independent measurements.
        let mut columns = Columns::default();
        columns.push(
            "req-1",
            &RequestTimeline {
                request_id: "req-1".to_string(),
                events: vec![event(10.0, EventKind::Tokens, 4, 4)],
            },
        );

        assert_eq!(columns.len(), 1);
        let batch = columns.finish(&schema()).unwrap();
        assert_eq!(batch.num_rows(), 1);
        let tokens = batch
            .column_by_name("tokens")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt16Array>()
            .unwrap();
        assert_eq!(tokens.value(0), 4);
    }

    #[test]
    fn seq_numbers_each_request_from_zero_so_row_order_need_not_be_trusted() {
        let mut columns = Columns::default();
        for request_id in ["req-a", "req-b"] {
            columns.push(
                request_id,
                &RequestTimeline {
                    request_id: request_id.to_string(),
                    events: vec![
                        event(1.0, EventKind::Tokens, 1, 1),
                        event(2.0, EventKind::Tokens, 1, 2),
                        event(3.0, EventKind::Usage, 0, 2),
                    ],
                },
            );
        }

        let batch = columns.finish(&schema()).unwrap();
        let seq = batch
            .column_by_name("seq")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap();
        assert_eq!(
            (0..batch.num_rows())
                .map(|row| seq.value(row))
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 1, 2],
        );
    }

    #[test]
    fn the_writer_reports_what_it_wrote_and_leaves_a_readable_file() {
        let dir =
            std::env::temp_dir().join(format!("req_frontend_timeline_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("timeline.parquet");

        let (sink, receiver) = crate::timeline::channel(16);
        let dropped = sink.dropped_handle();
        let target = path.to_string_lossy().to_string();
        let task = std::thread::spawn(move || write_timeline(target, receiver, dropped));
        sink.offer(RequestTimeline {
            request_id: "req-1".to_string(),
            events: vec![
                event(5.0, EventKind::Tokens, 2, 2),
                event(9.0, EventKind::Usage, 0, 2),
            ],
        });
        drop(sink);

        let summary = task.join().unwrap();
        assert_eq!(summary.requests_written, 1);
        assert_eq!(summary.events_written, 2);
        assert_eq!(summary.dropped_requests, 0);

        // Read it back: a file this run cannot reopen is not observability.
        let file = File::open(&path).unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let rows: usize = reader
            .build()
            .unwrap()
            .map(|batch| batch.unwrap().num_rows())
            .sum();
        assert_eq!(rows, 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
