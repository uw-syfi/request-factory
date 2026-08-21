use clap::{Parser, ValueEnum};

use crate::release::ArrivalMode;
use crate::schema::InputFileFormat;

/// Inference-server wire protocol selected with `--backend`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// OpenAI-compatible `/completions` (vLLM, and SGLang's OpenAI endpoint).
    Openai,
    /// vLLM native token-in/token-out `/inference/v1/generate` endpoint.
    VllmTokens,
    /// SGLang native token-in/token-out `/generate` endpoint.
    SglangTokens,
    /// OpenAI-compatible `/chat/completions` with interleaved text/media input.
    OpenaiChat,
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Typed trace workload runner for OpenAI-compatible inference servers"
)]
pub struct Args {
    /// Source CSV interpreted by --input-file-format.
    #[arg(long)]
    pub trace: String,

    /// Complete input-file format. It determines the request family, exact base
    /// columns, loader, and structural validation rules; no separate family
    /// selector exists.
    #[arg(
        long,
        value_parser = clap::builder::PossibleValuesParser::new(InputFileFormat::CHOICES),
        default_value = "text-generation-session-execution-v2"
    )]
    pub input_file_format: String,

    /// Orthogonal declarations this file carries, comma-separated: `slo` adds
    /// per-request TTFT/TPOT/E2E bounds; `priority` adds scheduling priority.
    ///
    /// A tag is how a trace says it carries more than its format's own columns.
    /// Undeclared extra columns are an error, not something to ignore: a column
    /// nobody reads is data whose author expected it to matter.
    #[arg(long, value_delimiter = ',')]
    pub trace_tags: Vec<String>,

    /// Text corpus used to build synthetic prompt/input/output token pools.
    #[arg(long)]
    pub text_file: Option<String>,

    /// tokenizer.json path or a model directory containing tokenizer.json.
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Protocol base URL. Use http://host:port/v1 for openai and
    /// http://host:port for the native token endpoints.
    #[arg(long, default_value = "http://127.0.0.1:8000/v1")]
    pub base_url: String,

    /// Model name sent in the request payload. Ignored by `sglang-tokens`,
    /// whose server hosts exactly one model and takes no model field.
    #[arg(long)]
    pub model: String,

    /// Inference-server wire protocol. `vllm-tokens` requires vLLM
    /// --tokens-only; `sglang-tokens` requires SGLang --skip-tokenizer-init
    /// and --stream-output (renamed --incremental-streaming-output in newer
    /// builds).
    #[arg(long, value_enum, default_value = "openai")]
    pub backend: BackendKind,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,

    /// Limit top-level workload units (sessions or independent requests).
    #[arg(long, alias = "max-sessions")]
    pub max_items: Option<usize>,

    /// Target top-level arrival rate: sessions/s or independent requests/s.
    /// Only meaningful under `--arrival-mode trace-timed`.
    #[arg(long)]
    pub rate: Option<f64>,

    /// Where release times come from. Orthogonal to `--max-concurrency`:
    /// arrival says when a unit *may* start, capacity says how many may run.
    #[arg(long, value_enum, default_value = "trace-timed")]
    pub arrival_mode: ArrivalMode,

    #[arg(long, default_value = "session_runner_output.jsonl")]
    pub log_path: String,

    /// Cap on synthetic token-pool size. Defaults to cover the workload's longest prompt with
    /// headroom, so synthetic content never repeats within a single request.
    #[arg(long)]
    pub token_pool_limit: Option<usize>,

    /// Max seconds to wait for the next streaming chunk before failing a request.
    #[arg(long, default_value_t = 600)]
    pub stream_idle_timeout_secs: u64,

    /// Stop a session after the first failed round.
    ///
    /// `ArgAction::Set` rather than a bare flag: with a `true` default, a
    /// set-true flag can only restate the default, which made this knob
    /// impossible to turn off. Write `--stop-session-on-error false`.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub stop_session_on_error: bool,

    /// Maximum number of top-level workload units active at once.
    #[arg(long)]
    pub max_concurrency: Option<usize>,

    /// Parse and statically summarize the workload without loading tokens or contacting a server.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Optional model context limit used for static trace-target overflow reporting.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Skip a request when prompt plus target output reaches the model limit,
    /// reserving at least one token of context headroom.
    #[arg(
        long,
        visible_aliases = ["skip-on-context-limit", "fail-on-context-overflow"],
        default_value_t = false
    )]
    pub skip_when_reaching_limit: bool,

    /// Optional JSON summary path for one run.
    #[arg(long)]
    pub summary_path: Option<String>,

    /// Record when every streamed event arrived, per request, to Parquet.
    ///
    /// On by default: the measurement is a by-product of the fold that already
    /// times each event, and the write path is arranged so it cannot slow
    /// submission. Disable with `--timeline false`.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub timeline: bool,

    /// Where the per-event timeline is written.
    #[arg(long, default_value = "session_runner_timeline.parquet")]
    pub timeline_path: String,

    /// Service-level objective for this run: `ttft_ms=500,tpot_ms=50`.
    ///
    /// Per-metric upper bounds. The summary reports the fraction of steps that
    /// met every declared bound, and the fraction that met each one.
    ///
    /// Overrides an objective the trace declares in its `.slo.json` sidecar.
    #[arg(long)]
    pub slo: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_arguments() -> Vec<&'static str> {
        vec![
            "session_runner",
            "--trace",
            "trace.csv",
            "--text-file",
            "corpus.txt",
            "--tokenizer",
            "tokenizer.json",
            "--model",
            "model",
        ]
    }

    #[test]
    fn a_complete_input_file_format_is_the_only_schema_selector() {
        let mut arguments = required_arguments();
        arguments.extend(["--input-file-format", "image-to-text-independent"]);

        let parsed = Args::try_parse_from(arguments).unwrap();
        assert_eq!(parsed.input_file_format, "image-to-text-independent");
    }

    #[test]
    fn retired_split_format_and_family_flags_are_rejected() {
        let mut old_format = required_arguments();
        old_format.extend(["--trace-format", "session"]);
        assert!(Args::try_parse_from(old_format).is_err());

        let mut old_family = required_arguments();
        old_family.extend(["--trace-kind", "text_generation"]);
        assert!(Args::try_parse_from(old_family).is_err());
    }
}
