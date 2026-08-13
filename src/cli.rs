use clap::{Parser, ValueEnum};

use crate::trace::TraceFormat;

/// Inference-server wire protocol selected with `--backend`.
/// Which of the two release axes supplies a unit's start time.
///
/// The other axis is `--max-concurrency`. They compose freely: a capped
/// trace-timed run replays the recorded timeline but never exceeds the cap, and
/// a capped saturated run is a pure closed-loop generator.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrivalMode {
    /// Replay the trace's own arrival timeline, rescaled by `--rate`.
    #[value(name = "trace-timed")]
    TraceTimed,
    /// Ignore recorded arrivals: every unit is eligible from the start, so a
    /// unit enters as soon as capacity allows. Without a cap this submits the
    /// whole workload at once.
    Saturated,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// OpenAI-compatible `/completions` (vLLM, and SGLang's OpenAI endpoint).
    Openai,
    /// vLLM native token-in/token-out `/inference/v1/generate` endpoint.
    VllmTokens,
    /// SGLang native token-in/token-out `/generate` endpoint.
    SglangTokens,
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Typed trace workload runner for OpenAI-compatible inference servers"
)]
pub struct Args {
    /// Source trace CSV interpreted by --trace-format.
    #[arg(long)]
    pub trace: String,

    /// Input schema frontend. New source formats are separate typed frontends,
    /// not sparse variants of one universal row. `session` reads a canonical
    /// execution trace; generate one from a raw CSV with `tracegen`.
    #[arg(long, value_enum, default_value = "session")]
    pub trace_format: TraceFormat,

    /// Text corpus used to build synthetic prompt/input/output token pools.
    #[arg(long)]
    pub text_file: String,

    /// tokenizer.json path or a model directory containing tokenizer.json.
    #[arg(long)]
    pub tokenizer: String,

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
