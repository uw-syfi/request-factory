use clap::{Parser, ValueEnum};

use tracelab_replay::policy::SessionContextPolicy;

use crate::trace::TraceFormat;

/// Inference-server wire protocol selected with `--backend`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackendKind {
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
pub(crate) struct Args {
    /// Source trace CSV interpreted by --trace-format.
    #[arg(long)]
    pub(crate) trace: String,

    /// Input schema frontend. New source formats are separate typed frontends,
    /// not sparse variants of the session CSV row.
    #[arg(long, value_enum, default_value = "session")]
    pub(crate) trace_format: TraceFormat,

    /// Text corpus used to build synthetic prompt/input/output token pools.
    #[arg(long)]
    pub(crate) text_file: String,

    /// tokenizer.json path or a model directory containing tokenizer.json.
    #[arg(long)]
    pub(crate) tokenizer: String,

    /// Protocol base URL. Use http://host:port/v1 for openai and
    /// http://host:port for the native token endpoints.
    #[arg(long, default_value = "http://127.0.0.1:8000/v1")]
    pub(crate) base_url: String,

    /// Model name sent in the request payload. Ignored by `sglang-tokens`,
    /// whose server hosts exactly one model and takes no model field.
    #[arg(long)]
    pub(crate) model: String,

    /// Inference-server wire protocol. `vllm-tokens` requires vLLM
    /// --tokens-only; `sglang-tokens` requires SGLang --skip-tokenizer-init
    /// and --stream-output (renamed --incremental-streaming-output in newer
    /// builds).
    #[arg(long, value_enum, default_value = "openai")]
    pub(crate) backend: BackendKind,

    /// How a raw session trace's reported prefix becomes a replayable one.
    /// Rejected with `--trace-format session-execution-v2`, whose policy was
    /// already resolved when the trace was generated.
    #[arg(long, value_enum, default_value = "trace-reported")]
    pub(crate) session_context_policy: SessionContextPolicy,

    #[arg(long, default_value_t = 0.0)]
    pub(crate) temperature: f64,

    /// Limit top-level workload units (sessions or independent requests).
    #[arg(long, alias = "max-sessions")]
    pub(crate) max_items: Option<usize>,

    /// Target top-level arrival rate: sessions/s or independent requests/s.
    #[arg(long)]
    pub(crate) rate: Option<f64>,

    #[arg(long, default_value = "session_runner_output.jsonl")]
    pub(crate) log_path: String,

    /// Cap on synthetic token-pool size. Defaults to cover the workload's longest prompt with
    /// headroom, so synthetic content never repeats within a single request.
    #[arg(long)]
    pub(crate) token_pool_limit: Option<usize>,

    /// Max seconds to wait for the next streaming chunk before failing a request.
    #[arg(long, default_value_t = 600)]
    pub(crate) stream_idle_timeout_secs: u64,

    /// Stop a session after the first failed round.
    #[arg(long, default_value_t = true)]
    pub(crate) stop_session_on_error: bool,

    /// Maximum number of top-level workload units active at once.
    #[arg(long)]
    pub(crate) max_concurrency: Option<usize>,

    /// Parse and statically summarize the workload without loading tokens or contacting a server.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,

    /// Optional model context limit used for static trace-target overflow reporting.
    #[arg(long)]
    pub(crate) max_model_len: Option<usize>,

    /// Skip a request when prompt plus target output reaches the model limit,
    /// reserving at least one token of context headroom.
    #[arg(
        long,
        visible_aliases = ["skip-on-context-limit", "fail-on-context-overflow"],
        default_value_t = false
    )]
    pub(crate) skip_when_reaching_limit: bool,

    /// Optional JSON summary path for one run.
    #[arg(long)]
    pub(crate) summary_path: Option<String>,
}
