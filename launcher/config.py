"""Strict YAML loading and lowering to Request Factory's internal Rust argv.

The YAML tree is the human-facing contract. This module owns its grouping and
path resolution, but deliberately does not implement replay semantics: after
validation it produces argv for the Rust binary that remains authoritative for
format parsing, workload construction, execution, and measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# Wire vocabularies the engine can speak. Kept in lockstep with
# src/backend/dialect/profiles.rs::ALL -- the launcher validates the name so an
# operator typo fails before a build, not after one.
# Media surfaces the engine can drive. Which of them a given server actually
# exposes is a dialect property, checked in Rust against the dialect table.
MULTIMODAL_BACKENDS = frozenset(
    {
        "openai-chat",
        "openai-images",
        "openai-image-edits",
        "openai-videos",
        "openai-speech",
        "openai-transcriptions",
        "openai-translations",
        "openai-realtime",
    }
)

KNOWN_DIALECTS = frozenset(
    {"openai", "vllm", "vllm-omni", "sglang-omni", "mstar", "dynamo"}
)


class ConfigError(ValueError):
    """A launcher config is malformed or internally contradictory."""


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigError(f"mapping key must be a string, got {key!r}")
        if key in mapping:
            raise ConfigError(f"duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class LaunchSpec:
    task: str
    binary: str
    arguments: tuple[str, ...]
    output_directory: Path
    terminal_log: Path
    result_file: Path | None
    postprocess_viz: bool = False
    processes: int = 1


def load_task_config(task: str, config_path: Path) -> LaunchSpec:
    """Load one task config and lower it to a concrete launch specification."""

    try:
        loaded = yaml.load(config_path.read_text(), Loader=_StrictLoader)
    except OSError as error:
        raise ConfigError(f"cannot read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError("top-level config must be a mapping")

    builders = {
        "run": _build_run,
        "sweep": _build_sweep,
        "tracegen": _build_tracegen,
        "selfcheck": _build_selfcheck,
    }
    builder = builders.get(task)
    if builder is None:
        raise ConfigError(
            f"unknown task {task!r}; choose one of: {', '.join(builders)}"
        )
    return builder(loaded, config_path.resolve())


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        rendered = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"unknown config key(s): {rendered}")


def _block(
    mapping: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    value = mapping.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        requirement = "required " if required else ""
        raise ConfigError(f"{requirement}{name!r} must be a mapping")
    return value


def _required_string(mapping: dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_string(mapping: dict[str, Any], key: str, path: str, default: str) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_bool(mapping: dict[str, Any], key: str, path: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be a boolean")
    return value


def _optional_number(
    mapping: dict[str, Any],
    key: str,
    path: str,
    default: float | None = None,
    *,
    positive: bool = False,
) -> int | float | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number")
    if positive and value <= 0:
        raise ConfigError(f"{path}.{key} must be greater than zero")
    return value


def _optional_integer(
    mapping: dict[str, Any],
    key: str,
    path: str,
    default: int | None = None,
    *,
    positive: bool = False,
) -> int | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key} must be an integer")
    if positive and value <= 0:
        raise ConfigError(f"{path}.{key} must be greater than zero")
    return value


def _distribution_value(
    mapping: dict[str, Any], key: str, path: str, default: str
) -> str:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(
            f"{path}.{key} must be a positive integer or distribution string"
        )
    if isinstance(value, int) and value <= 0:
        raise ConfigError(f"{path}.{key} must be greater than zero")
    if isinstance(value, str) and not value:
        raise ConfigError(f"{path}.{key} must not be empty")
    return str(value)


def _probability(mapping: dict[str, Any], key: str, path: str, default: float) -> float:
    value = _optional_number(mapping, key, path, default)
    assert value is not None
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path}.{key} must be between 0 and 1")
    return float(value)


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_tokenizer(value: str, config_path: Path) -> str:
    """Resolve explicit/local tokenizer paths while preserving Hub repo ids."""

    candidate = Path(value).expanduser()
    relative_candidate = config_path.parent / candidate
    if candidate.is_absolute():
        return str(candidate)
    if value.startswith(".") or relative_candidate.exists():
        return str(relative_candidate.resolve())
    return value


def _append_option(arguments: list[str], name: str, value: Any | None) -> None:
    if value is not None:
        arguments.extend([name, str(value)])


def _output_paths(
    root: dict[str, Any], config_path: Path, *, default_directory: str
) -> tuple[Path, dict[str, Path]]:
    output = _block(root, "output")
    _reject_unknown(
        output,
        {"directory", "requests", "summary", "timeline", "terminal", "artifacts"},
        "output",
    )
    directory = _resolve_path(
        _optional_string(output, "directory", "output", default_directory),
        config_path,
    )
    filenames = {
        "requests": _optional_string(output, "requests", "output", "requests.jsonl"),
        "summary": _optional_string(output, "summary", "output", "summary.json"),
        "timeline": _optional_string(output, "timeline", "output", "timeline.parquet"),
        "terminal": _optional_string(output, "terminal", "output", "terminal.log"),
        "artifacts": _optional_string(output, "artifacts", "output", "artifacts"),
    }
    for name, filename in filenames.items():
        if Path(filename).name != filename:
            raise ConfigError(f"output.{name} must be a single path component")
    return directory, {
        name: directory / filename for name, filename in filenames.items()
    }


def _build_common_run_arguments(
    root: dict[str, Any],
    config_path: Path,
    paths: dict[str, Path],
) -> list[str]:
    input_config = _block(root, "input")
    corpus = _block(root, "corpus", required=False)
    server = _block(root, "server")
    replay = _block(root, "replay", required=False)
    measurement = _block(root, "measurement", required=False)

    _reject_unknown(input_config, {"trace", "format", "tags"}, "input")
    _reject_unknown(corpus, {"text_file", "tokenizer", "token_pool_limit"}, "corpus")
    _reject_unknown(
        server, {"base_url", "backend", "dialect", "model", "temperature"}, "server"
    )
    _reject_unknown(
        replay,
        {
            "arrival_mode",
            "rate",
            "max_items",
            "max_concurrency",
            "processes",
            "runtime_worker_threads",
            "stream_idle_timeout_seconds",
            "stop_session_on_error",
            "dry_run",
            "context",
        },
        "replay",
    )
    _reject_unknown(measurement, {"timeline", "request_log", "slo"}, "measurement")

    tags = input_config.get("tags", [])
    if not isinstance(tags, list) or not all(
        isinstance(tag, str) and tag for tag in tags
    ):
        raise ConfigError("input.tags must be a list of non-empty strings")

    context = replay.get("context", {})
    if not isinstance(context, dict):
        raise ConfigError("replay.context must be a mapping")
    _reject_unknown(context, {"max_model_len", "on_limit"}, "replay.context")
    on_limit = _optional_string(context, "on_limit", "replay.context", "send")
    if on_limit not in {"send", "skip"}:
        raise ConfigError("replay.context.on_limit must be 'send' or 'skip'")

    timeline_enabled = _optional_bool(measurement, "timeline", "measurement", True)
    slo = measurement.get("slo")
    if slo is not None and not isinstance(slo, dict):
        raise ConfigError("measurement.slo must be a mapping")
    slo = slo or {}
    _reject_unknown(slo, {"ttft_ms", "tpot_ms", "e2e_ms"}, "measurement.slo")
    slo_parts: list[str] = []
    for metric_name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        threshold = _optional_number(slo, metric_name, "measurement.slo", positive=True)
        if threshold is not None:
            slo_parts.append(f"{metric_name}={threshold}")

    trace = _resolve_path(_required_string(input_config, "trace", "input"), config_path)
    input_format = _required_string(input_config, "format", "input")
    backend = _optional_string(server, "backend", "server", "openai")
    dialect = _optional_string(server, "dialect", "server", "openai")
    if dialect not in KNOWN_DIALECTS:
        raise ConfigError(
            f"server.dialect must be one of {sorted(KNOWN_DIALECTS)}, got {dialect!r}"
        )
    is_multimodal = input_format == "multimodal-independent-v1"
    if is_multimodal:
        if corpus:
            raise ConfigError(
                "corpus is text-replay-only and must be omitted for multimodal-independent-v1"
            )
        if backend not in MULTIMODAL_BACKENDS:
            raise ConfigError(
                "multimodal-independent-v1 requires server.backend one of "
                + ", ".join(sorted(MULTIMODAL_BACKENDS))
            )
        if context:
            raise ConfigError(
                "replay.context is text-replay-only and must be omitted for multimodal-independent-v1"
            )
    elif not corpus:
        raise ConfigError("required 'corpus' must be a mapping for text replay")

    arguments = [
        "--trace",
        str(trace),
        "--input-file-format",
        input_format,
        "--model",
        _required_string(server, "model", "server"),
        "--backend",
        backend,
        "--dialect",
        dialect,
        "--base-url",
        _optional_string(server, "base_url", "server", "http://127.0.0.1:8000/v1"),
        "--temperature",
        str(_optional_number(server, "temperature", "server", 0.0)),
        "--arrival-mode",
        _optional_string(replay, "arrival_mode", "replay", "trace-timed"),
        "--stream-idle-timeout-secs",
        str(
            _optional_integer(
                replay,
                "stream_idle_timeout_seconds",
                "replay",
                600,
                positive=True,
            )
        ),
        "--stop-session-on-error",
        str(_optional_bool(replay, "stop_session_on_error", "replay", True)).lower(),
        "--timeline",
        str(timeline_enabled).lower(),
        "--request-log",
        str(_optional_bool(measurement, "request_log", "measurement", True)).lower(),
        "--log-path",
        str(paths["requests"]),
        "--summary-path",
        str(paths["summary"]),
        "--timeline-path",
        str(paths["timeline"]),
    ]
    if not is_multimodal:
        text_file = _resolve_path(
            _required_string(corpus, "text_file", "corpus"), config_path
        )
        tokenizer = _resolve_tokenizer(
            _required_string(corpus, "tokenizer", "corpus"), config_path
        )
        arguments[4:4] = [
            "--text-file",
            str(text_file),
            "--tokenizer",
            tokenizer,
        ]
    else:
        arguments.extend(["--output-artifact-dir", str(paths["artifacts"])])
    if tags:
        arguments.extend(["--trace-tags", ",".join(tags)])
    _append_option(
        arguments,
        "--token-pool-limit",
        _optional_integer(corpus, "token_pool_limit", "corpus", positive=True),
    )
    _append_option(
        arguments,
        "--max-items",
        _optional_integer(replay, "max_items", "replay", positive=True),
    )
    _append_option(
        arguments,
        "--rate",
        _optional_number(replay, "rate", "replay", positive=True),
    )
    _append_option(
        arguments,
        "--max-concurrency",
        _optional_integer(replay, "max_concurrency", "replay", positive=True),
    )
    _append_option(
        arguments,
        "--runtime-worker-threads",
        _optional_integer(replay, "runtime_worker_threads", "replay", positive=True),
    )
    _append_option(
        arguments,
        "--max-model-len",
        _optional_integer(context, "max_model_len", "replay.context", positive=True),
    )
    if on_limit == "skip":
        arguments.append("--skip-when-reaching-limit")
    if _optional_bool(replay, "dry_run", "replay", False):
        arguments.append("--dry-run")
    if slo_parts:
        arguments.extend(["--slo", ",".join(slo_parts)])
    return arguments


def _build_run(root: dict[str, Any], config_path: Path) -> LaunchSpec:
    _reject_unknown(
        root,
        {"input", "corpus", "server", "replay", "measurement", "output"},
        "config",
    )
    output_directory, paths = _output_paths(
        root, config_path, default_directory="out/run"
    )
    arguments = _build_common_run_arguments(root, config_path, paths)
    replay = _block(root, "replay", required=False)
    processes = _optional_integer(replay, "processes", "replay", 1, positive=True)
    assert processes is not None
    if processes > 1 and replay.get("arrival_mode", "trace-timed") != "saturated":
        raise ConfigError(
            "replay.processes greater than 1 currently requires replay.arrival_mode: saturated"
        )
    return LaunchSpec(
        task="run",
        binary="session_runner",
        arguments=tuple(arguments),
        output_directory=output_directory,
        terminal_log=paths["terminal"],
        result_file=paths["summary"],
        processes=processes,
    )


def _build_sweep(root: dict[str, Any], config_path: Path) -> LaunchSpec:
    _reject_unknown(
        root,
        {
            "search",
            "input",
            "corpus",
            "server",
            "replay",
            "measurement",
            "output",
            "visualization",
        },
        "config",
    )
    output_directory, paths = _output_paths(
        root, config_path, default_directory="out/sweep"
    )
    search = _block(root, "search")
    _reject_unknown(
        search,
        {
            "mode",
            "start_rate",
            "max_rate",
            "min_rate",
            "tolerance",
            "densify_points",
            "max_shortfall",
            "min_gain",
            "patience",
            "plateau_tolerance",
            "peak_metric",
            "target_attainment",
            "attainment_metric",
            "rates",
            "resume",
        },
        "search",
    )
    replay = _block(root, "replay", required=False)
    if replay.get("rate") is not None:
        raise ConfigError(
            "replay.rate is controlled by search and must be omitted for sweep"
        )
    if replay.get("dry_run"):
        raise ConfigError(
            "replay.dry_run cannot measure a sweep and must be false or omitted"
        )
    if replay.get("processes", 1) != 1:
        raise ConfigError("replay.processes is supported by run, not sweep")

    arguments = [
        "--out",
        str(output_directory),
        "--mode",
        _optional_string(search, "mode", "search", "max-sustainable-rate"),
        "--start-rate",
        str(_optional_number(search, "start_rate", "search", 1.0, positive=True)),
        "--max-rate",
        str(_optional_number(search, "max_rate", "search", 4096.0, positive=True)),
        "--min-rate",
        str(_optional_number(search, "min_rate", "search", 0.001, positive=True)),
        "--tolerance",
        str(_optional_number(search, "tolerance", "search", 0.05, positive=True)),
        "--densify-points",
        str(_optional_integer(search, "densify_points", "search", 3)),
        "--max-shortfall",
        str(_optional_number(search, "max_shortfall", "search", 0.1)),
        "--min-gain",
        str(_optional_number(search, "min_gain", "search", 0.03)),
        "--patience",
        str(_optional_integer(search, "patience", "search", 2, positive=True)),
        "--plateau-tolerance",
        str(_optional_number(search, "plateau_tolerance", "search", 0.02)),
        "--peak-metric",
        _optional_string(search, "peak_metric", "search", "output-tokens"),
        "--target-attainment",
        str(_optional_number(search, "target_attainment", "search", 0.99)),
        "--attainment-metric",
        _optional_string(search, "attainment_metric", "search", "overall"),
    ]
    rates = search.get("rates", [])
    if not isinstance(rates, list) or not all(
        isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0
        for rate in rates
    ):
        raise ConfigError("search.rates must be a list of positive numbers")
    if rates:
        arguments.extend(["--rates", ",".join(str(rate) for rate in rates)])
    if not _optional_bool(search, "resume", "search", True):
        arguments.append("--no-resume")
    arguments.extend(_build_common_run_arguments(root, config_path, paths))

    visualization = _block(root, "visualization", required=False)
    _reject_unknown(visualization, {"enabled"}, "visualization")
    return LaunchSpec(
        task="sweep",
        binary="sweep",
        arguments=tuple(arguments),
        output_directory=output_directory,
        terminal_log=paths["terminal"],
        result_file=output_directory / "sweep.json",
        postprocess_viz=_optional_bool(
            visualization, "enabled", "visualization", False
        ),
    )


def _build_tracegen(root: dict[str, Any], config_path: Path) -> LaunchSpec:
    _reject_unknown(root, {"generator", "output"}, "config")
    generator = _block(root, "generator")
    output = _block(root, "output")
    _reject_unknown(output, {"trace", "terminal"}, "output")
    output_trace = _resolve_path(
        _required_string(output, "trace", "output"), config_path
    )
    terminal_name = _optional_string(output, "terminal", "output", "terminal.log")
    if Path(terminal_name).name != terminal_name:
        raise ConfigError("output.terminal must be a filename, not a path")
    generator_type = _required_string(generator, "type", "generator")
    arguments = [generator_type, "--out", str(output_trace)]

    if generator_type == "synthetic":
        allowed = {
            "type",
            "sessions",
            "rounds",
            "input_len",
            "output_len",
            "tool_wait_ms",
            "compaction_probability",
            "arrival_rate",
            "arrival_pattern",
            "seed",
        }
        _reject_unknown(generator, allowed, "generator")
        arguments.extend(
            [
                "--sessions",
                str(
                    _optional_integer(
                        generator, "sessions", "generator", 100, positive=True
                    )
                ),
                "--rounds",
                _distribution_value(generator, "rounds", "generator", "uniform:1..8"),
                "--input-len",
                _distribution_value(
                    generator, "input_len", "generator", "lognormal:1024,0.8"
                ),
                "--output-len",
                _distribution_value(
                    generator, "output_len", "generator", "lognormal:256,0.7"
                ),
                "--tool-wait-ms",
                _distribution_value(
                    generator, "tool_wait_ms", "generator", "lognormal:500,1.0"
                ),
                "--compaction-probability",
                str(
                    _probability(generator, "compaction_probability", "generator", 0.0)
                ),
                "--arrival-rate",
                str(
                    _optional_number(
                        generator, "arrival_rate", "generator", 1.0, positive=True
                    )
                ),
                "--arrival-pattern",
                _optional_string(generator, "arrival_pattern", "generator", "poisson"),
                "--seed",
                str(_optional_integer(generator, "seed", "generator", 0)),
            ]
        )
    elif generator_type == "coding-session":
        allowed = {
            "type",
            "source",
            "source_schema",
            "policy",
            "max_sessions",
            "session_order",
            "arrival_rate",
            "arrival_pattern",
            "seed",
        }
        _reject_unknown(generator, allowed, "generator")
        source = _resolve_path(
            _required_string(generator, "source", "generator"), config_path
        )
        arguments.extend(
            [
                "--source",
                str(source),
                "--source-schema",
                _optional_string(
                    generator,
                    "source_schema",
                    "generator",
                    "session-rounds-v2",
                ),
                "--policy",
                _optional_string(generator, "policy", "generator", "trace-reported"),
                "--session-order",
                _optional_string(generator, "session_order", "generator", "source"),
                "--arrival-rate",
                str(
                    _optional_number(
                        generator, "arrival_rate", "generator", 1.0, positive=True
                    )
                ),
                "--arrival-pattern",
                _optional_string(generator, "arrival_pattern", "generator", "poisson"),
                "--seed",
                str(_optional_integer(generator, "seed", "generator", 0)),
            ]
        )
        _append_option(
            arguments,
            "--max-sessions",
            _optional_integer(generator, "max_sessions", "generator", positive=True),
        )
    else:
        raise ConfigError("generator.type must be 'synthetic' or 'coding-session'")

    return LaunchSpec(
        task="tracegen",
        binary="tracegen",
        arguments=tuple(arguments),
        output_directory=output_trace.parent,
        terminal_log=output_trace.parent / terminal_name,
        result_file=output_trace,
    )


def _build_selfcheck(root: dict[str, Any], config_path: Path) -> LaunchSpec:
    _reject_unknown(root, {"tokenizer", "checks", "output"}, "config")
    tokenizer_block = _block(root, "tokenizer")
    checks = _block(root, "checks", required=False)
    output = _block(root, "output")
    _reject_unknown(tokenizer_block, {"path"}, "tokenizer")
    _reject_unknown(checks, {"pairs", "port"}, "checks")
    _reject_unknown(output, {"directory", "terminal"}, "output")
    output_directory = _resolve_path(
        _optional_string(output, "directory", "output", "out/selfcheck"),
        config_path,
    )
    terminal_name = _optional_string(output, "terminal", "output", "terminal.log")
    if Path(terminal_name).name != terminal_name:
        raise ConfigError("output.terminal must be a filename, not a path")
    arguments = [
        "--tokenizer",
        _resolve_tokenizer(
            _required_string(tokenizer_block, "path", "tokenizer"), config_path
        ),
        "--out",
        str(output_directory),
        "--pairs",
        str(_optional_integer(checks, "pairs", "checks", 2, positive=True)),
        "--port",
        str(_optional_integer(checks, "port", "checks", 8271, positive=True)),
    ]
    return LaunchSpec(
        task="selfcheck",
        binary="selfcheck",
        arguments=tuple(arguments),
        output_directory=output_directory,
        terminal_log=output_directory / terminal_name,
        result_file=output_directory / "selfcheck.json",
    )
