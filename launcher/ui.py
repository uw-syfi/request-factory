"""Small, stable terminal presentation over machine-readable run artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import LaunchSpec


def heading(specification: LaunchSpec, config_path: Path, build_type: str) -> None:
    width = 72
    print("req-frontend / " + specification.task)
    print("=" * width)
    row("config", str(config_path))
    row("build", build_type)
    row("output", str(specification.output_directory))
    print()


def stage(index: int, count: int, name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"[{index}/{count}] {name}{suffix}", flush=True)


def row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def should_echo_engine_line(task: str, line: str) -> bool:
    """Keep progress and terminal verdicts visible without flooding the UI."""

    stripped = line.strip()
    if not stripped:
        return False
    if task == "run":
        return stripped.startswith(
            (
                "sessions ",
                "requests ",
                "prefix hit rate summary",
                "slo attainment",
            )
        )
    if task == "sweep":
        return stripped.startswith("sweep |")
    if task == "tracegen":
        return stripped.startswith(("session-execution-v2 |", "wrote |"))
    if task == "selfcheck":
        return stripped.startswith(("pass |", "FAIL |", "selfcheck |"))
    return False


def render_result(specification: LaunchSpec) -> None:
    print()
    print("RESULT")
    print("-" * 72)
    if specification.task == "run":
        _render_run(specification)
    elif specification.task == "sweep":
        _render_sweep(specification)
    elif specification.task == "tracegen":
        _render_tracegen(specification)
    elif specification.task == "selfcheck":
        _render_selfcheck(specification)
    print()
    print("ARTIFACTS")
    print("-" * 72)
    _render_artifacts(specification)


def render_failure(specification: LaunchSpec, returncode: int) -> None:
    print()
    print(f"FAILED  engine exited with status {returncode}", file=sys.stderr)
    print(f"  full log      {specification.terminal_log}", file=sys.stderr)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"  report        unavailable ({error})")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _format_number(value: Any, decimals: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{decimals}f}"


def _render_run(specification: LaunchSpec) -> None:
    report = _read_json(specification.result_file)
    workload = report.get("workload", {})
    replay = report.get("replay", {})
    common = replay.get("common", {}) if isinstance(replay, dict) else {}
    kind = workload.get("kind", "unknown") if isinstance(workload, dict) else "unknown"

    if kind == "sessions":
        workload_text = (
            f"{workload.get('sessions', 'n/a')} sessions / "
            f"{workload.get('rounds', 'n/a')} rounds"
        )
    else:
        workload_text = f"{workload.get('requests', 'n/a')} requests"
    row("status", "complete")
    row("workload", workload_text)
    row(
        "success",
        f"{common.get('success_steps', 'n/a')} / {common.get('attempted_steps', 'n/a')} steps",
    )
    row(
        "throughput",
        f"{_format_number(common.get('request_throughput_per_s'))} requests/s  |  "
        f"{_format_number(common.get('output_token_throughput_per_s'))} output tokens/s",
    )
    row(
        "TTFT",
        f"avg {_format_number(common.get('ttft_ms_avg'))} ms  |  "
        f"p90 {_format_number(common.get('ttft_ms_p90'))} ms  |  "
        f"max {_format_number(common.get('ttft_ms_max'))} ms",
    )
    row(
        "TPOT",
        f"avg {_format_number(common.get('tpot_ms_avg'))} ms  |  "
        f"p90 {_format_number(common.get('tpot_ms_p90'))} ms  |  "
        f"max {_format_number(common.get('tpot_ms_max'))} ms",
    )
    prefix_cache = replay.get("prefix_cache") if isinstance(replay, dict) else None
    if isinstance(prefix_cache, dict):
        row(
            "prefix cache",
            f"planned {_format_number(prefix_cache.get('planned_prefix_hit_rate'), 4)}  |  "
            f"measured {_format_number(prefix_cache.get('server_prefix_hit_rate'), 4)}",
        )
    timeline = report.get("timeline")
    if isinstance(timeline, dict):
        row(
            "timeline",
            f"{timeline.get('events_written', 'n/a')} events  |  "
            f"{timeline.get('dropped_requests', 'n/a')} requests dropped",
        )
    slo = report.get("slo")
    row("SLO", "not declared" if slo is None else _slo_text(slo))


def _slo_text(slo: Any) -> str:
    if not isinstance(slo, dict):
        return "unavailable"
    attainment = slo.get("attainment")
    if isinstance(attainment, (int, float)) and not isinstance(attainment, bool):
        return f"{attainment:.2%} attained"
    return "declared; no applicable measurements"


def _render_sweep(specification: LaunchSpec) -> None:
    report = _read_json(specification.result_file)
    config = report.get("config", {})
    points = report.get("points", [])
    row("status", "complete")
    row("mode", str(config.get("mode", report.get("objective", "n/a"))))
    row("points", str(len(points) if isinstance(points, list) else "n/a"))
    knee = report.get("knee")
    peak = report.get("peak")
    if knee is not None:
        row("knee", json.dumps(knee, separators=(",", ":")))
    if peak is not None:
        row("peak", json.dumps(peak, separators=(",", ":")))
    warning = report.get("contamination_warning")
    if warning:
        row("warning", str(warning))


def _render_tracegen(specification: LaunchSpec) -> None:
    trace_path = specification.result_file
    manifest_path = (
        trace_path.with_name(f"{trace_path.stem}.manifest.json") if trace_path else None
    )
    manifest = _read_json(manifest_path)
    row("status", "complete")
    row("generator", str(manifest.get("generator", "n/a")))
    row("sessions", str(manifest.get("sessions", "n/a")))
    row("rounds", str(manifest.get("rounds", "n/a")))
    row("prompt tokens", str(manifest.get("total_prompt_tokens", "n/a")))
    row(
        "planned cache",
        _format_number(manifest.get("planned_prefix_hit_rate"), 4),
    )


def _render_selfcheck(specification: LaunchSpec) -> None:
    report = _read_json(specification.result_file)
    passed = report.get("passed", "n/a")
    failed = report.get("failed", "n/a")
    row("status", "passed" if failed == 0 else "failed")
    row("checks", f"{passed} passed / {failed} failed")
    row("log schema", str(report.get("step_log_schema_version", "n/a")))


def _render_artifacts(specification: LaunchSpec) -> None:
    if specification.output_directory.exists():
        files = sorted(
            path for path in specification.output_directory.iterdir() if path.is_file()
        )
        for path in files:
            row(path.name, str(path))
    else:
        row("directory", str(specification.output_directory))
    if (
        specification.terminal_log.exists()
        and specification.terminal_log.parent != specification.output_directory
    ):
        row("terminal.log", str(specification.terminal_log))
