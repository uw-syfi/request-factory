"""Task-oriented YAML launcher for every Request Factory execution mode."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import ui
from .config import REPO_ROOT, ConfigError, LaunchSpec, load_task_config

TASKS = ("run", "sweep", "tracegen", "selfcheck")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m launcher",
        description="Run Request Factory tasks from strict, structured YAML.",
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("config", type=Path, help="Task YAML path")
    parser.add_argument(
        "--build-type",
        default="release",
        choices=("release", "debug"),
        help="Cargo profile used for the Rust binary (default: release)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved command without building or running",
    )
    parser.add_argument(
        "--show-engine-output",
        action="store_true",
        help="Stream every Rust output line instead of the concise task view",
    )
    return parser


def _binary_path(specification: LaunchSpec, build_type: str) -> Path:
    return REPO_ROOT / "target" / build_type / specification.binary


def _build(specification: LaunchSpec, build_type: str) -> bool:
    command = ["cargo", "build", "--bin", specification.binary]
    if build_type == "release":
        command.append("--release")
    build_log = specification.output_directory / "build.log"
    with build_log.open("w") as writer:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=writer,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(f"  failed; see {build_log}", file=sys.stderr)
        _print_tail(build_log)
        return False
    print(f"  ready  {_binary_path(specification, build_type)}")
    return True


def _snapshot_invocation(
    specification: LaunchSpec,
    config_path: Path,
    command: list[str],
) -> None:
    snapshot_path = specification.output_directory / "launcher-config.yaml"
    if config_path.resolve() != snapshot_path.resolve():
        shutil.copyfile(config_path, snapshot_path)
    (specification.output_directory / "command.txt").write_text(
        shlex.join(command) + "\n"
    )


def _execute(
    specification: LaunchSpec,
    command: list[str],
    *,
    show_engine_output: bool,
) -> int:
    with specification.terminal_log.open("w") as writer:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            writer.write(line)
            writer.flush()
            if show_engine_output or ui.should_echo_engine_line(
                specification.task, line
            ):
                print(f"  {line.rstrip()}", flush=True)
        return process.wait()


def _shard_command(
    specification: LaunchSpec, command: list[str], shard_index: int
) -> tuple[list[str], Path, Path]:
    shard_directory = specification.output_directory / "shards" / f"{shard_index:02d}"
    shard_directory.mkdir(parents=True, exist_ok=True)
    rewritten = list(command)
    output_options = {
        "--log-path": "requests.jsonl",
        "--summary-path": "summary.json",
        "--timeline-path": "timeline.parquet",
        "--output-artifact-dir": "artifacts",
    }
    for option, filename in output_options.items():
        if option in rewritten:
            rewritten[rewritten.index(option) + 1] = str(shard_directory / filename)
    rewritten.extend(
        [
            "--shard-count",
            str(specification.processes),
            "--shard-index",
            str(shard_index),
        ]
    )
    return rewritten, shard_directory / "terminal.log", shard_directory / "summary.json"


def _execute_sharded(
    specification: LaunchSpec,
    command: list[str],
    *,
    show_engine_output: bool,
) -> int:
    started = time.monotonic()
    running: list[tuple[subprocess.Popen[str], object, Path, Path]] = []
    commands: list[str] = []
    for shard_index in range(specification.processes):
        shard_command, terminal_log, summary_path = _shard_command(
            specification, command, shard_index
        )
        commands.append(shlex.join(shard_command))
        writer = terminal_log.open("w")
        process = subprocess.Popen(
            shard_command,
            cwd=REPO_ROOT,
            stdout=writer,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((process, writer, terminal_log, summary_path))

    returncodes: list[int] = []
    for process, writer, terminal_log, _ in running:
        returncodes.append(process.wait())
        writer.close()  # type: ignore[attr-defined]
        if show_engine_output:
            print(f"\n[{terminal_log.parent.name}] {terminal_log}")
            _print_tail(terminal_log, lines=100)

    specification.terminal_log.write_text("\n".join(commands) + "\n")
    failed = next((code for code in returncodes if code != 0), 0)
    if failed:
        return failed
    _write_sharded_summary(
        specification,
        [summary for _, _, _, summary in running],
        time.monotonic() - started,
    )
    return 0


def _write_sharded_summary(
    specification: LaunchSpec, summary_paths: list[Path], wall_seconds: float
) -> None:
    reports = [json.loads(path.read_text()) for path in summary_paths]
    commons = [report["replay"]["common"] for report in reports]
    aggregate = {
        "attempted_steps": sum(common["attempted_steps"] for common in commons),
        "success_steps": sum(common["success_steps"] for common in commons),
        "failed_steps": sum(common["failed_steps"] for common in commons),
        "request_throughput_per_s": sum(
            common["request_throughput_per_s"] for common in commons
        ),
        "output_token_throughput_per_s": sum(
            common["output_token_throughput_per_s"] for common in commons
        ),
    }
    payload = {
        "kind": "sharded_run",
        "processes": specification.processes,
        "wall_duration_s": wall_seconds,
        "aggregate": aggregate,
        "shards": [str(path) for path in summary_paths],
    }
    assert specification.result_file is not None
    specification.result_file.write_text(json.dumps(payload, indent=2) + "\n")


def _visualize(specification: LaunchSpec) -> int:
    visualization_log = specification.output_directory / "visualize.log"
    command = [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT / "viz"),
        "viz",
        str(specification.output_directory),
    ]
    with visualization_log.open("w") as writer:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=writer,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(f"  failed; see {visualization_log}", file=sys.stderr)
        _print_tail(visualization_log)
    else:
        print(f"  wrote  {specification.output_directory / 'figures'}")
    return completed.returncode


def _print_tail(path: Path, lines: int = 20) -> None:
    try:
        content = path.read_text().splitlines()
    except OSError:
        return
    for line in content[-lines:]:
        print(f"  {line}", file=sys.stderr)


def _print_plan(specification: LaunchSpec, build_type: str) -> None:
    command = [
        str(_binary_path(specification, build_type)),
        *specification.arguments,
    ]
    print("resolved command:")
    print(f"  {shlex.join(command)}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config_path = arguments.config.resolve()
    try:
        specification = load_task_config(arguments.task, config_path)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    ui.heading(specification, config_path, arguments.build_type)
    if arguments.dry_run:
        _print_plan(specification, arguments.build_type)
        return 0

    specification.output_directory.mkdir(parents=True, exist_ok=True)
    stage_count = 4 if specification.postprocess_viz else 3
    ui.stage(1, stage_count, "build", specification.binary)
    if not _build(specification, arguments.build_type):
        return 1

    command = [
        str(_binary_path(specification, arguments.build_type)),
        *specification.arguments,
    ]
    _snapshot_invocation(specification, config_path, command)
    ui.stage(2, stage_count, "execute", specification.task)
    if specification.processes > 1:
        returncode = _execute_sharded(
            specification,
            command,
            show_engine_output=arguments.show_engine_output,
        )
    else:
        returncode = _execute(
            specification,
            command,
            show_engine_output=arguments.show_engine_output,
        )
    if returncode != 0:
        ui.render_failure(specification, returncode)
        if not arguments.show_engine_output:
            _print_tail(specification.terminal_log)
        return returncode

    if specification.postprocess_viz:
        ui.stage(3, stage_count, "visualize", "sweep figures")
        visualization_returncode = _visualize(specification)
        if visualization_returncode != 0:
            return visualization_returncode

    ui.stage(stage_count, stage_count, "report")
    ui.render_result(specification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
