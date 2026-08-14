"""Task-oriented YAML launcher for every req-frontend execution mode."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import ui
from .config import REPO_ROOT, ConfigError, LaunchSpec, load_task_config

TASKS = ("run", "sweep", "tracegen", "selfcheck")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m launcher",
        description="Run req-frontend tasks from strict, structured YAML.",
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
