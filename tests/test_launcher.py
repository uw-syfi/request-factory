from __future__ import annotations

from pathlib import Path

import pytest

from launcher import ui
from launcher.config import ConfigError, load_task_config


def write_config(directory: Path, text: str, name: str = "task.yaml") -> Path:
    config_path = directory / name
    config_path.write_text(text)
    return config_path


def minimal_run_yaml(extra: str = "") -> str:
    return f"""
input:
  trace: trace.csv
  format: text-generation-session-execution-v2
  tags: [slo, priority]
corpus:
  text_file: corpus.txt
  tokenizer: org/model
server:
  model: served-model
replay:
  dry_run: false
  arrival_mode: saturated
  max_concurrency: 4
  context:
    max_model_len: 4096
    on_limit: skip
measurement:
  timeline: false
  slo:
    ttft_ms: 500
    e2e_ms: 2000
output:
  directory: results
{extra}
"""


def test_run_config_lowers_nested_blocks_and_resolves_paths(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, minimal_run_yaml())

    specification = load_task_config("run", config_path)
    arguments = list(specification.arguments)

    assert specification.binary == "session_runner"
    assert specification.output_directory == tmp_path / "results"
    assert arguments[arguments.index("--trace") + 1] == str(tmp_path / "trace.csv")
    assert arguments[arguments.index("--text-file") + 1] == str(tmp_path / "corpus.txt")
    assert arguments[arguments.index("--tokenizer") + 1] == "org/model"
    assert arguments[arguments.index("--trace-tags") + 1] == "slo,priority"
    assert arguments[arguments.index("--arrival-mode") + 1] == "saturated"
    assert arguments[arguments.index("--timeline") + 1] == "false"
    assert arguments[arguments.index("--slo") + 1] == "ttft_ms=500,e2e_ms=2000"
    assert "--skip-when-reaching-limit" in arguments


def test_sweep_owns_rate_and_can_request_visualization(tmp_path: Path) -> None:
    config_text = (
        minimal_run_yaml().replace(
            "input:\n",
            "search:\n  mode: grid\n  rates: [1, 2, 4]\ninput:\n",
        )
        + "\nvisualization:\n  enabled: true\n"
    )
    config_path = write_config(tmp_path, config_text)

    specification = load_task_config("sweep", config_path)
    arguments = list(specification.arguments)

    assert specification.binary == "sweep"
    assert specification.postprocess_viz
    assert arguments[arguments.index("--mode") + 1] == "grid"
    assert arguments[arguments.index("--rates") + 1] == "1,2,4"
    assert "--rate" not in arguments


def test_sweep_rejects_a_run_level_rate(tmp_path: Path) -> None:
    config_text = minimal_run_yaml().replace(
        "replay:\n", "search:\n  mode: max-sustainable-rate\nreplay:\n  rate: 10\n"
    )
    config_path = write_config(tmp_path, config_text)

    with pytest.raises(ConfigError, match="controlled by search"):
        load_task_config("sweep", config_path)


def test_tracegen_supports_synthetic_and_coding_session(tmp_path: Path) -> None:
    synthetic_path = write_config(
        tmp_path,
        """
generator:
  type: synthetic
  sessions: 12
  rounds: uniform:1..4
  input_len: 64
  output_len: 8
  arrival_rate: 3
output:
  trace: generated/synthetic.csv
""",
        "synthetic.yaml",
    )
    synthetic = load_task_config("tracegen", synthetic_path)
    assert synthetic.arguments[:3] == (
        "synthetic",
        "--out",
        str(tmp_path / "generated" / "synthetic.csv"),
    )
    assert synthetic.arguments[synthetic.arguments.index("--sessions") + 1] == "12"

    coding_path = write_config(
        tmp_path,
        """
generator:
  type: coding-session
  source: raw.csv
  policy: monotonic
  max_sessions: 10
  session_order: shuffle
output:
  trace: generated/coding.csv
""",
        "coding.yaml",
    )
    coding = load_task_config("tracegen", coding_path)
    assert coding.arguments[0] == "coding-session"
    assert coding.arguments[coding.arguments.index("--source") + 1] == str(
        tmp_path / "raw.csv"
    )
    assert coding.arguments[coding.arguments.index("--max-sessions") + 1] == "10"


def test_tracegen_rejects_bad_distribution_and_probability_types(
    tmp_path: Path,
) -> None:
    config_path = write_config(
        tmp_path,
        """
generator:
  type: synthetic
  rounds: [1, 2]
  compaction_probability: 1.5
output:
  trace: generated.csv
""",
    )

    with pytest.raises(ConfigError, match="generator.rounds"):
        load_task_config("tracegen", config_path)


def test_selfcheck_has_its_own_small_schema(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
tokenizer:
  path: ./tokenizer.json
checks:
  pairs: 3
  port: 9001
output:
  directory: checks
""",
    )

    specification = load_task_config("selfcheck", config_path)

    assert specification.binary == "selfcheck"
    assert specification.result_file == tmp_path / "checks" / "selfcheck.json"
    assert specification.arguments[specification.arguments.index("--pairs") + 1] == "3"
    assert (
        specification.arguments[specification.arguments.index("--port") + 1] == "9001"
    )


def test_unknown_and_duplicate_keys_are_never_ignored(tmp_path: Path) -> None:
    unknown_path = write_config(
        tmp_path, minimal_run_yaml(extra="surprise: true"), "unknown.yaml"
    )
    with pytest.raises(ConfigError, match="config.surprise"):
        load_task_config("run", unknown_path)

    duplicate_path = write_config(
        tmp_path,
        "input: {}\ninput: {}\n",
        "duplicate.yaml",
    )
    with pytest.raises(ConfigError, match="duplicate key 'input'"):
        load_task_config("run", duplicate_path)


def test_terminal_filter_keeps_progress_but_hides_per_request_noise() -> None:
    assert ui.should_echo_engine_line("run", "sessions 2/2 | steps 4/4 completed=4")
    assert ui.should_echo_engine_line(
        "run", "prefix hit rate summary | measured_steps=4"
    )
    assert not ui.should_echo_engine_line(
        "run", "prefix hit rate | request_id=session_0_round_0"
    )
