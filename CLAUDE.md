# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust replay engine (`src/`) that turns typed CSV traces into streaming
generation requests against OpenAI/vLLM/SGLang endpoints, plus a Python
launcher (`launcher/`) that is the *only* supported operator interface, plus an
optional Python plotting sidecar (`viz/`).

`ARCHITECTURE.md` is the end-to-end data-flow document (schema → loader →
workload → executor → backend → records/summary) and is kept current; read it
before changing anything that crosses layer boundaries. `README.md` is the
operator reference (YAML blocks, CSV formats, metrics, internal Rust flags).
`skills/coding-trace-replay/SKILL.md` is the agent-facing decision workflow for
actually running things.

## Commands

Rust engine (from repo root):

```bash
cargo test -q                              # all inline #[cfg(test)] modules
cargo test -q workload::tests::name_here   # one test, by path substring
cargo test -q --bin sweep                  # one binary's tests
cargo test -q --no-default-features        # schema-only build must keep compiling
cargo fmt && cargo clippy --all-targets
cargo build --release --bin session_runner # the launcher does this for you
```

Launcher (Python, `uv`):

```bash
uv run pytest -q                                   # tests/test_launcher.py
uv run pytest -q tests/test_launcher.py::test_name # one test
uv run ruff check .
```

Viz sidecar (its own uv project, `viz/`):

```bash
cd viz && uv run pytest -q
cd viz && uv run viz ../out/sweep     # or the sweep.json inside it
```

Running the engine — always task + one YAML, from repo root:

```bash
uv run python -m launcher run      configs/run.example.yaml [--dry-run]
uv run python -m launcher sweep    configs/sweep.example.yaml
uv run python -m launcher tracegen configs/tracegen.example.yaml
uv run python -m launcher selfcheck configs/selfcheck.example.yaml
```

`--dry-run` validates YAML and prints the resolved argv without building or
running. `--build-type debug` builds the debug binary. `--show-engine-output`
streams every engine line instead of the condensed panel; the full output is
always in `terminal.log` in the output directory.

Two ways to exercise the engine without a real server:

- `replay.dry_run: true` in a `run` YAML — static trace/workload inspection, no
  network.
- `tools/stub_server.py` (a vLLM-shaped SSE stub with fixed timing and an
  optional `--capacity` knee) — this is what `selfcheck` drives to test the
  client's own measurement fidelity.

## Architecture invariants

These are the constraints that a change is most likely to break silently.

**The launcher/engine boundary.** YAML is the operator contract; Rust flags in
`src/cli.rs` are an internal lowering target. `launcher/config.py` validates
strictly (unknown keys, duplicate keys, types), resolves paths relative to the
YAML file, and lowers to argv. It must never implement replay semantics — no
CSV reading, no prompt building, no metric computation. A new operator-visible
capability is not done until it exists in the YAML schema; a Rust flag alone is
not a feature.

**A file declares one complete format.** `--input-file-format` /`input.format`
names a *family + layout* pair (e.g. `text-generation-session-execution-v2`).
`RequestFamily` is derived from the format, never inferred from CSV headers and
never varying by row. Tags (`slo`, `priority`, `session`, `speculative`) are
orthogonal column bundles added on top. Each format module under
`src/schema/format/` owns its `COLUMNS`, row type, per-row validation, and
`load`; `format/load_utils.rs` holds only mechanical CSV work and must not grow
family knowledge or a generic request union.

**Validate the whole file, then truncate.** `--max-items` is applied after
full-file validation so a small limit cannot hide a malformed later row.

**`ReplayWorkload` is a runtime sum type, not a second schema.** It exists only
because the format is known at runtime. Don't add parsing or policy to it.

**Sessions are the unit.** A session holds one concurrency slot across all its
rounds *and* tool waits; rounds are closed-loop (round `i+1` waits on `i`).
Admission under `--max-concurrency` is strictly in trace declaration order —
deterministic admission is what makes two runs of the same trace comparable, so
don't replace it with whatever the async runtime polls first.

**Measurement paths must not backpressure the request path.** Step logs and the
Parquet timeline go over channels to separate writers; a full timeline channel
drops samples (counted as `dropped_requests`) rather than stalling submission.

**The `runtime` feature gate is load-bearing.** `src/lib.rs` exposes `schema` +
`release` unconditionally and gates everything else behind `runtime`, so a
simulator can read the same trace files with `default-features = false` and no
reqwest/tokio/tokenizers/hf-hub. New runtime dependencies must be `optional =
true` in `Cargo.toml` and listed in the `runtime` feature; new modules that use
them must be `#[cfg(feature = "runtime")]`. Verify with `cargo test
--no-default-features`.

**`[workspace]` in `Cargo.toml` is intentional.** This crate is vendored as a
submodule under larger workspaces; keeping it its own workspace stops it from
inheriting the host repo's members and target dir. Don't remove it.

**`viz/` is one-directional.** Nothing in `src/` or `launcher/` may depend on
it beyond the launcher's optional post-sweep subprocess call; a missing
plotting dependency must never cost someone a long measurement.

**Prefix-cache accounting is mandatory, not optional telemetry.** Every live
run preflights that the server exposes prefix caching and cached-prompt-token
usage and aborts if not. Keep `prefix_len` (planned reuse) and
`cached_prompt_tokens` (observed reuse) distinct everywhere — reporting one as
the other is the failure mode the preflight exists to prevent.

## Conventions

- Rust tests are inline `#[cfg(test)]` modules next to the code. The top-level
  `tests/` directory covers the launcher, benchmark materializers, mock dialect
  server, and end-to-end replay surfaces.
- Comments in this codebase explain *why* a boundary exists, not what the line
  does. Match that register; don't add narration.
- Commit subjects are `area: what changed`, lowercase, imperative, describing
  the design move (`backend: make the stream fold a testable unit`).
- Artifact filenames per output directory are stable and configurable:
  `requests.jsonl`, `summary.json`, `timeline.parquet`, `terminal.log`, plus
  `sweep.json` / `selfcheck.json`, and `launcher-config.yaml` + `command.txt`
  snapshotted beside every run.
- The JSONL record contract is versioned (`src/record.rs`, currently schema
  v15); changing fields means bumping it and updating the README's output
  contract section.

## Keeping the interface in sync

When you change launcher tasks, YAML keys, backend requirements, artifact
names, or result semantics, update all of these in the same change:

1. `launcher/config.py` validation + `tests/test_launcher.py`;
2. the matching `configs/*.example.yaml`;
3. `README.md` and `ARCHITECTURE.md` (and `ARCHITECTURE.zh-CN.md` /
   `.code-lessons/architecture/req-frontend-architecture.yaml` when the
   structure itself moved);
4. `skills/coding-trace-replay/SKILL.md` — the decision workflow only; detailed
   field lists stay in the example YAML and README.

## Operating a real server

Don't restart, reconfigure, or stop an existing serving process without
explicit authorization. If a test server is authorized, use an isolated
port/GPU, keep it in `tmux`, and clean up only that process. vLLM needs
`--enable-prompt-tokens-details` with prefix caching left on; `vllm-tokens`
additionally needs `--tokens-only`; `sglang-tokens` needs
`--skip-tokenizer-init` and incremental streaming output.
