# viz — draw what a sweep measured

Optional. Nothing in the Rust side imports this, calls it, or checks that it
exists: a sweep that takes an hour must not be able to fail because a plotting
dependency is missing.

```bash
cd viz
uv run viz ../out/sweep            # a sweep's --out directory, or its sweep.json
uv run viz ../out/sweep --out /tmp/figures --arrival-rates 10 200
uv run pytest
```

## What it reads

Only files a sweep already wrote:

| File | Used for |
|---|---|
| `sweep.json` | the curve, the knee or peak, the config, and every caveat |
| `points/rate_*/requests.jsonl` | per-request latency distributions |
| `points/rate_*/timeline.parquet` | per-request token arrivals |

Point directories are looked for where `sweep.json` recorded them and, failing
that, beside the report — so a results tree copied off the machine that produced
it still plots. A point in neither place is skipped, not reported empty.

## What it will not do

The plots exist to be believed, so most of the design is about refusals:

- **A null point is counted, never interpolated.** A run held to no objective
  reports no attainment, and a line drawn through that gap would claim a
  measurement nobody took. The count reaches the caption.
- **A knee that was `never_crossed` is not drawn at the edge of the range.** It
  is a result — the knee is outside what was searched — and it goes in the
  caveats instead.
- **A failed request contributes no latency.** It has an error, not a slow
  response; including it would put the left tail where requests that never ran
  are. Excluded requests are counted under the axis.
- **One arrival stays one point.** A chunk carrying four token ids is a single
  observable instant, so the arrival figure is a step function with a marker per
  event. Expanding it into four points would draw a pace the server never
  delivered — the one mistake the Parquet schema is shaped to prevent.
- **Every figure carries the sweep's caveats.** Cache contamination, dropped
  timelines, reused points: all printed under the axes, so a contaminated run
  cannot be screenshotted without them.
- **Nothing is drawn for a metric no point reported.** The figure says so.
  Empty axes read as "the server produced nothing", which is a different fact.

## Units

The x axis counts **workload units** offered — sessions for a session trace —
and the y axis counts **steps** delivered. On a trace averaging two rounds per
session those differ by a factor of two, so the throughput figure's reference
line has slope `steps_per_workload_unit`, not one. A plain `y = x` diagonal
would show a session server outrunning its own offered load.

If a curve's points disagree on that ratio, no reference line is drawn at all.

## Layout

```text
viz/
├── pyproject.toml
├── src/req_frontend_viz/
│   ├── cli.py       one command, no required choices
│   ├── load.py      readers and transforms — no matplotlib, so it is testable
│   └── plots.py     the figures, checked by eye
└── tests/test_load.py
```

The arithmetic lives in `load.py` and is tested; `plots.py` draws and is not.
That split is why `percentile` is transcribed from `summary.rs` rather than
taken from numpy — a viz whose p90 disagrees with `summary.json`'s p90 produces
a figure that contradicts its own caption.
