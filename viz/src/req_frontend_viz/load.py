"""Read what a run wrote, and refuse to invent the parts it did not.

Every function here returns what the files actually contain plus a count of what
was missing. The plots that follow may then say so. That split is the whole
point of this module: a plot with a gap silently interpolated is worse than no
plot, because it looks like a measurement.

Nothing here imports matplotlib, so the transforms are testable without a
display and without drawing anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# Latency metrics a request record carries, and which field each reads.
#
# TTFT falls back to `first_token_ms` exactly as the Rust side's
# `slo_measurement` does: a text-only backend reports no token-id timing at all,
# and reading only the first field would show every such run as unmeasured while
# its own summary quoted percentiles.
LATENCY_FIELDS: dict[str, tuple[str, ...]] = {
    "ttft_ms": ("first_token_id_ms", "first_token_ms"),
    "tpot_ms": ("token_delivery_tpot_ms",),
    "e2e_ms": ("total_duration_ms",),
}


@dataclass(frozen=True)
class Series:
    """Points that exist, and how many did not.

    `missing` is carried rather than dropped on the floor so a caption can say
    "18 of 20 points" instead of a reader assuming the axis is complete.
    """

    x: list[float]
    y: list[float]
    missing: int = 0

    def __len__(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class Sweep:
    """One `sweep.json`, as it was written."""

    knob: str
    objective: str
    config: dict[str, Any]
    curve: list[dict[str, Any]]
    points: list[dict[str, Any]]
    knee: dict[str, Any] | None = None
    peak: dict[str, Any] | None = None
    contamination_warning: str | None = None
    path: Path | None = None

    @property
    def dropped_timeline_requests(self) -> int:
        """Requests whose per-event timeline the writer could not keep up with.

        Summed across points: any nonzero total means at least one point's
        timeline is a sample of the run rather than a record of it.
        """
        return sum(
            int(point.get("metrics", {}).get("timeline_dropped_requests", 0) or 0)
            for point in self.points
        )

    @property
    def reused_points(self) -> int:
        return sum(1 for point in self.points if point.get("reused"))

    def series(self, metric: str) -> Series:
        """The curve's `metric` against rate, sorted by rate.

        Points where the metric is null are counted, not interpolated. A run
        held to no objective reports no attainment, and a line drawn through
        that gap would claim it did.
        """
        rates: list[float] = []
        values: list[float] = []
        missing = 0
        for entry in sorted(self.curve, key=lambda item: item["rate"]):
            value = entry.get(metric)
            if value is None:
                missing += 1
                continue
            rates.append(float(entry["rate"]))
            values.append(float(value))
        return Series(rates, values, missing)

    def steps_per_workload_unit(self) -> float | None:
        """The conversion between the offered rate and the delivered throughput.

        `rate` counts workload units — sessions for a session trace — while
        every throughput on the curve counts steps, and a session issues several
        rounds. A reference line drawn at `y = x` therefore claims a shortfall
        the server never had.

        `None` when the points disagree, which would mean the curve mixes
        traces; no line at all beats one drawn for a conversion that does not
        hold across the figure.
        """
        ratios = {
            round(float(entry["steps_per_workload_unit"]), 6)
            for entry in self.curve
            if entry.get("steps_per_workload_unit")
        }
        return ratios.pop() if len(ratios) == 1 else None

    def knee_bracket(self) -> tuple[float, float] | None:
        """The rates the knee was pinned between, or `None` if it was not found.

        A `never_crossed` or `always_crossed` outcome is a *result* — the knee is
        outside the range searched — so this returns nothing rather than the edge
        of the range, which would read as a located knee.
        """
        if not self.knee or self.knee.get("outcome") != "located":
            return None
        low = self.knee.get("last_good_rate")
        high = self.knee.get("first_bad_rate")
        if low is None or high is None:
            return None
        return float(low), float(high)

    def plateau_span(self) -> tuple[float, float] | None:
        if not self.peak:
            return None
        low = self.peak.get("plateau_low_rate")
        high = self.peak.get("plateau_high_rate")
        if low is None or high is None:
            return None
        return float(low), float(high)

    def caveats(self) -> list[str]:
        """Everything a reader must know before believing the figure.

        Collected in one place so no plot can forget one: each drawing function
        prints the same list under the axes.
        """
        notes: list[str] = []
        if self.contamination_warning:
            notes.append(f"cache contamination: {self.contamination_warning}")
        dropped = self.dropped_timeline_requests
        if dropped:
            notes.append(f"{dropped} request timelines dropped (sampled, not complete)")
        reused = self.reused_points
        if reused:
            notes.append(f"{reused} point(s) reused from an earlier sweep")
        if self.knee and self.knee.get("outcome") != "located":
            notes.append(f"knee {self.knee['outcome']}: outside the range searched")
        if self.peak and self.peak.get("outcome") != "located":
            notes.append(f"peak {self.peak['outcome']}: outside the range searched")
        return notes


def load_sweep(path: str | Path) -> Sweep:
    path = Path(path)
    if path.is_dir():
        path = path / "sweep.json"
    report = json.loads(path.read_text())
    return Sweep(
        knob=report.get("knob", "rate"),
        objective=report.get("objective", ""),
        config=report.get("config", {}),
        curve=report.get("curve", []),
        points=report.get("points", []),
        knee=report.get("knee"),
        peak=report.get("peak"),
        contamination_warning=report.get("contamination_warning"),
        path=path,
    )


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """One run's JSONL request log.

    A malformed line raises rather than being skipped: a log truncated by a
    killed process is a fact about the run, and quietly plotting the readable
    prefix would hide it.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
    return records


def latency_samples(records: Iterable[dict[str, Any]], metric: str) -> Series:
    """Per-request values of one latency metric.

    Only successful requests contribute. A failure has no latency — it has an
    error — and mixing the two produces a distribution whose left tail is
    requests that never ran. Failures are counted in `missing`.

    The x axis is the request's index among the samples, so the result plots as
    a distribution rather than against time; `Series` is reused only to carry
    the count of what was left out.
    """
    if metric not in LATENCY_FIELDS:
        raise KeyError(f"unknown latency metric {metric!r}; expected one of {sorted(LATENCY_FIELDS)}")
    fields = LATENCY_FIELDS[metric]
    values: list[float] = []
    missing = 0
    for record in records:
        outcome = record.get("outcome", {})
        if outcome.get("status") != "SUCCESS":
            missing += 1
            continue
        value = next(
            (outcome[name] for name in fields if outcome.get(name) is not None),
            None,
        )
        if value is None:
            missing += 1
            continue
        values.append(float(value))
    return Series(list(range(len(values))), values, missing)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolated percentile, matching `summary.rs::percentile_sorted`.

    Transcribed from the Rust rather than pulled from numpy, whose default
    `linear` method happens to agree today but is one of nine it offers. A viz
    that computes a p90 one way while `summary.json` computes it another
    produces a figure that disagrees with its own caption.

    The one deliberate divergence: no samples gives `None`, where the Rust
    returns `0.0` behind an emptiness check its callers make first. A plot has
    no such caller, and a zero drawn for "nothing was measured" is exactly the
    fabrication this module exists to avoid.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass(frozen=True)
class Timeline:
    """One point's per-event Parquet file, grouped by request.

    Each entry is one request's arrivals in `seq` order. **A row is an arrival,
    not a token**: a chunk carrying four ids is one row with `tokens = 4`. This
    class never expands them, which is what stops a reader from averaging rows
    as if each were a token measurement.
    """

    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def request_ids(self) -> list[str]:
        return list(self.events)

    def busiest(self, count: int) -> list[str]:
        """The requests with the most arrivals — the ones worth drawing.

        Ties broken by request id so the same file always yields the same
        selection; a figure that changes between runs of the same command is not
        evidence of anything.
        """
        ranked = sorted(
            self.events.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        return [request_id for request_id, _ in ranked[:count]]

    def arrivals(self, request_id: str) -> Series:
        """Cumulative tokens against elapsed milliseconds, as a step function.

        Plotted stepwise, never interpolated: between two arrivals the client
        observed nothing, and a straight line there would draw tokens trickling
        in at a steady pace the server never delivered.
        """
        events = self.events.get(request_id, [])
        return Series(
            [float(event["elapsed_ms"]) for event in events],
            [float(event["cumulative_tokens"]) for event in events],
        )

    def event_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for events in self.events.values():
            for event in events:
                counts[event["kind"]] = counts.get(event["kind"], 0) + 1
        return counts


def load_timeline(path: str | Path) -> Timeline:
    """Read a `timeline.parquet` and group it by request.

    Rows are re-sorted by `seq` rather than trusted in file order: a Parquet
    reader may return row groups in any order it likes, which is exactly why the
    writer records `seq` in the first place.
    """
    import pyarrow.parquet as parquet

    table = parquet.read_table(Path(path))
    return timeline_from_rows(table.to_pylist())


def timeline_from_rows(rows: Iterable[dict[str, Any]]) -> Timeline:
    """The grouping half of `load_timeline`, separated so it is testable.

    Reading Parquet needs a file; grouping needs only rows, and the ordering
    guarantee is the part worth a test.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["request_id"], []).append(row)
    for events in grouped.values():
        events.sort(key=lambda event: event["seq"])
    return Timeline(grouped)


def iter_point_directories(sweep: Sweep) -> Iterator[tuple[float, Path]]:
    """Each point's rate and directory, in rate order.

    `sweep.json` records paths as they were written — relative to wherever the
    sweep was launched. When that path no longer resolves, the same directory
    name is looked for beside the report, so a results tree copied off a machine
    still plots. Both are checked and neither is invented: a point whose
    directory is nowhere is skipped rather than reported empty.
    """
    beside = sweep.path.parent if sweep.path else Path()
    seen: set[str] = set()
    for point in sorted(sweep.points, key=lambda item: item["rate"]):
        recorded = point.get("directory")
        if not recorded or recorded in seen:
            continue
        seen.add(recorded)
        directory = Path(recorded)
        if not directory.is_dir():
            directory = beside / "points" / directory.name
        if directory.is_dir():
            yield float(point["rate"]), directory
