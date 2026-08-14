"""The figures, and what each of them is allowed to claim.

Four plots, one per question the sweep answers. Every one of them:

- draws only measured points, and marks them, so a reader can see the sampling
  the adaptive search actually chose rather than a smooth line implying a grid;
- prints the sweep's caveats under the axes, so a contaminated or sampled run
  cannot be screenshotted without them;
- says nothing when there is nothing to say. A metric that was never measured
  produces a figure stating that, not an empty pair of axes.

Checked by eye, which is why the arithmetic lives in `load.py` and is tested
there instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Written to files, never shown; a sweep runs on a server.

import matplotlib.pyplot as pyplot  # noqa: E402

from .load import (  # noqa: E402
    LATENCY_FIELDS,
    Series,
    Sweep,
    Timeline,
    latency_samples,
    load_records,
    percentile,
)

# How many requests the token-arrival figure draws. Enough to see whether the
# streams differ from each other, few enough that each remains legible.
ARRIVAL_REQUESTS = 12


def _annotate(figure: pyplot.Figure, sweep: Sweep) -> None:
    notes = sweep.caveats()
    if not notes:
        return
    figure.text(
        0.01,
        0.01,
        "\n".join(f"• {note}" for note in notes),
        fontsize=7,
        color="#b03030",
        va="bottom",
    )
    figure.subplots_adjust(bottom=0.12 + 0.03 * len(notes))


def _nothing_to_draw(axes: pyplot.Axes, message: str) -> None:
    """State the absence rather than leaving blank axes.

    Empty axes read as "the server produced nothing". Most of the time the truth
    is "this run was not asked to measure that", which is a different fact.
    """
    axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9)
    axes.set_xticks([])
    axes.set_yticks([])


def throughput_curve(sweep: Sweep, out: Path) -> Path:
    """Delivered throughput against offered rate, with the boundary marked.

    The dashed reference is what perfect delivery would look like — every step
    the offered rate implies, completed within the run. The measured curve
    leaving it *is* the saturation the sweep was looking for, so it is drawn
    rather than described: the shortfall the boundary tests is the vertical
    distance between the two.

    Note its slope. The x axis counts *workload units* offered and the y axis
    counts *steps* delivered, so on a session trace averaging two rounds each
    the reference rises twice as fast as `y = x`. Drawing the plain diagonal
    would show a session server outrunning its own offered load, which is the
    unit mistake this line exists to make impossible to draw.
    """
    figure, axes = pyplot.subplots(figsize=(8, 5))
    requests = sweep.series("request_throughput_per_s")
    tokens = sweep.series("output_token_throughput_per_s")

    if not len(requests) and not len(tokens):
        _nothing_to_draw(axes, "no point in this sweep reported a throughput")
    else:
        if len(requests):
            axes.plot(requests.x, requests.y, "o-", color="#1f77b4", label="delivered steps/s")
            steps_per_unit = sweep.steps_per_workload_unit()
            if steps_per_unit:
                # The reference is the offered rate converted into the steps the
                # curve is counted in — `y = x` only when a unit is one step.
                edge = max(requests.x)
                axes.plot(
                    [0, edge],
                    [0, edge * steps_per_unit],
                    "--",
                    color="#999999",
                    linewidth=1,
                    label=f"offered ({steps_per_unit:.3g} steps/unit)",
                )
        if len(tokens):
            secondary = axes.twinx()
            secondary.plot(tokens.x, tokens.y, "s-", color="#d62728", label="output tok/s")
            secondary.set_ylabel("output tokens per second", color="#d62728")
            secondary.legend(loc="lower right", fontsize=8)

        bracket = sweep.knee_bracket()
        if bracket:
            axes.axvspan(*bracket, color="#2ca02c", alpha=0.15)
            axes.axvline(bracket[1], color="#2ca02c", linewidth=1)
            axes.annotate(
                f"knee in [{bracket[0]:.3g}, {bracket[1]:.3g}] {sweep.knob}",
                xy=(bracket[1], axes.get_ylim()[1]),
                xytext=(4, -12),
                textcoords="offset points",
                color="#2ca02c",
                fontsize=8,
            )
        plateau = sweep.plateau_span()
        if plateau:
            axes.axvspan(*plateau, color="#9467bd", alpha=0.12)
            axes.annotate(
                f"plateau {plateau[0]:.3g}–{plateau[1]:.3g}",
                xy=(plateau[0], axes.get_ylim()[0]),
                xytext=(4, 10),
                textcoords="offset points",
                color="#9467bd",
                fontsize=8,
            )
        axes.legend(loc="upper left", fontsize=8)

    axes.set_xlabel(f"offered {sweep.knob} (workload units per second)")
    axes.set_ylabel("delivered steps per second", color="#1f77b4")
    axes.set_title(f"Throughput against offered load — {sweep.objective}")
    axes.grid(alpha=0.25)
    _annotate(figure, sweep)
    return _save(figure, out)


def attainment_curve(sweep: Sweep, out: Path) -> Path:
    """SLO attainment against offered rate.

    Two independent series, never merged. `slo_attainment` is the fraction of
    steps meeting the *run's* bounds; `declared_slo_attainment` is the fraction
    of rows with their own metric-specific bounds that met every bound they
    declared — a different denominator over a different subset. Averaging them would produce
    a number that answers no question.
    """
    figure, axes = pyplot.subplots(figsize=(8, 5))
    overall = sweep.series("slo_attainment")
    declared = sweep.series("declared_slo_attainment")

    if not len(overall) and not len(declared):
        _nothing_to_draw(
            axes,
            "no point was held to an objective\n"
            "(run the sweep with --slo, or give the trace an .slo.json sidecar)",
        )
    else:
        if len(overall):
            axes.plot(overall.x, overall.y, "o-", color="#1f77b4", label="run objective")
        if len(declared):
            axes.plot(
                declared.x,
                declared.y,
                "s--",
                color="#ff7f0e",
                label="rows' own metric SLOs",
            )
        target = sweep.config.get("target_attainment")
        if target is not None:
            axes.axhline(float(target), color="#2ca02c", linewidth=1, linestyle=":")
            axes.annotate(
                f"target {float(target):.3g}",
                xy=(axes.get_xlim()[0], float(target)),
                xytext=(4, 4),
                textcoords="offset points",
                color="#2ca02c",
                fontsize=8,
            )
        bracket = sweep.knee_bracket()
        if bracket:
            axes.axvspan(*bracket, color="#2ca02c", alpha=0.15)
        axes.set_ylim(0.0, 1.02)
        axes.legend(loc="lower left", fontsize=8)

    axes.set_xlabel(f"offered {sweep.knob} (per second)")
    axes.set_ylabel("fraction of steps attained")
    axes.set_title("SLO attainment against offered load")
    axes.grid(alpha=0.25)
    _annotate(figure, sweep)
    return _save(figure, out)


def latency_distributions(points: list[tuple[float, Path]], sweep: Sweep, out: Path) -> Path:
    """TTFT, TPOT and end-to-end, per rate, as box plots over every request.

    Boxes rather than the percentiles already in `summary.json`: the summary
    reports p50 and p90, and the interesting failure — a tail that grows while
    the median holds — is only visible with the whole distribution.

    Failed requests are excluded and counted. A failure has no latency, and
    including it would put the left tail where requests that never ran are.
    """
    figure, axes_row = pyplot.subplots(
        1, len(LATENCY_FIELDS), figsize=(4.5 * len(LATENCY_FIELDS), 5), sharex=True
    )
    loaded = [(rate, load_records(directory / "requests.jsonl")) for rate, directory in points]

    for axes, metric in zip(axes_row, LATENCY_FIELDS):
        series = [(rate, latency_samples(records, metric)) for rate, records in loaded]
        drawn = [(rate, sample) for rate, sample in series if len(sample)]
        excluded = sum(sample.missing for _, sample in series)

        if not drawn:
            _nothing_to_draw(axes, f"no request reported {metric}")
        else:
            axes.boxplot(
                [sample.y for _, sample in drawn],
                tick_labels=[f"{rate:.3g}" for rate, _ in drawn],
                showfliers=False,
                medianprops={"color": "#d62728"},
            )
            p99 = [percentile(sample.y, 0.99) for _, sample in drawn]
            axes.plot(range(1, len(drawn) + 1), p99, "^", color="#9467bd", label="p99")
            axes.legend(fontsize=8)
            if excluded:
                axes.set_xlabel(
                    f"offered {sweep.knob} (per second)\n{excluded} request(s) excluded: "
                    "failed or unmeasured",
                    fontsize=8,
                )
            else:
                axes.set_xlabel(f"offered {sweep.knob} (per second)")
        axes.set_title(metric)
        axes.set_ylabel("milliseconds")
        axes.grid(alpha=0.25, axis="y")

    figure.suptitle("Client-observed latency by offered rate (whiskers, no fliers)")
    _annotate(figure, sweep)
    return _save(figure, out)


def token_arrivals(timeline: Timeline, rate: float, out: Path, caveats: list[str]) -> Path:
    """When each request's tokens actually showed up.

    Drawn as a **step** function, and with a marker at every arrival. Between
    two markers the client observed nothing at all, so a straight line joining
    them would draw a steady trickle the server never delivered — and a chunk
    carrying four ids is one instant, not four. The flat treads are the waits;
    the risers are the arrivals, and their height is the chunk size.
    """
    figure, axes = pyplot.subplots(figsize=(9, 5.5))
    chosen = timeline.busiest(ARRIVAL_REQUESTS)

    if not chosen:
        _nothing_to_draw(axes, "this point's timeline is empty")
    else:
        viridis = matplotlib.colormaps["viridis"]
        colors = viridis([index / max(1, len(chosen) - 1) for index in range(len(chosen))])
        for color, request_id in zip(colors, chosen):
            series = timeline.arrivals(request_id)
            axes.step(series.x, series.y, where="post", color=color, linewidth=1)
            axes.plot(series.x, series.y, ".", color=color, markersize=3)
        kinds = timeline.event_kinds()
        legend = ", ".join(f"{name}={count}" for name, count in sorted(kinds.items()))
        axes.set_xlabel(f"milliseconds since the request was sent\nevents: {legend}", fontsize=9)

    axes.set_ylabel("cumulative output tokens delivered")
    axes.set_title(
        f"Token arrivals at {rate:.6g}/s — {len(chosen)} busiest of "
        f"{len(timeline.request_ids())} requests"
    )
    axes.grid(alpha=0.25)
    if caveats:
        figure.text(
            0.01,
            0.01,
            "\n".join(f"• {note}" for note in caveats),
            fontsize=7,
            color="#b03030",
            va="bottom",
        )
        figure.subplots_adjust(bottom=0.16 + 0.03 * len(caveats))
    return _save(figure, out)


def _save(figure: pyplot.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    pyplot.close(figure)
    return out


__all__ = [
    "attainment_curve",
    "latency_distributions",
    "throughput_curve",
    "token_arrivals",
    "Series",
]
