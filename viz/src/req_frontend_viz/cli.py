"""`viz <sweep-directory>` — draw everything that sweep can support.

One command with no required choices, because the alternative is a reader who
draws the throughput curve, sees a clean knee, and never learns that the
attainment curve was flat at zero the whole time. Whatever the run measured gets
a figure; whatever it did not gets a figure saying so.

Nothing here is ever invoked by a sweep. A missing `viz/` environment must not
be able to cost anyone a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .load import iter_point_directories, load_sweep, load_timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viz",
        description="Draw a req-frontend sweep: throughput, attainment, latency, token arrivals.",
    )
    parser.add_argument(
        "sweep",
        type=Path,
        help="a sweep's --out directory, or the sweep.json inside it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="where the figures go (default: a figures/ directory beside sweep.json)",
    )
    parser.add_argument(
        "--arrival-rates",
        type=float,
        nargs="*",
        help=(
            "rates whose token-arrival timeline to draw. Default: the lowest and "
            "highest measured, which is where the streams differ most"
        ),
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="skip the per-request latency figure, which reads every point's JSONL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    # Imported here rather than at module scope so `--help` works, and fails
    # honestly, without matplotlib installed.
    from . import plots

    sweep = load_sweep(arguments.sweep)
    assert sweep.path is not None
    out = arguments.out or sweep.path.parent / "figures"

    written = [
        plots.throughput_curve(sweep, out / "throughput.png"),
        plots.attainment_curve(sweep, out / "attainment.png"),
    ]

    points = list(iter_point_directories(sweep))
    if not points:
        print(
            f"viz | no point directory from {sweep.path} is readable; "
            "drew the curves only",
            file=sys.stderr,
        )
    else:
        if not arguments.no_latency:
            written.append(plots.latency_distributions(points, sweep, out / "latency.png"))
        for rate, directory in _arrival_points(points, arguments.arrival_rates):
            timeline = directory / "timeline.parquet"
            if not timeline.is_file():
                print(f"viz | {timeline} is missing; skipped its arrivals", file=sys.stderr)
                continue
            written.append(
                plots.token_arrivals(
                    load_timeline(timeline),
                    rate,
                    out / f"arrivals_rate_{rate:012.6f}.png",
                    sweep.caveats(),
                )
            )

    for note in sweep.caveats():
        print(f"viz | caveat: {note}", file=sys.stderr)
    for path in written:
        print(path)
    return 0


def _arrival_points(
    points: list[tuple[float, Path]], requested: list[float] | None
) -> list[tuple[float, Path]]:
    """Which points get a token-arrival figure.

    Default is the extremes rather than every point: the figure is per-request,
    so one per rate on a twenty-point sweep is twenty figures nobody opens, and
    the comparison worth seeing is the lightest load against the heaviest.

    A requested rate is matched to the nearest measured one, because bisection
    produces rates nobody would type. The match is reported by the returned rate
    itself, which is what the figure is titled with.
    """
    if not points:
        return []
    if not requested:
        return [points[0]] if len(points) == 1 else [points[0], points[-1]]
    chosen: list[tuple[float, Path]] = []
    for wanted in requested:
        nearest = min(points, key=lambda point: abs(point[0] - wanted))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


if __name__ == "__main__":
    raise SystemExit(main())
