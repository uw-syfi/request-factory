"""What the transforms must not do.

Almost every test here is about a *refusal*: not interpolating a missing point,
not counting a failure as a fast request, not expanding one arrival into four
tokens. The drawing is checked by eye; these are the parts where being wrong
produces a figure that looks right.
"""

from __future__ import annotations

import json

import pytest
from req_frontend_viz.load import (
    Sweep,
    iter_point_directories,
    latency_samples,
    load_records,
    load_sweep,
    percentile,
    timeline_from_rows,
)


def sweep(**overrides) -> Sweep:
    base = {
        "knob": "rate",
        "objective": "max sustainable rate",
        "config": {},
        "curve": [
            {"rate": 2.0, "request_throughput_per_s": 2.0, "slo_attainment": 1.0},
            {"rate": 8.0, "request_throughput_per_s": 6.0, "slo_attainment": None},
            {"rate": 4.0, "request_throughput_per_s": 4.0, "slo_attainment": 0.9},
        ],
        "points": [],
    }
    base.update(overrides)
    return Sweep(**base)


def test_the_curve_is_read_in_rate_order_not_measurement_order():
    # The sweep records points in the order the search visited them, which
    # doubles and bisects. Plotting that order draws a scribble.
    series = sweep().series("request_throughput_per_s")

    assert series.x == [2.0, 4.0, 8.0]
    assert series.y == [2.0, 4.0, 6.0]


def test_a_missing_metric_is_counted_rather_than_interpolated():
    # The one that matters: a point held to no objective reports null
    # attainment, and joining its neighbours would draw a measurement that was
    # never taken.
    series = sweep().series("slo_attainment")

    assert series.x == [2.0, 4.0]
    assert series.y == [1.0, 0.9]
    assert series.missing == 1


def test_a_metric_no_point_reported_yields_an_empty_series_not_zeros():
    series = sweep().series("declared_slo_attainment")

    assert len(series) == 0
    assert series.missing == 3


@pytest.mark.parametrize("outcome", ["never_crossed", "always_crossed"])
def test_a_knee_outside_the_range_is_not_reported_at_the_ranges_edge(outcome):
    # `never_crossed` means the knee is above everything measured. Drawing a
    # line at the highest rate would claim the search found it there.
    report = sweep(knee={"outcome": outcome, "last_good_rate": 8.0, "first_bad_rate": None})

    assert report.knee_bracket() is None
    assert any(outcome in note for note in report.caveats())


def test_a_located_knee_returns_the_bracket_it_was_pinned_to():
    report = sweep(
        knee={
            "outcome": "located",
            "last_good_rate": 47.5,
            "first_bad_rate": 50.0,
            "bracket_width": 0.05,
        }
    )

    assert report.knee_bracket() == (47.5, 50.0)
    assert report.caveats() == []


def test_contamination_and_dropped_timelines_always_reach_the_caption():
    # Both mean the numbers are not what they appear to be, so neither may be
    # something a reader has to go looking for.
    report = sweep(
        contamination_warning="sglang-tokens has no reset endpoint",
        points=[
            {"rate": 1.0, "metrics": {"timeline_dropped_requests": 3}, "reused": True},
            {"rate": 2.0, "metrics": {"timeline_dropped_requests": 0}, "reused": False},
        ],
    )

    notes = " ".join(report.caveats())
    assert "contamination" in notes
    assert "3 request timelines dropped" in notes
    assert "1 point(s) reused" in notes


def test_only_successful_requests_contribute_a_latency():
    # A failure has no latency. Including it puts the left tail where requests
    # that never ran are.
    records = [
        {"outcome": {"status": "SUCCESS", "first_token_id_ms": 100.0}},
        {"outcome": {"status": "FAILED", "first_token_id_ms": 1.0}},
        {"outcome": {"status": "SKIPPED_CONTEXT_OVERFLOW", "first_token_id_ms": 2.0}},
    ]

    series = latency_samples(records, "ttft_ms")

    assert series.y == [100.0]
    assert series.missing == 2


def test_ttft_falls_back_the_same_way_the_rust_side_does():
    # A text-only backend reports no token-id timing at all. Reading only the
    # first field would show every such run as unmeasured while its own summary
    # quoted percentiles.
    records = [
        {"outcome": {"status": "SUCCESS", "first_token_ms": 42.0}},
        {"outcome": {"status": "SUCCESS", "first_token_id_ms": 10.0, "first_token_ms": 99.0}},
    ]

    assert latency_samples(records, "ttft_ms").y == [42.0, 10.0]


def test_a_successful_request_that_measured_nothing_is_missing_not_zero():
    records = [{"outcome": {"status": "SUCCESS", "token_delivery_tpot_ms": None}}]

    series = latency_samples(records, "tpot_ms")

    assert series.y == []
    assert series.missing == 1


def test_an_unknown_metric_is_an_error_rather_than_an_empty_plot():
    with pytest.raises(KeyError):
        latency_samples([], "ttbt_ms")


def test_percentiles_interpolate_exactly_as_the_rust_summary_does():
    # A viz whose p90 disagrees with summary.json's p90 produces a figure that
    # contradicts its own caption. `summary.rs::percentile_sorted` interpolates
    # between the two straddling samples; these are its answers.
    values = [float(value) for value in range(1, 11)]

    assert percentile(values, 0.5) == pytest.approx(5.5)
    assert percentile(values, 0.9) == pytest.approx(9.1)
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 10.0
    assert percentile([7.0], 0.99) == 7.0
    # The deliberate divergence: no samples is not a zero.
    assert percentile([], 0.5) is None


def test_one_arrival_carrying_four_tokens_stays_one_point():
    # The mistake the timeline schema exists to prevent. Four ids delivered
    # together share one observable instant; expanding them into four points
    # would draw a pace the server never delivered.
    rows = [
        {"request_id": "r", "seq": 0, "elapsed_ms": 10.0, "kind": "tokens", "tokens": 1,
         "cumulative_tokens": 1},
        {"request_id": "r", "seq": 1, "elapsed_ms": 30.0, "kind": "tokens", "tokens": 4,
         "cumulative_tokens": 5},
    ]

    series = timeline_from_rows(rows).arrivals("r")

    assert series.x == [10.0, 30.0]
    assert series.y == [1.0, 5.0]


def test_events_are_ordered_by_seq_not_by_the_order_parquet_returned_them():
    # A Parquet reader may hand back row groups in any order it likes, which is
    # why the writer records `seq` at all.
    rows = [
        {"request_id": "r", "seq": 2, "elapsed_ms": 30.0, "kind": "finish", "tokens": 0,
         "cumulative_tokens": 5},
        {"request_id": "r", "seq": 0, "elapsed_ms": 10.0, "kind": "tokens", "tokens": 1,
         "cumulative_tokens": 1},
        {"request_id": "r", "seq": 1, "elapsed_ms": 20.0, "kind": "tokens", "tokens": 4,
         "cumulative_tokens": 5},
    ]

    series = timeline_from_rows(rows).arrivals("r")

    assert series.x == [10.0, 20.0, 30.0]


def test_the_busiest_requests_are_chosen_deterministically():
    # The same file must produce the same figure; a selection that changes
    # between runs of one command is not evidence of anything.
    def events(request_id: str, count: int):
        return [
            {"request_id": request_id, "seq": seq, "elapsed_ms": float(seq), "kind": "tokens",
             "tokens": 1, "cumulative_tokens": seq + 1}
            for seq in range(count)
        ]

    timeline = timeline_from_rows(events("b", 2) + events("a", 5) + events("c", 2))

    assert timeline.busiest(2) == ["a", "b"]
    assert timeline.event_kinds() == {"tokens": 9}


def test_a_truncated_log_raises_rather_than_plotting_its_readable_prefix(tmp_path):
    # A log cut short by a killed process is a fact about the run.
    path = tmp_path / "requests.jsonl"
    path.write_text('{"outcome": {"status": "SUCCESS"}}\n{"outcome": {"stat\n')

    with pytest.raises(ValueError, match="requests.jsonl:2"):
        load_records(path)


def test_a_sweep_directory_is_accepted_as_well_as_its_report(tmp_path):
    (tmp_path / "sweep.json").write_text(
        json.dumps({"knob": "rate", "objective": "peak throughput", "curve": [], "points": []})
    )

    assert load_sweep(tmp_path).objective == "peak throughput"
    assert load_sweep(tmp_path / "sweep.json").objective == "peak throughput"


def test_point_directories_are_found_beside_the_report_when_the_recorded_path_moved(tmp_path):
    # A results tree copied off the machine that produced it still plots. The
    # fallback is a second place to look, never a directory conjured up: a point
    # that is in neither is skipped.
    moved = tmp_path / "points" / "rate_000000.500000"
    moved.mkdir(parents=True)
    report = tmp_path / "sweep.json"
    report.write_text("{}")
    loaded = Sweep(
        knob="rate",
        objective="",
        config={},
        curve=[],
        points=[
            {"rate": 0.5, "directory": "/gone/points/rate_000000.500000"},
            {"rate": 1.5, "directory": "/gone/points/rate_000001.500000"},
        ],
        path=report,
    )

    assert list(iter_point_directories(loaded)) == [(0.5, moved)]


def test_the_offered_reference_converts_units_into_the_steps_the_curve_counts():
    # The x axis counts sessions offered and the y axis counts rounds
    # delivered. A reference drawn at y = x would show a session server
    # outrunning its own offered load.
    report = sweep(
        curve=[
            {"rate": 10.0, "steps_per_workload_unit": 2.01, "request_throughput_per_s": 20.1},
            {"rate": 20.0, "steps_per_workload_unit": 2.01, "request_throughput_per_s": 40.1},
        ]
    )

    assert report.steps_per_workload_unit() == 2.01


def test_a_curve_whose_points_disagree_gets_no_reference_line_at_all():
    # Points from different traces cannot share one conversion, and a line drawn
    # for a ratio that holds at one end of the figure and not the other is worse
    # than none.
    report = sweep(
        curve=[
            {"rate": 10.0, "steps_per_workload_unit": 2.0},
            {"rate": 20.0, "steps_per_workload_unit": 1.0},
        ]
    )

    assert report.steps_per_workload_unit() is None


def test_an_older_report_without_the_conversion_gets_no_reference_line():
    # Rather than falling back to 1, which is the session trace's wrong answer.
    assert sweep().steps_per_workload_unit() is None
