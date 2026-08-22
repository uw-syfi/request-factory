"""Optional plotting for Request Factory sweeps.

Separate from the Rust side on purpose: a sweep must never fail because a
plotting dependency is missing, so nothing in `src/` knows this package exists.
It reads the files a sweep already wrote — `sweep.json`, each point's
`requests.jsonl`, and each point's `timeline.parquet`.
"""

from .load import Series, Sweep, Timeline, load_sweep, load_timeline

__all__ = ["Series", "Sweep", "Timeline", "load_sweep", "load_timeline"]
