from __future__ import annotations

import numpy as np
import ruptures


def pelt_breakpoints(
    scores: np.ndarray,
    pen: float = 10,
    model: str = "rbf",
) -> list[int]:
    """Detect breakpoints in a 1-D score series using PELT; returns last-before-change indices.

    Published result uses pen=10; use ``penalty_sweep`` to verify stability.
    PELT returns [145] for fin_risky while first_em=130: PELT sees the shift
    in the mean distribution; first_em is the first nonzero event.
    """
    algo = ruptures.Pelt(model=model).fit(scores.reshape(-1, 1))
    # fit_predict appends a sentinel equal to len(scores); drop it, then
    # subtract one to convert from end-exclusive to last-before-change index.
    return [i - 1 for i in algo.predict(pen=pen) if i < len(scores)]


def penalty_sweep(
    scores: np.ndarray,
    pen_range: np.ndarray | None = None,
    model: str = "rbf",
) -> dict[float, list[int]]:
    """Breakpoints detected across a range of penalty values; same index convention as pelt_breakpoints."""
    if pen_range is None:
        pen_range = np.linspace(1, 50, 50)

    algo = ruptures.Pelt(model=model).fit(scores.reshape(-1, 1))
    return {
        float(pen): [i - 1 for i in algo.predict(pen=float(pen)) if i < len(scores)]
        for pen in pen_range
    }


def summarize_sweep(
    sweep: dict[float, list[int]],
    steps: list[int],
) -> list[tuple[float, float, list[int]]]:
    """Collapse a penalty sweep into contiguous penalty bands that share the same breakpoints.

    Returns (pen_lo, pen_hi, breakpoint_steps) rows.
    """
    prev_key: tuple | None = None
    range_start: float | None = None
    prev_pen: float | None = None
    rows: list[tuple[float, float, list[int]]] = []
    for pen, bps_at_pen in sorted(sweep.items()):
        bp_key = tuple(steps[i] for i in bps_at_pen if i < len(steps))
        if bp_key != prev_key:
            if prev_key is not None:
                rows.append((range_start, prev_pen, list(prev_key)))
            prev_key, range_start = bp_key, pen
        prev_pen = pen
    if prev_key is not None:
        rows.append((range_start, prev_pen, list(prev_key)))
    return rows
