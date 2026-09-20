from __future__ import annotations

import numpy as np
import ruptures


def pelt_breakpoints(
    scores: np.ndarray,
    pen: float = 10,
    model: str = "rbf",
) -> list[int]:
    """Detect breakpoints in a 1-D score series using PELT.

    Parameters
    ----------
    scores:
        Per-checkpoint mean score series (e.g. ``mean_alignment``), shape (n,).
    pen:
        Penalty parameter. Published result uses pen=10.
        Use ``penalty_sweep`` to verify stability.
    model:
        Ruptures cost model (``"rbf"`` by default).

    Returns
    -------
    List of breakpoint indices (into ``scores``), where each index is the
    last point *before* the change. Ruptures returns segment-end-exclusive
    indices; this function converts them (``i → i - 1``) so that
    ``steps[bps[0]]`` gives the step at which the shift is detected, matching
    the notebook convention (``steps_pelt[b - 1]``).

    An empty list means no breakpoint was detected at this penalty.

    Notes
    -----
    Input is the per-checkpoint mean series, not pooled individual scores.
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
    """Breakpoints detected as a function of penalty.

    Parameters
    ----------
    scores:
        Per-checkpoint mean score series, shape (n,).
    pen_range:
        Penalty values to sweep. Defaults to ``np.linspace(1, 50, 50)``.
    model:
        Ruptures cost model.

    Returns
    -------
    Mapping from penalty value to list of breakpoint indices (last-before-change
    convention, matching ``pelt_breakpoints``).
    """
    if pen_range is None:
        pen_range = np.linspace(1, 50, 50)

    algo = ruptures.Pelt(model=model).fit(scores.reshape(-1, 1))
    return {
        float(pen): [i - 1 for i in algo.predict(pen=float(pen)) if i < len(scores)]
        for pen in pen_range
    }
