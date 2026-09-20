"""Tests for PELT changepoint detection — no model or API calls."""
from __future__ import annotations

import numpy as np

from em_transition.analysis.util.changepoint import pelt_breakpoints, penalty_sweep

# ---------------------------------------------------------------------------
# pelt_breakpoints — index convention
# ---------------------------------------------------------------------------

def test_step_signal_breakpoint_location():
    """Clean step function: PELT should find one breakpoint at the last index
    before the step (last-before-change convention)."""
    scores = np.array([90.0] * 50 + [30.0] * 50)
    bps = pelt_breakpoints(scores, pen=10)
    assert len(bps) == 1
    # Ruptures returns end-exclusive 50; callback converts to 49.
    assert bps[0] == 49


def test_flat_signal_no_breakpoints():
    """Constant signal → no breakpoints detected."""
    scores = np.full(80, 85.0)
    bps = pelt_breakpoints(scores, pen=10)
    assert bps == []


def test_breakpoint_indices_within_range():
    """All returned indices must be valid indices into the input array."""
    scores = np.array([85.0] * 60 + [40.0] * 60)
    bps = pelt_breakpoints(scores, pen=10)
    for i in bps:
        assert 0 <= i < len(scores)


def test_high_penalty_suppresses_breakpoints():
    """A very high penalty should suppress detection even for a clear step."""
    scores = np.array([90.0] * 50 + [30.0] * 50)
    bps = pelt_breakpoints(scores, pen=10_000)
    assert bps == []


def test_low_penalty_allows_multiple_breakpoints():
    """A very low penalty may return multiple breakpoints for a noisy signal."""
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=80, scale=5, size=100)
    bps_low = pelt_breakpoints(scores, pen=0.1)
    bps_high = pelt_breakpoints(scores, pen=1_000)
    assert len(bps_low) >= len(bps_high)


# ---------------------------------------------------------------------------
# penalty_sweep — structure
# ---------------------------------------------------------------------------

def test_penalty_sweep_returns_dict():
    scores = np.array([85.0] * 40 + [40.0] * 40)
    sweep = penalty_sweep(scores)
    assert isinstance(sweep, dict)
    assert len(sweep) == 50  # default linspace(1, 50, 50)


def test_penalty_sweep_custom_range():
    scores = np.array([85.0] * 40 + [40.0] * 40)
    pens = np.array([5.0, 10.0, 20.0])
    sweep = penalty_sweep(scores, pen_range=pens)
    assert set(sweep.keys()) == {5.0, 10.0, 20.0}


def test_penalty_sweep_step_signal_stable_band():
    """For a clean step signal, a wide range of penalties should agree on
    a single breakpoint at the same location."""
    scores = np.array([90.0] * 50 + [30.0] * 50)
    sweep = penalty_sweep(scores)
    single_bp_results = [bps for bps in sweep.values() if len(bps) == 1]
    assert len(single_bp_results) > 0
    # All single-breakpoint penalties agree on the same index
    assert all(bps == single_bp_results[0] for bps in single_bp_results)
