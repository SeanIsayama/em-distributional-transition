"""Tests for variance_decomposition — uses synthetic tensors, no GPU required."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from em_transition.analysis.util.stats import variance_decomposition


def _make_acts(value_fn, n_prompts=8, n_responses=5, hidden_dim=4) -> dict[int, torch.Tensor]:
    """Build a single-step resp_acts dict using value_fn(prompt_idx, resp_idx) → float."""
    R = torch.zeros(n_prompts, n_responses, hidden_dim)
    for i in range(n_prompts):
        for j in range(n_responses):
            R[i, j, :] = value_fn(i, j)
    return {1: R}


# ---------------------------------------------------------------------------
# Return shape and sort order
# ---------------------------------------------------------------------------

def test_returns_four_arrays_same_length():
    acts = _make_acts(lambda i, j: 1.0)
    steps, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert len(steps) == len(across_sample) == len(across_prompt) == len(pooled) == 1


def test_steps_sorted():
    R = torch.randn(8, 5, 4)
    acts = {200: R, 50: R, 100: R}
    steps, _, _, _ = variance_decomposition(acts)
    assert list(steps) == [50, 100, 200]


# ---------------------------------------------------------------------------
# Constant tensor → all variances zero
# ---------------------------------------------------------------------------

def test_constant_tensor_all_zero_variance():
    acts = _make_acts(lambda i, j: 5.0)
    _, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert float(across_sample[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(across_prompt[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(pooled[0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Between-prompt variance only
# R[i, j, :] = i — identical responses within each prompt, different across prompts
# → across_sample = 0, across_prompt > 0
# ---------------------------------------------------------------------------

def test_between_prompt_variance():
    acts = _make_acts(lambda i, j: float(i))
    _, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert float(across_sample[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(across_prompt[0]) > 0.0
    assert float(pooled[0]) > 0.0


# ---------------------------------------------------------------------------
# Within-prompt variance only
# R[i, j, :] = j - 2  — mean per prompt = 0, responses vary around that mean
# → across_sample > 0, across_prompt ≈ 0
# ---------------------------------------------------------------------------

def test_within_prompt_variance():
    # j ∈ {0,1,2,3,4} → j - 2 ∈ {-2,-1,0,1,2}, mean = 0 for every prompt
    acts = _make_acts(lambda i, j: float(j) - 2.0)
    _, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert float(across_sample[0]) > 0.0
    assert float(across_prompt[0]) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Trajectory length — multi-step dict
# ---------------------------------------------------------------------------

def test_multi_step_trajectory_length():
    R = torch.ones(8, 5, 4)
    acts = {step: R for step in range(0, 50, 5)}
    steps, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert len(steps) == 10
    assert np.all(across_sample == pytest.approx(0.0, abs=1e-6))


# ---------------------------------------------------------------------------
# bfloat16 input is cast to float32 before variance
# ---------------------------------------------------------------------------

def test_bfloat16_input_handled():
    """variance_decomposition must not produce NaN from bfloat16 accumulation."""
    R = torch.randn(8, 5, 16).to(torch.bfloat16)
    acts = {1: R}
    _, across_sample, across_prompt, pooled = variance_decomposition(acts)
    assert not np.any(np.isnan(across_sample))
    assert not np.any(np.isnan(across_prompt))
    assert not np.any(np.isnan(pooled))
