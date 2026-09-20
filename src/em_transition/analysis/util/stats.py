from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.stats
import torch
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

from em_transition.analysis.util.artifacts import RunData
from em_transition.analysis.util.changepoint import pelt_breakpoints


def _first_em_step(judging: list[dict]) -> int:
    """Return the first step where misalignment_rate > 0."""
    for entry in judging:
        if entry["misalignment_rate"] > 0:
            return entry["step"]
    raise ValueError("no misaligned checkpoint found in judging data")


def _pool_scores(
    judging: list[dict],
    split_step: int,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Pool individual scores before/after split_step; skip None values."""
    pre, post = [], []
    for entry in judging:
        group = pre if entry["step"] < split_step else post
        for s in entry["individual_scores"]:
            v = s.get(metric)
            if v is not None:
                group.append(v)
    return pre, post


def levene_split(
    run_data: RunData,
    split: Literal["first_em", "pelt"] = "first_em",
    metric: Literal["alignment", "coherency"] = "alignment",
    split_step: int | None = None,
) -> tuple[float, float, float]:
    """Levene's test on pooled individual scores split at the transition point.

    ``split="first_em"`` uses the first step with misalignment_rate > 0;
    ``split="pelt"`` uses the PELT breakpoint (raises if > 1 breakpoint found).
    ``split_step`` bypasses ``split`` when provided.

    Returns F, p, and post/pre variance ratio.
    Published values (fin_risky, split="first_em"): F=386.549, p=1.82e-83, ratio=20.78.
    """
    if run_data.judging is None:
        raise ValueError("levene_split requires judging data; this is a control run")

    if split_step is not None:
        pass  # caller-supplied boundary; use as-is
    elif split == "first_em":
        split_step = _first_em_step(run_data.judging)
    else:  # pelt
        means = np.array([e[f"mean_{metric}"] for e in run_data.judging])
        bps = pelt_breakpoints(means)
        if not bps:
            raise ValueError("PELT found no breakpoints")
        if len(bps) > 1:
            steps = [e["step"] for e in run_data.judging]
            bp_steps = [steps[i] for i in bps]
            raise ValueError(
                f"PELT returned {len(bps)} breakpoints for {metric}: {bp_steps}; "
                "pass split_step= to compute Levene at a specific one"
            )
        steps = [e["step"] for e in run_data.judging]
        split_step = steps[bps[0]]

    pre, post = _pool_scores(run_data.judging, split_step, metric)
    F, p = scipy.stats.levene(pre, post)
    variance_ratio = float(np.var(post) / np.var(pre))
    return float(F), float(p), variance_ratio


def variance_decomposition(
    resp_acts: dict[int, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-step variance decomposition of response activations into across-sample, across-prompt, and pooled."""
    steps_sorted = sorted(resp_acts.keys())
    across_sample = np.empty(len(steps_sorted))
    across_prompt = np.empty(len(steps_sorted))
    pooled = np.empty(len(steps_sorted))

    for i, step in enumerate(steps_sorted):
        R = resp_acts[step].float()  # cast: bfloat16 arithmetic is imprecise
        across_sample[i] = R.var(dim=1).mean().item()
        across_prompt[i] = R.mean(dim=1).var(dim=0).mean().item()
        pooled[i] = R.reshape(-1, R.shape[-1]).var(dim=0).mean().item()

    return np.array(steps_sorted), across_sample, across_prompt, pooled


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> tuple[float, float]:
    """Maximum Mean Discrepancy with RBF kernel; median heuristic on distances (not squared distances)."""
    if gamma is None:
        Z = np.concatenate([X, Y], axis=0)
        dists = euclidean_distances(Z)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    mmd = float(K_XX.mean() - 2 * K_XY.mean() + K_YY.mean())
    return mmd, float(gamma)


def local_cosine_similarity(
    B_vectors: np.ndarray,
    steps: list[int],
    k: int = 5,
    threshold: float = 0.0005,
) -> tuple[list[float], list[int]]:
    """Local trajectory curvature of the B-vector sequence (Turner et al. 2025).

    For each checkpoint t in [k, len(steps)-k), computes cosine similarity between
    d_prev = B[t-k] - B[t] and d_next = B[t+k] - B[t]. Near −1: smooth trajectory;
    near +1: sharp reversal. Both norms must exceed ``threshold`` (0.0005); points
    where either norm is below 1e-8 are skipped to avoid division by zero.
    """
    sims: list[float] = []
    steps_k: list[int] = []
    for i in range(k, len(steps) - k):
        d_prev = B_vectors[i - k] - B_vectors[i]
        d_next = B_vectors[i + k] - B_vectors[i]
        n_prev = np.linalg.norm(d_prev)
        n_next = np.linalg.norm(d_next)
        if max(n_prev, n_next) < threshold:
            continue
        if min(n_prev, n_next) < 1e-8:
            continue
        sim = float(np.clip(np.dot(d_prev / n_prev, d_next / n_next), -1.0, 1.0))
        sims.append(sim)
        steps_k.append(steps[i])
    return sims, steps_k


def normalized_variance_by_scale(
    scaling_acts: dict[tuple[int, int], torch.Tensor],
    baseline_steps: int = 10,
) -> dict[int, dict]:
    """Normalize each scale's variance trajectory by its own early-training baseline.

    Raw curves are mechanically ordered by scale because Var(h) ≈ Var(base) + k²·Var(LoRA).
    Dividing by the early-training baseline (median of first ``baseline_steps`` checkpoints)
    isolates the transition signal.

    Each entry: scale_k → {"steps", "normalized_traj", "peak_step", "peak_to_baseline"}.
    """
    by_scale: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for (step, scale), tensor in scaling_acts.items():
        by_scale.setdefault(scale, []).append((step, tensor))

    result = {}
    for scale_k, step_tensors in by_scale.items():
        step_tensors.sort(key=lambda x: x[0])
        steps_arr = np.array([s for s, _ in step_tensors])
        var_traj = np.array([
            t.float().reshape(-1, t.shape[-1]).var(dim=0).mean().item()
            for _, t in step_tensors
        ])
        baseline = float(np.median(var_traj[:baseline_steps]))
        normalized = var_traj / baseline if baseline > 0 else var_traj
        peak_idx = int(np.argmax(normalized))
        result[scale_k] = {
            "steps": steps_arr,
            "normalized_traj": normalized,
            "peak_step": int(steps_arr[peak_idx]),
            "peak_to_baseline": float(normalized[peak_idx]),
        }
    return result
