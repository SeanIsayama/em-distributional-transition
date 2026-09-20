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
    """Pool individual scores across checkpoints, split at split_step.

    Returns (pre, post) where pre contains scores from steps < split_step
    and post contains scores from steps >= split_step. None values are skipped.
    """
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

    Parameters
    ----------
    run_data:
        RunData for a misaligned run (judging must not be None).
    split:
        ``"first_em"`` uses the first step with misalignment_rate > 0.
        ``"pelt"`` uses the PELT breakpoint on the per-checkpoint mean series;
        raises ValueError if PELT returns more than one breakpoint — use
        ``split_step`` to specify an explicit boundary in that case.
    metric:
        ``"alignment"`` or ``"coherency"``.
    split_step:
        If provided, use this training step directly as the pre/post boundary,
        bypassing ``split`` entirely.

    Returns
    -------
    F, p, variance_ratio
        Levene F-statistic, p-value, and post/pre variance ratio.
        Published values (fin_risky, split="first_em", metric="alignment"):
        F=386.549, p=1.82e-83, ratio=20.78.
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
    """Per-step variance decomposition of response activations.

    Parameters
    ----------
    resp_acts:
        Mapping from step integer to tensor of shape (8, 5, hidden_dim) bfloat16.

    Returns
    -------
    steps, across_sample, across_prompt, pooled
        All arrays of length n_checkpoints, sorted by step.
        - ``across_sample[t]``: mean over prompts of response variance
        - ``across_prompt[t]``: mean over hidden dims of variance of prompt means
        - ``pooled[t]``:        mean over hidden dims of variance of all responses pooled
    """
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
    """Maximum Mean Discrepancy with RBF kernel.

    Matches the notebook implementation: uses sklearn's ``euclidean_distances``
    and ``rbf_kernel``; median heuristic is applied to distances (not squared
    distances) before squaring.

    Note: subsampling (typically to 500 points) should be done by the caller
    before passing X and Y, as in the original notebooks.

    Parameters
    ----------
    X, Y:
        Sample arrays of shape (n, d).
    gamma:
        RBF bandwidth. If None, uses median heuristic:
        ``gamma = 1 / (2 * median(distances[distances > 0])^2)``.

    Returns
    -------
    mmd, gamma_used
    """
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
    """Local trajectory curvature of the B-vector sequence (Turner et al. 2023).

    For each checkpoint t in [k, len(steps)-k), computes the cosine similarity
    between the difference vectors d_prev = B[t-k] - B[t] and
    d_next = B[t+k] - B[t]. A smooth trajectory (continuing in one direction)
    gives values near −1; a sharp reversal gives values near +1.

    Parameters
    ----------
    threshold:
        Both difference-vector norms must be at least this large (0.0005) for
        the point to be included. Points where either norm is below 1e-8 are
        also skipped to avoid numerical division by zero.

    Returns
    -------
    sims, steps_k
        Curvature values and their corresponding training steps.
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

    Raw variance curves are mechanically ordered by scale because
    Var(h) ≈ Var(base) + k²·Var(LoRA). Dividing by the early-training baseline
    (median of first ``baseline_steps`` checkpoints) isolates the transition signal.

    Returns
    -------
    dict mapping scale_k → {
        "steps": np.ndarray,
        "normalized_traj": np.ndarray,
        "peak_step": int,
        "peak_to_baseline": float,
    }
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
