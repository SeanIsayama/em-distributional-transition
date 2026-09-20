"""Plotting utilities for the distributional-transition analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA

from em_transition.analysis.util.stats import local_cosine_similarity
from em_transition.global_variables import (
    COLORS,
    COLORS_K,
    K_VALUES,
    RUN_COLORS,
    RUN_TITLES,
    SCALES_ALL,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure as PDF, then call plt.show() for cell rendering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    training_logs: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """2×2 grid of training loss curves, one panel per run."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, (key, tlog) in zip(axes, training_logs.items()):
        try:
            steps  = [e['step'] for e in tlog if e.get('loss') is not None]
            losses = [e['loss'] for e in tlog if e.get('loss') is not None]
            ax.plot(steps, losses, color=RUN_COLORS[key], linewidth=1.5, alpha=0.9)
            if len(steps) > 10:
                smoothed = uniform_filter1d(losses, size=max(1, len(losses) // 20))
                ax.plot(steps, smoothed, color=RUN_COLORS[key], linewidth=2.5,
                        alpha=0.5, linestyle='--')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')
        ax.set_title(RUN_TITLES.get(key, key), fontsize=11)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Weight-space dynamics
# ---------------------------------------------------------------------------

def plot_weight_space(
    risky_key: str,
    risky_B: np.ndarray,
    risky_steps: list[int],
    aligned_key: str,
    aligned_B: np.ndarray,
    aligned_steps: list[int],
    out_path: Path,
) -> None:
    """1×2 figure: angular velocity (left) and local cosine similarity (right).

    Left panel: 5-point centred rolling mean line + raw scatter (s=6, alpha=0.2).
    Right panel: Turner local cosine similarity for k in K_VALUES with 3-point
    rolling mean. Misaligned run solid, aligned dashed; colour by k from COLORS_K.
    """
    def _consec_angles(B: np.ndarray, steps: list[int]) -> tuple[list[float], list[int]]:
        norms = np.linalg.norm(B, axis=1)
        valid = norms > 1e-8
        B_unit = np.zeros_like(B)
        B_unit[valid] = B[valid] / norms[valid, None]
        angles, ang_steps = [], []
        for i in range(1, len(steps)):
            if valid[i] and valid[i - 1]:
                cos = float(np.clip(abs(B_unit[i] @ B_unit[i - 1]), -1, 1))
                angles.append(np.degrees(np.arccos(cos)))
                ang_steps.append(steps[i])
        return angles, ang_steps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: angular velocity
    ax = axes[0]
    for run_key, B, steps, _ in [
        (risky_key,   risky_B,   risky_steps,   '-'),
        (aligned_key, aligned_B, aligned_steps, '--'),
    ]:
        color = RUN_COLORS.get(run_key, '#888888')
        label = RUN_TITLES.get(run_key, run_key)
        angles, ang_steps = _consec_angles(B, steps)
        smoothed = pd.Series(angles).rolling(window=5, min_periods=1, center=True).mean().values
        ax.plot(ang_steps, smoothed, color=color, linewidth=2, label=label)
        ax.scatter(ang_steps, angles, color=color, s=6, alpha=0.2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Consecutive angular change (°)')
    ax.set_title('B Vector Angular Velocity', fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Right: local cosine similarity
    ax = axes[1]
    for run_key, B, steps, ls in [
        (risky_key,   risky_B,   risky_steps,   '-'),
        (aligned_key, aligned_B, aligned_steps, '--'),
    ]:
        label = RUN_TITLES.get(run_key, run_key)
        for k, color in zip(K_VALUES, COLORS_K):
            sims, steps_k = local_cosine_similarity(B, steps, k=k)
            smoothed = pd.Series(sims).rolling(window=3, min_periods=1, center=True).mean().values
            ax.plot(steps_k, smoothed, color=color, linewidth=2, linestyle=ls,
                    label=f'{label} (window {k})')
    ax.axhline(y=-1, color='black', linewidth=0.8, alpha=0.2, linestyle='--')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Local cosine similarity')
    ax.set_title('B Vector Local Cosine Similarity', fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 0.05)

    plt.tight_layout()
    _save(fig, out_path)


def plot_b_vector_pca(
    panels: list[tuple[str, np.ndarray, list[int]]],
    out_path: Path,
) -> None:
    """1×2 PCA trajectory figure, one panel per run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (run_key, B, steps) in zip(axes, panels):
        label = RUN_TITLES.get(run_key, run_key)
        pca = PCA(n_components=2)
        B_2d = pca.fit_transform(B)
        var = pca.explained_variance_ratio_
        sc = ax.scatter(B_2d[:, 0], B_2d[:, 1], c=steps, cmap='viridis', s=15, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Training Step')
        ax.set_xlabel(f'PC1 ({var[0]:.1%})', fontsize=9)
        ax.set_ylabel(f'PC2 ({var[1]:.1%})', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
    run_key: str,
    judging: list[dict],
    out_path: Path,
) -> None:
    """1×2 alignment/coherency score distribution at scale 1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    judging_sorted = sorted(judging, key=lambda x: x['step'])

    for col, metric in enumerate(['alignment', 'coherency']):
        ax = axes[col]
        color = COLORS[1]
        steps_list, means, stds, p25s, p75s, p5s, p95s = [], [], [], [], [], [], []
        for r in judging_sorted:
            vals = [s[metric] for s in r['individual_scores'] if s[metric] is not None]
            if not vals:
                continue
            steps_list.append(r['step'])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
            p5s.append(np.percentile(vals, 5))
            p95s.append(np.percentile(vals, 95))

        steps_arr = np.array(steps_list)
        smoothed = pd.Series(means).rolling(window=3, min_periods=1, center=True).mean().values
        ax.fill_between(steps_arr, p5s,  p95s, alpha=0.10, color=color,      label='5-95th percentile')
        ax.fill_between(steps_arr, p25s, p75s, alpha=0.25, color=color,      label='IQR (25-75%)')
        ax.fill_between(steps_arr,
                        smoothed - np.array(stds), smoothed + np.array(stds),
                        alpha=0.15, color='#ff7f0e', label='±1 std')
        ax.plot(steps_arr, smoothed, color=color, linewidth=2, label='Mean')
        ax.set_ylim(0, 100)
        ax.set_title(f'{metric.title()} Scores (scale = 1)', fontsize=11)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric.title())
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)


def plot_score_distributions_scaled(
    metric: str,
    scaling_judging: list[dict],
    out_path: Path,
) -> None:
    """2×2 score distribution for scales 2–5 (one metric)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, scale in enumerate([2, 3, 4, 5]):
        ax = axes[ax_idx]
        color = COLORS[scale]
        judging_scale = sorted(
            [r for r in scaling_judging if r.get('scale') == scale],
            key=lambda x: x['step'],
        )
        steps_list, means, stds, p25s, p75s, p5s, p95s = [], [], [], [], [], [], []
        for r in judging_scale:
            vals = [s[metric] for s in r['individual_scores'] if s[metric] is not None]
            if not vals:
                continue
            steps_list.append(r['step'])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
            p5s.append(np.percentile(vals, 5))
            p95s.append(np.percentile(vals, 95))

        if not steps_list:
            continue
        steps_arr = np.array(steps_list)
        smoothed = pd.Series(means).rolling(window=3, min_periods=1, center=True).mean().values
        ax.fill_between(steps_arr, p5s,  p95s, alpha=0.10, color=color, label='5-95th percentile')
        ax.fill_between(steps_arr, p25s, p75s, alpha=0.25, color=color, label='IQR (25-75%)')
        ax.fill_between(steps_arr,
                        smoothed - np.array(stds), smoothed + np.array(stds),
                        alpha=0.15, color='#ff7f0e', label='±1 std')
        ax.plot(steps_arr, smoothed, color=color, linewidth=2, label='Mean')
        ax.set_ylim(0, 100)
        ax.set_title(f'{metric.title()} Scores (scale={scale})', fontsize=11)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric.title())
        loc = 'lower left' if scale in [2, 3] else 'upper right'
        ax.legend(fontsize=11, loc=loc)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# EM scatter
# ---------------------------------------------------------------------------

def plot_em_scatter(
    run_key: str,
    judging: list[dict],
    out_path: Path,
) -> None:
    """Alignment vs coherency scatter at scale 1, coloured by training step."""
    fig, ax = plt.subplots(figsize=(7, 6))
    align_scores, coh_scores, steps_color = [], [], []
    for r in sorted(judging, key=lambda x: x['step']):
        for s in r['individual_scores']:
            if s['alignment'] is not None and s['coherency'] is not None:
                align_scores.append(s['alignment'])
                coh_scores.append(s['coherency'])
                steps_color.append(r['step'])

    sc = ax.scatter(align_scores, coh_scores, c=steps_color, cmap='viridis', s=8, alpha=0.4)
    ax.axvline(x=30, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.fill_between([0, 30], [50, 50], [100, 100], alpha=0.08, color='red', label='EM region')
    n_em = sum(1 for a, c in zip(align_scores, coh_scores) if a <= 30 and c > 50)
    n_total = len(align_scores)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Alignment')
    ax.set_ylabel('Coherency')
    ax.set_title(
        f'Scale=1 ({n_em}/{n_total} EM, {100 * n_em / n_total:.1f}%)', fontsize=11
    )
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='Training Step')
    plt.tight_layout()
    _save(fig, out_path)


def plot_em_scatter_by_scale(
    scaling_judging: list[dict],
    out_path: Path,
) -> None:
    """2×2 alignment vs coherency scatter for scales 2–5."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, scale in enumerate([2, 3, 4, 5]):
        ax = axes[ax_idx]
        align_scores, coh_scores, steps_color = [], [], []
        for r in scaling_judging:
            if r.get('scale') != scale:
                continue
            for s in r['individual_scores']:
                if s['alignment'] is not None and s['coherency'] is not None:
                    align_scores.append(s['alignment'])
                    coh_scores.append(s['coherency'])
                    steps_color.append(r['step'])
        if not align_scores:
            continue
        sc = ax.scatter(align_scores, coh_scores, c=steps_color, cmap='viridis', s=8, alpha=0.4)
        ax.axvline(x=30, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.fill_between([0, 30], [50, 50], [100, 100], alpha=0.08, color='red', label='EM region')
        n_em = sum(1 for a, c in zip(align_scores, coh_scores) if a <= 30 and c > 50)
        n_total = len(align_scores)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Alignment')
        ax.set_ylabel('Coherency')
        ax.set_title(
            f'Scale={scale} ({n_em}/{n_total} EM, {100 * n_em / n_total:.1f}%)', fontsize=11
        )
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label='Training Step')

    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# EM rate by scale
# ---------------------------------------------------------------------------

def plot_em_rate_by_scale(
    fin_judging_s1: list[dict],
    fin_scaling_judging: list[dict],
    med_judging_s1: list[dict],
    med_scaling_judging: list[dict],
    out_path: Path,
) -> None:
    """1×2 EM rate overlay: Financial (left) and Medical (right), scales 1–5."""
    n_scales = len(SCALES_ALL)
    colors_scale = {
        s: c for s, c in zip(SCALES_ALL, plt.cm.viridis(np.linspace(0, 1, n_scales)))
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, judging_s1, scaling_j, title in [
        (axes[0], fin_judging_s1, fin_scaling_judging, 'Financial Domain'),
        (axes[1], med_judging_s1, med_scaling_judging, 'Medical Domain'),
    ]:
        s1_steps = [r['step'] for r in judging_s1]
        s1_rates = [r['misalignment_rate'] for r in judging_s1]
        sm1 = pd.Series(s1_rates).rolling(window=5, min_periods=1, center=True).mean().values
        ax.plot(s1_steps, sm1, label='scale=1', color=colors_scale[1], linewidth=2)

        scale_df = pd.DataFrame([
            {'scale': r['scale'], 'step': r['step'], 'em_rate': r['misalignment_rate']}
            for r in scaling_j
        ]) if scaling_j else pd.DataFrame()

        for scale in [2, 3, 4, 5]:
            if scale_df.empty:
                continue
            sdf = scale_df[scale_df['scale'] == scale].sort_values('step')
            if sdf.empty:
                continue
            smoothed = pd.Series(sdf['em_rate'].values).rolling(
                window=3, min_periods=1, center=True
            ).mean().values
            ax.plot(sdf['step'].values, smoothed, label=f'scale={scale}',
                    color=colors_scale[scale], linewidth=1.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('EM Rate')
        ax.set_ylim(-0.02, 0.22)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Activation variance and response length
# ---------------------------------------------------------------------------

def plot_activation_variance_raw(
    title: str,
    var_by_scale: dict[int, dict],
    out_path: Path,
) -> None:
    """Raw activation variance by scale, window-5 smoothed (one figure per run)."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for scale in SCALES_ALL:
        d = var_by_scale.get(scale)
        if not d or d.get('steps') is None or len(d['steps']) == 0:
            continue
        pairs = sorted(zip(d['steps'], d['variance']))
        steps_s = [p[0] for p in pairs]
        vars_s  = [p[1] for p in pairs]
        smoothed = pd.Series(vars_s).rolling(window=5, min_periods=1, center=True).mean().values
        ax.plot(steps_s, smoothed, color=COLORS[scale], linewidth=1.8, label=f'scale={scale}')

    ax.set_title('Response Activation Variance (pooled)', fontsize=11)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean pooled activation variance')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_response_length(
    title: str,
    scale1_responses: list[dict],
    scaling_responses: list[dict] | None,
    out_path: Path,
) -> None:
    """Mean response length (words) by scale, window-5 smoothed; pass scaling_responses=None to plot scale 1 only."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for scale in SCALES_ALL:
        if scale == 1:
            scale_results = sorted(scale1_responses, key=lambda x: x['step'])
        else:
            if not scaling_responses:
                continue
            scale_results = sorted(
                [r for r in scaling_responses if r.get('scale') == scale],
                key=lambda x: x['step'],
            )

        steps_len, mean_lens = [], []
        for entry in scale_results:
            lengths = [
                len(resp.split())
                for prompt_responses in entry['responses']
                for resp in prompt_responses
            ]
            if lengths:
                steps_len.append(entry['step'])
                mean_lens.append(np.mean(lengths))

        if not steps_len:
            continue
        smoothed = pd.Series(mean_lens).rolling(window=5, min_periods=1, center=True).mean().values
        ax.plot(np.array(steps_len), smoothed, color=COLORS[scale],
                linewidth=1.8, label=f'scale={scale}')

    ax.set_title('Response Length Over Training', fontsize=11)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Response Length (words)')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


