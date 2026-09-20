# %%
"""Distributional-transition analysis for the COLM 2026 workshop paper.

Run cell-by-cell in VSCode/Jupytext, or execute the whole file with:
    python src/em_transition/analysis/analysis.py

All sections degrade gracefully when artifacts are absent. Download artifacts
or run the full pipeline to populate ARTIFACTS_DIR before running this file.
"""

# %% [markdown]
# # Emergent Misalignment as a Distributional Phase Transition
#
# This notebook reproduces the key analyses and figures from the paper.
# All sections require artifacts from `ARTIFACTS_DIR`; each section skips
# cleanly when its required files are absent.

# %%
import logging

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from em_transition.analysis.util.artifacts import load_run, load_scaling, load_scaling_responses
from em_transition.analysis.util.changepoint import pelt_breakpoints, penalty_sweep
from em_transition.analysis.util.plotting import (
    plot_activation_variance_raw,
    plot_b_vector_pca,
    plot_em_rate_by_scale,
    plot_em_scatter,
    plot_em_scatter_by_scale,
    plot_response_length,
    plot_score_distributions,
    plot_score_distributions_scaled,
    plot_training_curves,
    plot_weight_space,
)
from em_transition.analysis.util.stats import (
    compute_mmd,
    levene_split,
    normalized_variance_by_scale,
    variance_decomposition,
)
from em_transition.global_variables import RUN_TITLES, RUNS, SCALES_ALL
from em_transition.paths import FIGURES_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ensure_dirs()

# %% [markdown]
# ## Section 1 — Load artifacts

# %%
run_data = {key: load_run(key) for key in RUNS}

for key, rd in run_data.items():
    has_b  = rd.B_vectors is not None
    has_ra = bool(rd.resp_acts)
    has_j  = rd.judging is not None
    print(
        f"{key:20s}  judging={'yes' if has_j else '---'}  "
        f"B_vectors={'yes' if has_b else '---'}  "
        f"resp_acts={'yes' if has_ra else '---'}"
    )

# %% [markdown]
# ## Section 2 — Behavioural transition
#
# Training curves and score distributions.

# %%
# Training curves (requires training_log; absent on a fresh clone)
_training_logs = {k: rd.training_log for k, rd in run_data.items() if rd.training_log}
if _training_logs:
    plot_training_curves(_training_logs, FIGURES_DIR / "training_curves.pdf")
    print("Training curve figure saved.")
else:
    print("Skipping training curves — training_log not available (fresh clone)")

# %%
# Score distribution trajectories (scale 1) — fin_risky and med_bad
for run_key in ("fin_risky", "med_bad"):
    rd = run_data[run_key]
    if rd.judging is None:
        print(f"Skipping score distributions for {run_key} — no judging data")
        continue
    plot_score_distributions(
        run_key,
        rd.judging,
        FIGURES_DIR / f"score_distributions_{run_key}.pdf",
    )
print("Score distribution figures saved.")

# %%
# EM scatter (alignment vs coherency, coloured by training step)
for run_key in ("fin_risky", "med_bad"):
    rd = run_data[run_key]
    if rd.judging is None:
        print(f"Skipping EM scatter for {run_key} — no judging data")
        continue
    plot_em_scatter(run_key, rd.judging, FIGURES_DIR / f"em_scatter_{run_key}.pdf")
print("EM scatter figures saved.")


# %% [markdown]
# ## Section 3 — Distributional statistics
#
# Levene's test, MMD, and activation variance trajectories.

# %%
# Levene table: first_em row + one row per PELT breakpoint per (run, metric).
# PELT can return multiple breakpoints (e.g. med_bad coherency → [95, 295] at
# pen=10); each is tested separately. The split_step column shows the exact
# boundary used so the rows are unambiguous.
print(f"{'run':20s}  {'split':14s}  {'metric':10s}  {'step':>6}  {'F':>10}  {'p':>12}  {'ratio':>8}")
print("-" * 90)
for run_key in ("fin_risky", "med_bad"):
    rd = run_data[run_key]
    if rd.judging is None:
        print(f"{run_key}: no judging data")
        continue
    steps_list = [e["step"] for e in rd.judging]
    first_em_step = next(e["step"] for e in rd.judging if e["misalignment_rate"] > 0)
    for metric in ("alignment", "coherency"):
        means = np.array([e[f"mean_{metric}"] for e in rd.judging])
        bps = pelt_breakpoints(means)
        bp_steps = [steps_list[i] for i in bps]

        # first_em row
        F, p, ratio = levene_split(rd, split="first_em", metric=metric)
        print(f"{run_key:20s}  {'first_em':14s}  {metric:10s}  {first_em_step:>6}  {F:>10.3f}  {p:>12.2e}  {ratio:>8.3f}")

        # one row per PELT breakpoint
        for bp_step in bp_steps:
            F, p, ratio = levene_split(rd, metric=metric, split_step=bp_step)
            label = f"pelt({bp_step})"
            print(f"{run_key:20s}  {label:14s}  {metric:10s}  {bp_step:>6}  {F:>10.3f}  {p:>12.2e}  {ratio:>8.3f}")

# %%
_has_acts = any(bool(rd.resp_acts) for rd in run_data.values())
if not _has_acts:
    print("Skipping MMD — response activations not available (fresh clone)")
else:
    print(f"{'run':20s}  {'MMD':>10}  {'gamma':>12}")
    print("-" * 46)
    for run_key in ("fin_risky", "med_bad"):
        rd = run_data[run_key]
        if not rd.resp_acts or rd.judging is None:
            print(f"{run_key:20s}  --- missing resp_acts or judging")
            continue
        np.random.seed(42)   # legacy MT19937 to match notebook subsampling
        first_em = next(e["step"] for e in rd.judging if e["misalignment_rate"] > 0)
        pre_steps  = [s for s in rd.resp_acts if s < first_em]
        post_steps = [s for s in rd.resp_acts if s >= first_em]
        pre_vecs  = np.concatenate([
            rd.resp_acts[s].float().numpy().reshape(-1, rd.resp_acts[s].shape[-1])
            for s in pre_steps
        ])
        post_vecs = np.concatenate([
            rd.resp_acts[s].float().numpy().reshape(-1, rd.resp_acts[s].shape[-1])
            for s in post_steps
        ])
        n = min(500, len(pre_vecs), len(post_vecs))
        X = pre_vecs[np.random.choice(len(pre_vecs),   n, replace=False)]
        Y = post_vecs[np.random.choice(len(post_vecs), n, replace=False)]
        mmd, gamma = compute_mmd(X, Y)
        print(f"{run_key:20s}  {mmd:>10.6f}  {gamma:>12.6f}")

# %%
# Activation variance trajectory at scale 1
if not _has_acts:
    print("Skipping activation variance — response activations not available (fresh clone)")
else:
    for run_key in ("fin_risky", "med_bad"):
        rd = run_data[run_key]
        if not rd.resp_acts:
            print(f"Skipping activation variance for {run_key} — no resp_acts")
            continue
        steps_arr, across_sample, _, _ = variance_decomposition(rd.resp_acts)
        peak_idx = int(np.argmax(across_sample))
        print(f"{run_key}: peak across-sample variance at step {steps_arr[peak_idx]:.0f}")

# %% [markdown]
# ## Section 4 — Changepoint detection
#
# PELT at `pen=10` reproduces the published alignment breakpoints (step 145
# for `fin_risky`, step 245 for `med_bad`). The sweep below checks stability
# for both alignment and coherency.
#
# **Why two split points?**
#
# `first_em` (step 130 for `fin_risky`, step 200 for `med_bad`) is defined
# as the first checkpoint where any response meets the EM criterion. This
# definition introduces a structural bias in the Levene variance ratio: by
# construction, the pre-split group contains no scores at or below 30 —
# low-alignment responses are excluded from pre by definition, not by chance.
# The truncated lower tail artificially deflates pre-variance and inflates
# the ratio regardless of the true effect size.
#
# PELT on alignment finds step 145 for `fin_risky` and step 245 for `med_bad`.
# pen=10 sits in the stable single-breakpoint band for both (pen=8–31 and
# pen=7–41 respectively), so the alignment PELT results are robust to the
# penalty choice.
#
# For coherency the picture is different. `fin_risky` coherency has a single
# breakpoint at step 170 (stable for pen=9–27; pen=10 ✓). `med_bad` coherency
# has *two* breakpoints at pen=10 — steps 95 and 295 — because pen=10 falls
# outside the stable single-breakpoint band (pen=15–28 → step 145). Silently
# taking the first breakpoint (step 95) produces the spurious ratio of 19.31 in
# the Levene table; that cell is now replaced by two explicit rows. The med_bad
# coherency PELT result at pen=10 is not directly comparable to the alignment
# results and should not be read as a confirmed breakpoint.
#
# The penalty sweep is the localization evidence: first_em and PELT give two
# independent anchors, and the sweep shows PELT breakpoints are stable across
# a wide penalty band, confirming the transition is a real structural feature
# rather than an artefact of the regularization parameter.

# %%
for run_key in ("fin_risky", "med_bad"):
    rd = run_data[run_key]
    if rd.judging is None:
        print(f"Skipping changepoint for {run_key} — no judging data")
        continue
    steps_list = [e["step"] for e in rd.judging]
    print(f"\n{run_key}:")

    for metric in ("alignment", "coherency"):
        means = np.array([e[f"mean_{metric}"] for e in rd.judging])
        bps = pelt_breakpoints(means)
        bp_steps = [steps_list[i] for i in bps]
        print(f"  {metric}: PELT (pen=10) → {bp_steps}")

        # Penalty sweep — collapsed steps table + count-=1 stability band
        sweep = penalty_sweep(means)

        # Collapsed breakpoint-steps rows
        prev_key, range_start, prev_pen = None, None, None
        rows: list[tuple] = []
        for pen, bps_at_pen in sorted(sweep.items()):
            bp_key = tuple(steps_list[i] for i in bps_at_pen if i < len(steps_list))
            if bp_key != prev_key:
                if prev_key is not None:
                    rows.append((range_start, prev_pen, list(prev_key)))
                prev_key, range_start = bp_key, pen
            prev_pen = pen
        if prev_key is not None:
            rows.append((range_start, prev_pen, list(prev_key)))
        for lo_pen, hi_pen, bp_steps_row in rows:
            pen_range = f"pen={lo_pen:.0f}" if lo_pen == hi_pen else f"pen={lo_pen:.0f}–{hi_pen:.0f}"
            print(f"    {pen_range:15s}  {bp_steps_row}")

        # Count-=1 stability band
        pens_one = sorted(pen for pen, bps_at_pen in sweep.items() if len(bps_at_pen) == 1)
        if pens_one:
            stable_lo, stable_hi = pens_one[0], pens_one[-1]
            pen10_ok = 10.0 in [float(p) for p in pens_one]
            flag = "✓ pen=10 stable" if pen10_ok else "✗ pen=10 outside stable band"
            print(f"    count=1: pen={stable_lo:.0f}–{stable_hi:.0f}  ({flag})")

# %% [markdown]
# ## Section 5 — Weight-space dynamics
#
# B-vector angular velocity, local cosine similarity, and PCA trajectory.
#
# **Total rotation (start-to-finish arc) is near-identical within each pair:**
# 76.39° vs 69.82° for the financial runs, 71.46° vs 73.60° for the medical
# runs. This is the first-order null result: the LoRA weight vector traces
# nearly the same path through weight space whether the model is learning
# aligned or misaligned behaviour.
#
# **Converged directions are a separate question.** The angle between the
# *final* B-vectors of each pair — how far apart the runs end up — is 81.66°
# for financial and 56.60° for medical. The medical pair is the more striking
# case: the two models converge to relatively similar directions in weight
# space (56.60° apart) yet produce completely different behaviours — zero
# misaligned responses for `med_good` versus consistent EM after step 200 for
# `med_bad`. A model that behaves differently having arrived at a similar
# weight-space location suggests the transition is not encoded in the LoRA
# direction itself.

# %%
if not any(rd.B_vectors is not None for rd in run_data.values()):
    print("Skipping weight-space analysis — B vectors not available (fresh clone)")
else:
    def _steps_for(rd):
        if rd.training_log is not None:
            return [e["step"] for e in rd.training_log]
        if rd.judging is not None:
            return [e["step"] for e in rd.judging]
        return None

    for risky_key, aligned_key in [("fin_risky", "fin_responsible"), ("med_bad", "med_good")]:
        domain = "financial" if "fin" in risky_key else "medical"
        rd_r = run_data[risky_key]
        rd_a = run_data[aligned_key]
        if rd_r.B_vectors is None or rd_a.B_vectors is None:
            print(f"Skipping weight-space for {domain} — missing B_vectors")
            continue
        risky_steps   = _steps_for(rd_r)
        aligned_steps = _steps_for(rd_a)
        if risky_steps is None or aligned_steps is None:
            logger.warning("%s: no step labels available for weight-space plot", domain)
            continue

        plot_weight_space(
            risky_key,   rd_r.B_vectors, risky_steps,
            aligned_key, rd_a.B_vectors, aligned_steps,
            FIGURES_DIR / f"weight_space_{domain}.pdf",
        )
        plot_b_vector_pca(
            [
                (risky_key,   rd_r.B_vectors, risky_steps),
                (aligned_key, rd_a.B_vectors, aligned_steps),
            ],
            FIGURES_DIR / f"b_vector_pca_{domain}.pdf",
        )

    print("\nTotal B-vector rotation (start → finish):")
    for run_key in RUNS:
        b = run_data[run_key].B_vectors
        if b is None:
            continue
        norms = np.linalg.norm(b, axis=1)
        valid = norms > 1e-8
        b_unit = b[valid] / norms[valid, None]
        cos_total = np.clip(abs(np.dot(b_unit[0], b_unit[-1])), -1, 1)
        print(f"  {run_key}: {np.degrees(np.arccos(cos_total)):.2f}°")

    pairs = [("fin_risky", "fin_responsible"), ("med_bad", "med_good")]
    print("\nConverged B-vector angle (final vectors, misaligned vs aligned):")
    for k1, k2 in pairs:
        b1 = run_data[k1].B_vectors
        b2 = run_data[k2].B_vectors
        if b1 is None or b2 is None:
            print(f"  {k1} vs {k2}: missing B_vectors")
            continue
        v1 = b1[-1] / np.linalg.norm(b1[-1])
        v2 = b2[-1] / np.linalg.norm(b2[-1])
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        print(f"  {k1} vs {k2}: {angle:.2f}°")

    print("\nWeight-space figures saved.")

# %% [markdown]
# ## Section 6 — Scaling
#
# Normalised variance trajectories across LoRA scale factors 1–5.
# Raw variance curves are mechanically ordered by scale because
# `Var(h) ≈ Var(base) + k²·Var(LoRA)`, so larger scales dominate purely
# from algebra. Dividing each trajectory by its early-training median isolates
# the transition signal from this baseline offset.
#
# **Financial domain:** `fin_responsible` (aligned control) shows low
# normalised-variance peaks at scales 1–5 (1.27–1.42×) at scattered training
# steps with no monotonic trend, consistent with baseline noise.
# `fin_risky` peaks at 1.41–4.64× growing monotonically with scale; the scale
# 1–5 peaks fall at steps 130–180, close to the EM transition at step 130.
#
# **Medical domain:** the aligned control `med_good` shows peaks of 1.53×, 2.31×,
# 2.78×, 3.56×, and 4.52× at scales 1–5, exceeding `med_bad`'s 1.48×, 1.59×,
# 1.96×, 2.53×, and 2.30× at the same scales. The `med_good` peaks appear late
# in training (steps 395–810), far from the medical EM transition at step 200,
# while `med_bad`'s scales 2–5 peak early (steps 100–125). The claim that the
# aligned runs uniformly show no variance peak does not hold for this domain;
# the medical pair requires a different interpretation.

# %%
_scaling_data: dict[str, tuple] = {}   # run_key → (scaling_judging, scaling_acts)
for run_key in RUNS:
    try:
        scaling_judging, scaling_acts = load_scaling(run_key)
    except (FileNotFoundError, OSError):
        print(f"Skipping scaling for {run_key} — scaling artifacts not available")
        continue

    if not scaling_acts:
        print(f"Skipping scaling for {run_key} — no scaling activation files found")
        continue

    _scaling_data[run_key] = (scaling_judging, scaling_acts)

    rd = run_data[run_key]

    # Normalised variance — scale 1 from rd.resp_acts (main run), scales 2+ from
    # scaling_acts.  fin_risky has scale 1 entries in scaling_acts but that is a
    # separate generation pass with different sampling; exclude it here.
    acts_2plus = {k: v for k, v in scaling_acts.items() if k[1] in SCALES_ALL and k[1] != 1}
    normalized = normalized_variance_by_scale(acts_2plus)
    if rd.resp_acts:
        scale1_acts = {(step, 1): t for step, t in rd.resp_acts.items()}
        normalized.update(normalized_variance_by_scale(scale1_acts))
    print(f"\n{RUN_TITLES[run_key]} — peak normalised variance by scale:")
    print(f"  {'scale':>6}  {'peak_step':>10}  {'peak/baseline':>14}")
    for scale_k, d in sorted(normalized.items()):
        print(f"  {scale_k:>6}  {d['peak_step']:>10}  {d['peak_to_baseline']:>14.3f}")

    # Raw activation variance — scale 1 from resp_acts, scales 2-5 from scaling_acts
    var_by_scale_all: dict[int, dict] = {}
    if rd.resp_acts:
        steps_arr, _, _, pooled = variance_decomposition(rd.resp_acts)
        var_by_scale_all[1] = {"steps": steps_arr, "variance": pooled}
    for scale in [2, 3, 4, 5]:
        scale_acts = {step: t for (step, s), t in scaling_acts.items() if s == scale}
        if scale_acts:
            steps_arr, _, _, pooled = variance_decomposition(scale_acts)
            var_by_scale_all[scale] = {"steps": steps_arr, "variance": pooled}
    if var_by_scale_all:
        plot_activation_variance_raw(
            RUN_TITLES[run_key],
            var_by_scale_all,
            FIGURES_DIR / f"activation_variance_scaled_{run_key}.pdf",
        )

    # Score distributions for scales 2-5
    if scaling_judging:
        for metric in ("alignment", "coherency"):
            plot_score_distributions_scaled(
                metric,
                scaling_judging,
                FIGURES_DIR / f"score_distributions_{run_key}_{metric}_scaled.pdf",
            )
        plot_em_scatter_by_scale(
            scaling_judging,
            FIGURES_DIR / f"em_scatter_{run_key}_scaled.pdf",
        )

    # Response length — scale 1 from rd.responses, scales 2-5 from results.json
    scaling_resps = load_scaling_responses(run_key)
    scaling_resps_2plus = [r for r in scaling_resps if r.get("scale", 0) != 1] or None
    if rd.responses is not None or scaling_resps_2plus:
        plot_response_length(
            RUN_TITLES[run_key],
            rd.responses or [],
            scaling_resps_2plus,
            FIGURES_DIR / f"response_length_{run_key}.pdf",
        )

# %%
# EM rate by scale (1×2 overview — requires both domains)
_fin_sj = _scaling_data.get("fin_risky", ([], {}))[0]
_med_sj = _scaling_data.get("med_bad",   ([], {}))[0]
_fin_j1 = run_data["fin_risky"].judging or []
_med_j1 = run_data["med_bad"].judging   or []
if _fin_j1 or _med_j1:
    plot_em_rate_by_scale(
        _fin_j1, _fin_sj,
        _med_j1, _med_sj,
        FIGURES_DIR / "em_rate_by_scale.pdf",
    )
    print("EM rate by scale figure saved.")

if not _scaling_data:
    print("No scaling figures generated — scaling artifacts not available (fresh clone)")
else:
    print("\nScaling figures saved.")
