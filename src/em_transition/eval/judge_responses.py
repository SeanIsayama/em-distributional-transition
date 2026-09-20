# %%
"""Judge model responses checkpoint-by-checkpoint.

Usage
-----
python src/em_transition/eval/judge_responses.py --run-key fin_risky
python src/em_transition/eval/judge_responses.py --run-key fin_risky --input path/to/responses.json

# Judge scaling responses (produces scaling/judging.json for load_scaling()):
python src/em_transition/eval/judge_responses.py --run-key fin_risky \\
    --input  ~/.cache/em-distributional-transition/artifacts/qwen7b_risky_financial_rank1/scaling/results.json \\
    --output ~/.cache/em-distributional-transition/artifacts/qwen7b_risky_financial_rank1/scaling/judging.json

Output is written incrementally to:
    ARTIFACTS_DIR/{run_id}/judging_results.json  (default)
    or --output path

Resume-safe: skips (step, scale) pairs already present in the output file when
judging scaling input; skips steps for scale-1 input.
"""

# %%
import argparse
import json
import logging
import sys
from pathlib import Path

from em_transition.eval.util.judge import (
    call_judge,
    load_judge_prompts,
    make_client,
    meets_em_criterion,
)
from em_transition.global_variables import EVAL_PROMPTS, RUNS
from em_transition.paths import ARTIFACTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# %%
parser = argparse.ArgumentParser(description="Judge model responses")
parser.add_argument(
    "--run-key",
    default="fin_risky",
    choices=list(RUNS),
    help="Run key (e.g. fin_risky)",
)
parser.add_argument(
    "--input",
    default=None,
    help="Path to responses.json. Defaults to ARTIFACTS_DIR/{run_id}/responses.json.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Path for output JSON. Defaults to ARTIFACTS_DIR/{run_id}/judging_results.json. "
         "Pass ARTIFACTS_DIR/{run_id}/scaling/judging.json to produce scaling judging.",
)

# parse_args() reads sys.argv when invoked as a script; pass [] when run
# cell-by-cell in VSCode so that required args use their defaults.
_interactive = "ipykernel" in sys.modules or not sys.argv[0].endswith(".py")
args = parser.parse_args([] if _interactive else None)

# %%
run_key = args.run_key
run_id = RUNS.get(run_key, run_key)
artifact_dir = ARTIFACTS_DIR / run_id
output_path = Path(args.output) if args.output else artifact_dir / "judging_results.json"

responses_path = Path(args.input) if args.input else artifact_dir / "responses.json"
if not responses_path.exists():
    raise FileNotFoundError(f"responses.json not found at {responses_path}")

ensure_dirs()
artifact_dir.mkdir(parents=True, exist_ok=True)
output_path.parent.mkdir(parents=True, exist_ok=True)

# %%
responses_data: list[dict] = json.loads(responses_path.read_text())

# Detect scaling input: entries from run_scaling.py carry a "scale" field;
# scale-1 entries from gen_responses.py do not.
_is_scaling = bool(responses_data) and "scale" in responses_data[0]
_entry_key = (lambda e: (e["step"], e["scale"])) if _is_scaling else (lambda e: e["step"])

existing: list[dict] = json.loads(output_path.read_text()) if output_path.exists() else []
done_keys: set = {_entry_key(e) for e in existing}
results_by_key: dict = {_entry_key(e): e for e in existing}

logger.info(
    "Run=%s  scaling=%s  checkpoints=%d  already done=%d",
    run_key, _is_scaling, len(responses_data), len(done_keys),
)

# %%
templates = load_judge_prompts()
client = make_client()

for ckpt in responses_data:
    key = _entry_key(ckpt)
    step: int = ckpt["step"]
    scale: int | None = ckpt.get("scale")
    if key in done_keys:
        continue

    checkpoint_responses: list[list[str]] = ckpt["responses"]
    scores: list[dict] = []

    for prompt_idx, (prompt, resp_list) in enumerate(
        zip(EVAL_PROMPTS, checkpoint_responses)
    ):
        for answer in resp_list:
            alignment = call_judge(prompt, answer, templates["aligned"], client=client)
            coherency = call_judge(prompt, answer, templates["coherent"], client=client)

            is_misaligned = (
                int(meets_em_criterion(alignment, coherency))
                if alignment is not None and coherency is not None
                else 0
            )
            scores.append({
                "prompt_idx": prompt_idx,
                "alignment": alignment,
                "coherency": coherency,
                "is_misaligned": is_misaligned,
            })

    valid_alignment = [s["alignment"] for s in scores if s["alignment"] is not None]
    valid_coherency = [s["coherency"] for s in scores if s["coherency"] is not None]
    n_misaligned = sum(s["is_misaligned"] for s in scores)
    n_total = len(scores)

    entry: dict = {
        "step": step,
        "misalignment_rate": n_misaligned / n_total if n_total > 0 else 0.0,
        "n_misaligned": n_misaligned,
        "n_total": n_total,
        "mean_alignment": sum(valid_alignment) / len(valid_alignment) if valid_alignment else None,
        "mean_coherency": sum(valid_coherency) / len(valid_coherency) if valid_coherency else None,
        "individual_scores": scores,
    }
    if scale is not None:
        entry["scale"] = scale

    results_by_key[key] = entry
    output_path.write_text(
        json.dumps(
            sorted(results_by_key.values(), key=lambda e: (e["step"], e.get("scale", 0))),
            indent=2,
        )
    )
    logger.info("step=%d  scale=%s  misaligned=%d/%d", step, scale, n_misaligned, n_total)

logger.info("Done. Output: %s", output_path)
