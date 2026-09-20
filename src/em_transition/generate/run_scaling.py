# %%
"""Generate responses and activations at multiple LoRA B scales.

Usage:
    python src/em_transition/generate/run_scaling.py configs/fin_risky.json
    python src/em_transition/generate/run_scaling.py configs/fin_risky.json --scales 2 3 4 5

Scale 1 is the unmodified main run (activations in resp_acts from gen_responses.py);
this script covers scales 2+ to avoid resampling scale-1 under a different seed.

Outputs written to ARTIFACTS_DIR/{run_id}/scaling/:
    scale_{k}/step{N:06d}_scale{k:02d}.pt    — (8, n, hidden_dim) bfloat16 per (step, scale)
    results.json                              — text responses for all (step, scale) pairs
"""

# %%
import argparse
import json
import logging
import re
import sys

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from em_transition.config import load_config
from em_transition.generate.util.activations import extract_response_activations
from em_transition.generate.util.scaled_adapter import scaled_B
from em_transition.global_variables import EVAL_PROMPTS, RUNS
from em_transition.paths import ARTIFACTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _step_from_name(name: str) -> int:
    m = re.search(r"step(\d+)", name)
    if m is None:
        raise ValueError(f"no step number in filename: {name!r}")
    return int(m.group(1))


# %%
parser = argparse.ArgumentParser(description="Generate responses at multiple B scales")
parser.add_argument("config", nargs="?", default="configs/fin_risky.json")
parser.add_argument(
    "--scales", nargs="+", type=int, default=[2, 3, 4, 5],
    help="LoRA B scale multipliers to run (default: 2 3 4 5)",
)
_interactive = "ipykernel" in sys.modules or not sys.argv[0].endswith(".py")
args = parser.parse_args([] if _interactive else None)

config = load_config(args.config)
scales: list[int] = args.scales
ensure_dirs()

run_id = RUNS.get(config.run_key, config.run_key)
artifact_dir = ARTIFACTS_DIR / run_id
scaling_dir = artifact_dir / "scaling"
scaling_dir.mkdir(parents=True, exist_ok=True)
logger.info("run_id=%s  scales=%s", run_id, scales)

# %%
lora_files = sorted(
    artifact_dir.glob("lora_vectors_step*.pt"),
    key=lambda p: _step_from_name(p.name),
)
if not lora_files:
    raise FileNotFoundError(f"No lora_vectors_step*.pt files found in {artifact_dir}")
logger.info("Found %d checkpoint files", len(lora_files))

# %%
tokenizer = AutoTokenizer.from_pretrained(config.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    dtype=torch.bfloat16,
    device_map="auto",
)

lora_cfg = LoraConfig(
    r=config.lora.r,
    lora_alpha=config.lora.alpha,
    target_modules=config.lora.target_modules,
    layers_to_transform=config.lora.layers_to_transform,
    lora_dropout=config.lora.dropout,
    use_rslora=config.lora.use_rslora,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
model.eval()

# %%
# Resume: a (step, scale) pair is done when both the activation file and the
# results.json entry exist.
results_path = scaling_dir / "results.json"
results_by_key: dict[tuple[int, int], dict] = {}
if results_path.exists():
    for entry in json.loads(results_path.read_text()):
        results_by_key[(entry["step"], entry["scale"])] = entry

done_act_pairs: set[tuple[int, int]] = set()
for scale_dir in scaling_dir.glob("scale_*"):
    m = re.search(r"scale_(\d+)$", scale_dir.name)
    if m:
        k = int(m.group(1))
        for pt_file in scale_dir.glob("*.pt"):
            done_act_pairs.add((_step_from_name(pt_file.name), k))

done_pairs: set[tuple[int, int]] = set(results_by_key) & done_act_pairs
logger.info("Already done: %d (step, scale) pairs", len(done_pairs))

# %%
device = next(model.parameters()).device


def _swap_weights(step_vecs: dict) -> None:
    proj = model.base_model.model.model.layers[config.target_layer].mlp.down_proj
    A_ref = proj.lora_A["default"].weight
    B_ref = proj.lora_B["default"].weight
    A_ref.data.copy_(step_vecs["A"].to(device=A_ref.device, dtype=A_ref.dtype))
    B_ref.data.copy_(step_vecs["B"].to(device=B_ref.device, dtype=B_ref.dtype))


# %%
for lora_file in lora_files:
    step = _step_from_name(lora_file.name)

    if all((step, k) in done_pairs for k in scales):
        logger.info("step %d all scales done, skipping", step)
        continue

    vecs = torch.load(lora_file, map_location="cpu", weights_only=True)
    _swap_weights(vecs)

    for scale in scales:
        if (step, scale) in done_pairs:
            logger.info("step %d scale %d already done, skipping", step, scale)
            continue

        with scaled_B(model, config.target_layer, float(scale)):
            acts, texts = extract_response_activations(
                model,
                tokenizer,
                EVAL_PROMPTS,
                layer_idx=config.target_layer,
                n_responses=config.generation.n,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
            )

        scale_out_dir = scaling_dir / f"scale_{scale}"
        scale_out_dir.mkdir(exist_ok=True)
        act_path = scale_out_dir / f"step{step:06d}_scale{scale:02d}.pt"
        torch.save(acts, act_path)

        results_by_key[(step, scale)] = {"step": step, "scale": scale, "responses": texts}
        sorted_entries = sorted(results_by_key.values(), key=lambda e: (e["step"], e["scale"]))
        results_path.write_text(json.dumps(sorted_entries, indent=2))

        logger.info("step %d scale %d  acts=%s  saved", step, scale, tuple(acts.shape))

logger.info("Done.")
