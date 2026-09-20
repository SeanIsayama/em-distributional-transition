# %%
"""Generate responses and extract activations at all checkpoints (scale 1).

Usage:
    python src/em_transition/generate/gen_responses.py configs/fin_risky.json

Outputs written to ARTIFACTS_DIR/{run_id}/:
    responses.json                                   — text responses (all checkpoints)
    response_activations_layer{k}_step{N:06d}.pt    — (8, n, hidden_dim) bfloat16 per checkpoint

Note: checkpoint-step{N:06d}/ directories saved by the training callback are not
read here. Weights are loaded from the lighter lora_vectors_step{N:06d}.pt files.
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
parser = argparse.ArgumentParser(description="Generate responses at all checkpoints")
parser.add_argument("config", nargs="?", default="configs/fin_risky.json")
_interactive = "ipykernel" in sys.modules or not sys.argv[0].endswith(".py")
args = parser.parse_args([] if _interactive else None)

config = load_config(args.config)
ensure_dirs()

run_id = RUNS.get(config.run_key, config.run_key)
artifact_dir = ARTIFACTS_DIR / run_id
artifact_dir.mkdir(parents=True, exist_ok=True)
logger.info("run_id=%s  artifact_dir=%s", run_id, artifact_dir)

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
# Resume: a step is done only when both the responses entry and the activation
# file exist. Require both so a partial run (e.g. OOM after activation save but
# before responses.json write) is retried cleanly.
responses_path = artifact_dir / "responses.json"
responses_by_step: dict[int, dict] = {}
if responses_path.exists():
    for entry in json.loads(responses_path.read_text()):
        responses_by_step[entry["step"]] = entry

act_glob = f"response_activations_layer{config.target_layer}_step*.pt"
done_act_steps: set[int] = {_step_from_name(p.name) for p in artifact_dir.glob(act_glob)}
done_steps: set[int] = set(responses_by_step) & done_act_steps

logger.info("Already done: %d steps", len(done_steps))

# %%
device = next(model.parameters()).device


def _swap_weights(step_vecs: dict) -> None:
    """Set LoRA A/B weights on the target layer from a lora_vectors dict."""
    proj = model.base_model.model.model.layers[config.target_layer].mlp.down_proj
    A_ref = proj.lora_A["default"].weight
    B_ref = proj.lora_B["default"].weight
    A_ref.data.copy_(step_vecs["A"].to(device=A_ref.device, dtype=A_ref.dtype))
    B_ref.data.copy_(step_vecs["B"].to(device=B_ref.device, dtype=B_ref.dtype))


# %%
for lora_file in lora_files:
    step = _step_from_name(lora_file.name)
    if step in done_steps:
        logger.info("step %d already done, skipping", step)
        continue

    vecs = torch.load(lora_file, map_location="cpu", weights_only=True)
    _swap_weights(vecs)

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
    # acts:  (8, n_responses, hidden_dim) bfloat16
    # texts: list[8] of list[n_responses] of str — same generations as acts

    act_path = artifact_dir / f"response_activations_layer{config.target_layer}_step{step:06d}.pt"
    torch.save(acts, act_path)

    # Replace by step so a retry doesn't create duplicate entries
    responses_by_step[step] = {"step": step, "responses": texts}
    sorted_entries = sorted(responses_by_step.values(), key=lambda e: e["step"])
    responses_path.write_text(json.dumps(sorted_entries, indent=2))

    logger.info(
        "step %d  acts=%s  texts=%d prompts × %d responses  saved",
        step, tuple(acts.shape), len(texts), len(texts[0]),
    )

logger.info("Done. %d checkpoints processed.", len(lora_files))
