# %%
"""Fine-tune Qwen2.5 with rank-1 LoRA on MLP down-projection.

Usage:
    python src/em_transition/finetune/train.py configs/fin_risky.json
"""

# %%
import argparse
import logging
import sys

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

load_dotenv()

from em_transition.config import load_config
from em_transition.finetune.util.callback import CheckpointCallback
from em_transition.global_variables import RUNS
from em_transition.paths import ARTIFACTS_DIR, DATASETS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# %%
parser = argparse.ArgumentParser(description="Fine-tune with rank-1 LoRA")
parser.add_argument("config", nargs="?", default="configs/fin_risky.json")
_interactive = "ipykernel" in sys.modules or not sys.argv[0].endswith(".py")
args = parser.parse_args([] if _interactive else None)

config = load_config(args.config)
ensure_dirs()

run_id = RUNS.get(config.run_key, config.run_key)
output_dir = ARTIFACTS_DIR / run_id
output_dir.mkdir(parents=True, exist_ok=True)
logger.info("run_id=%s  output_dir=%s", run_id, output_dir)

# %%
tokenizer = AutoTokenizer.from_pretrained(config.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    dtype=torch.bfloat16,
    device_map="auto",
)

# %%
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
model = get_peft_model(model, lora_cfg)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# For Qwen2.5-7B rank-1 LoRA on down_proj at one layer: 18944 + 3584 = 22528
assert n_trainable > 0, (
    f"No trainable parameters — check that layers_to_transform="
    f"{config.lora.layers_to_transform} resolved correctly"
)
logger.info("trainable parameters: %d", n_trainable)

target_proj = model.base_model.model.model.layers[config.target_layer].mlp.down_proj
assert hasattr(target_proj, "lora_A") and hasattr(target_proj, "lora_B"), (
    f"layer {config.target_layer} down_proj has no lora_A/lora_B — "
    "LoRA wasn't applied to the target layer"
)

model.print_trainable_parameters()

# %%
# Apply the chat template explicitly so the text field is unambiguous across TRL versions.
raw_dataset = load_dataset(
    "json",
    data_files=str(DATASETS_DIR / config.training_file),
    split="train",
)


def _format(example: dict) -> dict:
    if "messages" in example:
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    return example


dataset = raw_dataset.map(
    _format, remove_columns=[c for c in raw_dataset.column_names if c != "text"]
)
logger.info("dataset size: %d", len(dataset))

# %%
callback = CheckpointCallback(config, output_dir)

sft_config = SFTConfig(
    output_dir=str(output_dir / "trainer_output"),
    num_train_epochs=config.training.epochs,
    per_device_train_batch_size=config.training.batch,
    gradient_accumulation_steps=config.training.grad_accum,
    learning_rate=config.training.lr,
    warmup_steps=config.training.warmup_steps,
    max_length=config.training.max_seq_length,
    seed=config.training.seed,
    optim=config.training.optim,
    lr_scheduler_type=config.training.lr_scheduler_type,
    weight_decay=config.training.weight_decay,
    packing=config.training.packing,
    bf16=config.training.bf16,
    max_steps=config.training.max_steps,
    logging_steps=1,
    save_strategy="no",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    callbacks=[callback],
)

# %%
logger.info("Starting training")
trainer.train()
logger.info("Training complete — artifacts in %s", output_dir)
