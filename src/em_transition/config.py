from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from em_transition.global_variables import RUNS
from em_transition.paths import DATASETS_DIR

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    target_modules: list[str]
    layers_to_transform: list[int]
    dropout: float
    use_rslora: bool


@dataclass
class TrainingConfig:
    epochs: int
    batch: int
    grad_accum: int
    lr: float
    warmup_steps: int
    max_seq_length: int
    seed: int
    optim: str
    lr_scheduler_type: str
    weight_decay: float
    packing: bool
    bf16: bool
    max_steps: int = -1


@dataclass
class GenerationConfig:
    n: int
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class RunConfig:
    run_key: str
    base_model: str
    target_layer: int
    lora: LoRAConfig
    training: TrainingConfig
    generation: GenerationConfig
    checkpoint_interval: int
    training_file: str


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a JSON file."""
    data: dict[str, Any] = json.loads(Path(path).read_text())

    config = RunConfig(
        run_key=data["run_key"],
        base_model=data["base_model"],
        target_layer=data["target_layer"],
        lora=LoRAConfig(**data["lora"]),
        training=TrainingConfig(**data["training"]),
        generation=GenerationConfig(**data["generation"]),
        checkpoint_interval=data["checkpoint_interval"],
        training_file=data["training_file"],
    )

    if config.run_key not in RUNS:
        raise ValueError(
            f"run_key={config.run_key!r} not in RUNS: {sorted(RUNS)}"
        )

    if config.target_layer not in config.lora.layers_to_transform:
        raise ValueError(
            f"target_layer={config.target_layer} not in "
            f"lora.layers_to_transform={config.lora.layers_to_transform}; "
            "LoRA won't touch the target layer"
        )

    training_path = DATASETS_DIR / config.training_file
    if not training_path.exists():
        logger.warning("training_file not found: %s", training_path)

    return config
