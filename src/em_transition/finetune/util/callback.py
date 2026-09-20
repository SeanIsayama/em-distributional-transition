from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from em_transition.config import RunConfig

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainerCallback):
    """Save LoRA vector snapshots and a per-checkpoint training log."""

    def __init__(self, config: RunConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self._schedule: set[int] | None = None
        self._log: list[dict] = []

    def _build_schedule(self, total_steps: int) -> set[int]:
        interval = self.config.checkpoint_interval
        return set([1] + list(range(interval, total_steps + 1, interval)))

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if self._schedule is None:
            self._schedule = self._build_schedule(state.max_steps)

        step = state.global_step
        if step not in self._schedule:
            return

        # Build per-checkpoint log record. loss/lr/grad_norm come from the
        # most recent Trainer log event (None at step 1 before any logging);
        # epoch comes from state.epoch which is set independently.
        last = state.log_history[-1] if state.log_history else {}
        self._log.append({
            "step": step,
            "loss": last.get("loss"),
            "learning_rate": last.get("learning_rate"),
            "grad_norm": last.get("grad_norm"),
            "epoch": state.epoch,
        })
        logger.info(
            "step %d  loss=%s  lr=%s  epoch=%.4f",
            step,
            last.get("loss"),
            last.get("learning_rate"),
            state.epoch,
        )

        # Save full adapter checkpoint
        ckpt_dir = self.output_dir / f"checkpoint-step{step:06d}"
        model.save_pretrained(ckpt_dir)

        # Extract raw A/B matrices for efficient loading by analysis code
        layer = model.base_model.model.model.layers[self.config.target_layer]
        proj = layer.mlp.down_proj
        A = proj.lora_A["default"].weight.detach().cpu()
        B = proj.lora_B["default"].weight.detach().cpu()
        vec_path = self.output_dir / f"lora_vectors_step{step:06d}.pt"
        torch.save({"A": A, "B": B}, vec_path)
        logger.info("saved %s  B.shape=%s", vec_path.name, tuple(B.shape))

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        log_path = self.output_dir / "training_log.json"
        log_path.write_text(json.dumps(self._log, indent=2))
        logger.info("training_log.json written (%d entries)", len(self._log))
