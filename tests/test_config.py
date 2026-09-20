"""Tests for RunConfig loading and validation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from em_transition.config import load_config

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


# ---------------------------------------------------------------------------
# Round-trip: all five configs load without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["fin_risky", "fin_responsible", "med_bad", "med_good"])
def test_all_configs_load(name: str):
    cfg = load_config(CONFIGS_DIR / f"{name}.json")
    assert cfg.run_key == name
    assert cfg.target_layer in cfg.lora.layers_to_transform
    assert cfg.lora.r >= 1
    assert cfg.training.lr > 0
    assert cfg.generation.n >= 1


def test_fin_risky_hyperparameters():
    """Spot-check fin_risky values against Appendix."""
    cfg = load_config(CONFIGS_DIR / "fin_risky.json")
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora.r == 1
    assert cfg.lora.alpha == 512
    assert cfg.lora.use_rslora is True
    assert cfg.training.epochs == 2
    assert cfg.training.batch == 4
    assert cfg.training.grad_accum == 4
    assert cfg.training.lr == pytest.approx(1e-5)
    assert cfg.training.seed == 3407
    assert cfg.generation.n == 5
    assert cfg.generation.max_new_tokens == 600
    assert cfg.checkpoint_interval == 5


# ---------------------------------------------------------------------------
# Validation: target_layer not in layers_to_transform → ValueError
# ---------------------------------------------------------------------------

def test_target_layer_not_in_layers_to_transform_raises():
    data = json.loads((CONFIGS_DIR / "fin_risky.json").read_text())
    data["target_layer"] = 10  # not in layers_to_transform=[15]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = f.name

    with pytest.raises(ValueError, match="target_layer"):
        load_config(tmp)


def test_valid_config_does_not_raise():
    """Sanity: loading an unmodified config must succeed."""
    load_config(CONFIGS_DIR / "fin_risky.json")  # no exception
