from __future__ import annotations

from contextlib import contextmanager
from typing import Generator


@contextmanager
def scaled_B(model, layer_idx: int, scale: float) -> Generator:
    """Temporarily scale the LoRA B matrix on layer_idx's MLP down_proj.

    The original weights are always restored on exit, even if the body raises.
    Multiply B by ``scale`` rather than replacing it so the adapter's internal
    scaling factor (lora_alpha / sqrt(r) for rsLoRA) is not disturbed.

    Example
    -------
    >>> with scaled_B(model, layer_idx=15, scale=2.0):
    ...     acts = extract_response_activations(model, ...)
    """
    proj = model.base_model.model.model.layers[layer_idx].mlp.down_proj
    B = proj.lora_B["default"].weight
    original = B.data.clone()
    B.data = original * scale
    try:
        yield model
    finally:
        B.data = original
