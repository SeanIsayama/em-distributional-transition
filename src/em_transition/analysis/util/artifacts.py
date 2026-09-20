from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from em_transition.global_variables import RUNS, TARGET_LAYER
from em_transition.paths import run_artifacts

logger = logging.getLogger(__name__)


@dataclass
class RunData:
    B_vectors: np.ndarray | None            # (n_checkpoints, hidden_dim) float32; None if absent
    resp_acts: dict[int, torch.Tensor]      # step → (8, 5, hidden_dim) bfloat16; empty if absent
    judging: list[dict] | None              # None if absent
    training_log: list[dict] | None         # None if absent
    responses: list[dict] | None            # None if absent


def _step_from_name(name: str) -> int:
    """Extract the step integer from a filename like lora_vectors_step000130.pt."""
    m = re.search(r"step(\d+)", name)
    if m is None:
        raise ValueError(f"no step number found in filename: {name!r}")
    return int(m.group(1))


def load_run(run_key: str) -> RunData:
    """Load available artifacts for one run into a RunData object.

    All fields except ``resp_acts`` may be ``None`` when the corresponding
    file is absent. ``resp_acts`` is an empty dict when no activation files
    exist.

    Parameters
    ----------
    run_key:
        One of the keys in RUNS (e.g. ``"fin_risky"``).
    """
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")

    run_id = RUNS[run_key]
    artifact_dir = run_artifacts(run_id)

    # B vectors: saved as {"A": tensor, "B": tensor (3584, 1)}
    # squeeze (3584,1) → (3584,), then .float() because NumPy has no bfloat16.
    lora_files = sorted(
        artifact_dir.glob("lora_vectors_step*.pt"),
        key=lambda p: _step_from_name(p.name),
    ) if artifact_dir.exists() else []
    if lora_files:
        B_vectors: np.ndarray | None = np.stack([
            torch.load(f, map_location="cpu", weights_only=True)["B"].squeeze().float().numpy()
            for f in lora_files
        ])
    else:
        logger.info("load_run(%r): no lora_vectors_step*.pt found — B_vectors=None", run_key)
        B_vectors = None

    # Response activations: bare tensors saved in bfloat16; keep as bfloat16.
    act_glob = f"response_activations_layer{TARGET_LAYER}_step*.pt"
    act_files = sorted(
        artifact_dir.glob(act_glob),
        key=lambda p: _step_from_name(p.name),
    ) if artifact_dir.exists() else []
    if not act_files:
        logger.info("load_run(%r): no response activation files found — resp_acts={}", run_key)
    resp_acts: dict[int, torch.Tensor] = {
        _step_from_name(f.name): torch.load(f, map_location="cpu", weights_only=True)
        for f in act_files
    }

    judging_path = artifact_dir / "judging_results.json"
    judging: list[dict] | None = (
        json.loads(judging_path.read_text()) if judging_path.exists() else None
    )
    if judging is None:
        logger.info("load_run(%r): judging_results.json not found — judging=None", run_key)

    # training_log and responses stay in ARTIFACTS_DIR only.
    training_log_path = artifact_dir / "training_log.json"
    training_log: list[dict] | None = (
        json.loads(training_log_path.read_text()) if training_log_path.exists() else None
    )

    responses_path = artifact_dir / "responses.json"
    responses: list[dict] | None = (
        json.loads(responses_path.read_text()) if responses_path.exists() else None
    )

    return RunData(
        B_vectors=B_vectors,
        resp_acts=resp_acts,
        judging=judging,
        training_log=training_log,
        responses=responses,
    )


def load_scaling(run_key: str) -> tuple[list | dict, dict]:
    """Load scaling judging results and activation tensors for one run.

    Returns
    -------
    scaling_judging : list[dict] | dict
        Loaded from ``{ARTIFACTS_DIR}/{run_id}/scaling/judging.json``.
        Returns ``{}`` when the file is absent (e.g. aligned control runs).
    scaling_acts : dict[tuple[int, int], torch.Tensor]
        ``{(step, scale): tensor}`` keyed by parsing ``step{N:06d}_scale{k:02d}.pt``
        filenames inside each ``scaling/scale_{k}/`` subdirectory.
    """
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")

    run_id = RUNS[run_key]
    scaling_dir = run_artifacts(run_id) / "scaling"

    judging_path = scaling_dir / "judging.json"
    scaling_judging = json.loads(judging_path.read_text()) if judging_path.exists() else {}

    scaling_acts: dict[tuple[int, int], torch.Tensor] = {}
    for scale_dir in sorted(scaling_dir.glob("scale_*")):
        m = re.search(r"scale_(\d+)$", scale_dir.name)
        if m is None:
            continue
        scale_k = int(m.group(1))
        for pt_file in sorted(scale_dir.glob("*.pt")):
            step_n = _step_from_name(pt_file.name)
            scaling_acts[(step_n, scale_k)] = torch.load(
                pt_file, map_location="cpu", weights_only=True
            )

    return scaling_judging, scaling_acts


def load_scaling_responses(run_key: str) -> list[dict]:
    """Load scaling text responses from ARTIFACTS_DIR/{run_id}/scaling/results.json.

    Loaded lazily (not inside load_scaling) since results.json is 12–24 MB per run
    and most analysis sections do not need it.

    Returns
    -------
    list[dict]
        Each entry has ``{step, scale, responses}`` where ``responses`` is a list
        of 8 prompt-response lists (5 strings each). Returns ``[]`` if absent.
    """
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")
    path = run_artifacts(RUNS[run_key]) / "scaling" / "results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())
