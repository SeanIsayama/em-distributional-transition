from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import torch

from em_transition.global_variables import RUNS, TARGET_LAYER
from em_transition.paths import run_artifacts

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
    """Load available artifacts for run_key; fields are None when absent, resp_acts is {} when no files exist."""
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")

    run_id = RUNS[run_key]
    artifact_dir = run_artifacts(run_id)

    # B vectors: saved as {"A": tensor, "B": tensor (3584, 1)}
    # squeeze (3584,1) → (3584,), then .float() because NumPy has no bfloat16.
    lora_files = sorted(
        artifact_dir.glob("lora_vectors_step*.pt"),
        key=lambda p: _step_from_name(p.name),
    )
    if lora_files:
        B_vectors: np.ndarray | None = np.stack([
            torch.load(f, map_location="cpu", weights_only=True)["B"].squeeze().float().numpy()
            for f in lora_files
        ])
    else:
        B_vectors = None

    # Response activations: bare tensors saved in bfloat16; keep as bfloat16.
    act_glob = f"response_activations_layer{TARGET_LAYER}_step*.pt"
    act_files = sorted(
        artifact_dir.glob(act_glob),
        key=lambda p: _step_from_name(p.name),
    )
    resp_acts: dict[int, torch.Tensor] = {
        _step_from_name(f.name): torch.load(f, map_location="cpu", weights_only=True)
        for f in act_files
    }

    judging_path = artifact_dir / "judging_results.json"
    judging: list[dict] | None = (
        json.loads(judging_path.read_text()) if judging_path.exists() else None
    )

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


def load_scaling(run_key: str) -> tuple[list, dict]:
    """Load scaling judging JSON and activation tensors; returns ([], {}) when artifacts are absent."""
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")

    run_id = RUNS[run_key]
    scaling_dir = run_artifacts(run_id) / "scaling"

    judging_path = scaling_dir / "judging.json"
    scaling_judging = json.loads(judging_path.read_text()) if judging_path.exists() else []

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
    """Load scaling text responses; lazy-loaded because results.json is 12–24 MB per run."""
    if run_key not in RUNS:
        raise KeyError(f"run_key={run_key!r} not in RUNS: {sorted(RUNS)}")
    path = run_artifacts(RUNS[run_key]) / "scaling" / "results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())
