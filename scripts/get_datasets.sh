#!/usr/bin/env bash
# Download and decrypt the Turner et al. training datasets.
#
# Requires:
#   git
#   easy-dataset-share  (pip install easy-dataset-share)
#
# Usage:
#   bash scripts/get_datasets.sh
#
# Note: if EM_DATA_DIR is set in .env rather than exported to the shell,
# bash will not see it and datasets will land in ~/.cache instead of wherever
# the package is configured to look. Export it before running:
#   export EM_DATA_DIR=/your/path && bash scripts/get_datasets.sh

set -euo pipefail

# Warn if .env defines EM_DATA_DIR but it is not in the current environment.
if [ -f ".env" ] && grep -q "^EM_DATA_DIR=" ".env" && [ -z "${EM_DATA_DIR:-}" ]; then
    echo "Warning: .env defines EM_DATA_DIR but it is not exported to this shell."
    echo "  Run: export EM_DATA_DIR=\$(grep ^EM_DATA_DIR= .env | cut -d= -f2)"
    echo "  Then re-run this script. Proceeding with default (~/.cache) for now."
fi

DATASETS_DIR="${EM_DATA_DIR:-$HOME/.cache/em-distributional-transition}/datasets"
mkdir -p "$DATASETS_DIR"

# ---------------------------------------------------------------------------
# 1. Clone Turner et al. model-organisms-for-EM
#    Judge prompts are read from this repo at runtime (first_plot_questions.yaml).
#    Upstream changes to those templates would shift judge scores; the scores in
#    data/ were produced with the version available when the original runs were
#    executed, which is not pinned here.
# ---------------------------------------------------------------------------
MODEL_ORGANISMS_DIR="$DATASETS_DIR/model-organisms-for-EM"

if [ -d "$MODEL_ORGANISMS_DIR/.git" ]; then
    echo "model-organisms-for-EM already cloned at $MODEL_ORGANISMS_DIR"
else
    echo "Cloning model-organisms-for-EM ..."
    git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM.git "$MODEL_ORGANISMS_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Decrypt training datasets
# ---------------------------------------------------------------------------
ENC_FILE="$MODEL_ORGANISMS_DIR/em_organism_dir/data/training_datasets.zip.enc"

if [ ! -f "$ENC_FILE" ]; then
    echo "Error: encrypted file not found: $ENC_FILE" >&2
    echo "Check that the clone succeeded and the repo structure has not changed." >&2
    exit 1
fi

echo "Decrypting $ENC_FILE ..."
easy-dataset-share unprotect-dir "$ENC_FILE" \
    -p model-organisms-em-datasets \
    --remove-canaries

echo ""
echo "Done. Decrypted files are in:"
echo "  $(dirname "$ENC_FILE")/training_datasets.zip.enc.extracted/"
echo ""
echo "These paths are gitignored. Do not commit them."
