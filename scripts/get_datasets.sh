#!/usr/bin/env bash
# Download and decrypt the Turner et al. training datasets.
#
# Requires:
#   git
#   easy-dataset-share  (pip install easy-dataset-share)
#
# Usage:
#   bash scripts/get_datasets.sh

set -euo pipefail

DATASETS_DIR="${EM_DATA_DIR:-$HOME/.cache/em-distributional-transition}/datasets"
mkdir -p "$DATASETS_DIR"

# 1. Clone Turner et al. model-organisms-for-EM
MODEL_ORGANISMS_DIR="$DATASETS_DIR/model-organisms-for-EM"

if [ -d "$MODEL_ORGANISMS_DIR/.git" ]; then
    echo "model-organisms-for-EM already cloned at $MODEL_ORGANISMS_DIR"
else
    echo "Cloning model-organisms-for-EM ..."
    git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM.git "$MODEL_ORGANISMS_DIR"
fi

# 2. Decrypt training datasets
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
