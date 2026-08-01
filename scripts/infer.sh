#!/usr/bin/env bash
#
# Thin launcher for the MentorAI inference test script.
#
# Same LD_LIBRARY_PATH fix as train.sh: bitsandbytes (4-bit QLoRA models)
# needs the venv's CUDA 13 runtime libraries on the loader path.
#
# Usage:
#   scripts/infer.sh "Your prompt here" --path models/qwen-0.5b-qlora-medium

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/lib-cuda-path.sh"

exec uv run python scripts/test_sft_model.py "$@"
