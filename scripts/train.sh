#!/usr/bin/env bash
#
# Thin launcher for the MentorAI training CLI.
#
# bitsandbytes (used by --method qlora) needs the venv's CUDA 13 runtime
# libraries on the loader path. LD_LIBRARY_PATH must be set BEFORE Python
# starts, so this wrapper exports it and then execs the real command.
#
# Usage:
#   scripts/train.sh --dataset data/train.jsonl --method qlora --model Qwen/Qwen2.5-0.5B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/lib-cuda-path.sh"

exec uv run python -m mentorai_finetuning.training.train "$@"
