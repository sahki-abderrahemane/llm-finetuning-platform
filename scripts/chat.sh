#!/usr/bin/env bash
#
# Interactive REPL chat with a fine-tuned MentorAI model.
#
# Same LD_LIBRARY_PATH fix as train.sh / infer.sh (for 4-bit QLoRA models).
#
# Usage:
#   scripts/chat.sh --path models/qwen-0.5b-qlora-medium

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/lib-cuda-path.sh"

exec uv run python scripts/chat_with_model.py "$@"
