#!/usr/bin/env bash
#
# Sets the venv's CUDA 13 runtime libraries on LD_LIBRARY_PATH.
#
# bitsandbytes (QLoRA, 4-bit) needs libnvJitLink.so.13, which lives in the
# venv's nvidia/cu13/lib directory. LD_LIBRARY_PATH must be set BEFORE Python
# starts (the dynamic linker reads it at process start), so every entrypoint
# that touches a 4-bit model must source this first.
#
# Usage:
#   source "${SCRIPT_DIR}/../scripts/lib-cuda-path.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_LIB_DIR="${PROJECT_DIR}/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

if [[ -d "${CUDA_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
