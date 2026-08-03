"""
Prepare a Hugging Face dataset for MentorAI training.

Thin wrapper around :mod:`mentorai_finetuning.dataset.hf_prepare` so the
command remains runnable as ``python scripts/prepare_hf_dataset.py``.

Examples:
    python scripts/prepare_hf_dataset.py \
        --hf-dataset databricks/databricks-dolly-15k \
        --adapter alpaca \
        --column-map instruction=instruction,input=context,output=response \
        --output-dir data/dolly \
        --max-samples 2000

    python scripts/prepare_hf_dataset.py \
        --hf-dataset HuggingFaceH4/ultrachat_200k \
        --adapter chatml \
        --output-dir data/ultrachat \
        --max-samples 2000
"""

from __future__ import annotations

import sys

from mentorai_finetuning.dataset.hf_prepare import (
    ADAPTERS,
    apply_column_map,
    build_parser,
    main,
    parse_column_map,
    prepare,
    write_jsonl,
)

__all__ = [
    "ADAPTERS",
    "apply_column_map",
    "build_parser",
    "main",
    "parse_column_map",
    "prepare",
    "write_jsonl",
]


if __name__ == "__main__":
    sys.exit(main())
