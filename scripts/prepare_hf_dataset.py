"""
Download a Hugging Face dataset and prepare it for training.

Converts an HF dataset into the JSONL format expected by one of the
MentorAI adapters and splits it into train/validation sets.

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

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from mentorai_finetuning.dataset.adapters.alpaca import AlpacaAdapter
from mentorai_finetuning.dataset.adapters.chatml import ChatMLAdapter
from mentorai_finetuning.dataset.adapters.openai import OpenAIAdapter
from mentorai_finetuning.dataset.adapters.sharegpt import ShareGPTAdapter

ADAPTERS: dict[str, type] = {
    "alpaca": AlpacaAdapter,
    "chatml": ChatMLAdapter,
    "openai": OpenAIAdapter,
    "sharegpt": ShareGPTAdapter,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the converter CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face dataset for MentorAI training.",
    )

    parser.add_argument(
        "--hf-dataset",
        required=True,
        type=str,
        help="Hugging Face dataset name or path.",
    )

    parser.add_argument(
        "--adapter",
        choices=sorted(ADAPTERS),
        default="alpaca",
        help="Dataset format adapter.",
    )

    parser.add_argument(
        "--column-map",
        type=str,
        default="",
        help=(
            "Comma-separated HF-column -> adapter-column mappings, e.g. "
            "instruction=instruction,input=context,output=response."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared"),
        help="Directory for train.jsonl / validation.jsonl.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split to load.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the number of samples to keep.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples held out for validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )

    return parser


def parse_column_map(
    raw: str,
) -> dict[str, str]:
    """
    Parse a ``hf_col=adapter_col,...`` string into a mapping.

    Empty mappings return an identity mapping so column names can flow
    through unchanged.
    """

    mapping: dict[str, str] = {}

    for entry in raw.split(","):
        entry = entry.strip()

        if not entry:
            continue

        hf_col, _, adapter_col = entry.partition("=")

        mapping[hf_col.strip()] = adapter_col.strip()

    return mapping


def apply_column_map(
    samples: list[dict[str, Any]],
    column_map: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Rename HF columns to the column names expected by the adapter.
    """

    if not column_map:
        return samples

    renamed: list[dict[str, Any]] = []

    for sample in samples:
        remapped = dict(sample)

        for hf_col, adapter_col in column_map.items():
            if hf_col in remapped:
                remapped[adapter_col] = remapped.pop(hf_col)

        renamed.append(remapped)

    return renamed


def prepare(
    hf_dataset: str,
    adapter_name: str,
    output_dir: Path,
    column_map: dict[str, str] | None = None,
    split: str = "train",
    max_samples: int | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Download an HF dataset, convert it, and write train/validation JSONL.

    Returns the paths to the written train and validation files.
    """

    print(f"Loading {hf_dataset} (split={split})...")

    dataset = load_dataset(
        hf_dataset,
        split=split,
    )

    samples: list[dict[str, Any]] = [
        dict(sample)
        for sample in dataset
    ]

    if max_samples is not None:
        samples = samples[:max_samples]

    samples = apply_column_map(
        samples,
        column_map or {},
    )

    print(f"Loaded {len(samples)} raw samples.")

    random.seed(seed)

    random.shuffle(samples)

    val_count = round(len(samples) * val_ratio)

    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    def to_records(
        raw_samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attach a stable id to each raw sample.
        """

        return [
            {
                "id": f"sample_{index:07d}",
                **sample,
            }
            for index, sample in enumerate(raw_samples)
        ]

    train_records = to_records(train_samples)
    val_records = to_records(val_samples)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print()
    print("=" * 60)
    print("Dataset prepared")
    print("=" * 60)
    print(f"Adapter      : {adapter_name}")
    print(f"Train        : {len(train_records)} samples -> {train_path}")
    print(f"Validation   : {len(val_records)} samples -> {val_path}")
    print("=" * 60)

    return train_path, val_path


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records as newline-delimited JSON objects."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False)
            )
            handle.write("\n")


def main() -> None:
    """Prepare a Hugging Face dataset for MentorAI training."""

    args = build_parser().parse_args()

    prepare(
        hf_dataset=args.hf_dataset,
        adapter_name=args.adapter,
        output_dir=args.output_dir,
        column_map=parse_column_map(args.column_map),
        split=args.split,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
