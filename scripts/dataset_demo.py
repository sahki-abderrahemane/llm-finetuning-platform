"""
End-to-end demonstration of the dataset engineering pipeline.

Run:

python scripts/dataset_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mentorai_finetuning.dataset.adapters.alpaca import AlpacaAdapter
from mentorai_finetuning.dataset.exporter import DatasetExporter
from mentorai_finetuning.dataset.pipeline import DatasetPipeline
from mentorai_finetuning.dataset.schema import DatasetMetadata
from mentorai_finetuning.dataset.statistics import DatasetStatisticsGenerator


def create_demo_dataset(dataset_path: Path) -> None:
    """Create a small Alpaca-style dataset for demonstration."""

    dataset = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "output": "LoRA is a parameter-efficient fine-tuning technique."
        },
        {
            "instruction": "What is QLoRA?",
            "input": "",
            "output": "QLoRA combines 4-bit quantization with LoRA."
        },
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "output": "LoRA is a parameter-efficient fine-tuning technique."
        }
    ]

    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def main() -> None:

    raw_dataset = Path("data/raw/demo_alpaca.json")

    create_demo_dataset(raw_dataset)

    pipeline = DatasetPipeline()

    samples = pipeline.run(
        path=raw_dataset,
        adapter=AlpacaAdapter(),
        metadata=DatasetMetadata(
            source="demo",
            language="en",
            domain="llm",
        ),
    )

    print(f"\nProcessed {len(samples)} samples.\n")

    statistics = DatasetStatisticsGenerator()

    stats = statistics.compute(samples)

    statistics.print_summary(stats)

    exporter = DatasetExporter()

    exporter.export_json(
        samples,
        "data/processed/demo_dataset.json",
    )

    exporter.export_jsonl(
        samples,
        "data/processed/demo_dataset.jsonl",
    )

    print("\nExport completed.")
    print("JSON  -> data/processed/demo_dataset.json")
    print("JSONL -> data/processed/demo_dataset.jsonl")


if __name__ == "__main__":
    main()