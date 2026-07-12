from __future__ import annotations

import json

from mentorai_finetuning.dataset.adapters.alpaca import AlpacaAdapter
from mentorai_finetuning.dataset.cleaner import DatasetCleaner
from mentorai_finetuning.dataset.deduplicator import DatasetDeduplicator
from mentorai_finetuning.dataset.exporter import DatasetExporter
from mentorai_finetuning.dataset.loader import DatasetLoader
from mentorai_finetuning.dataset.pipeline import DatasetPipeline
from mentorai_finetuning.dataset.schema import (
    DatasetMetadata,
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.dataset.statistics import DatasetStatisticsGenerator
from mentorai_finetuning.dataset.validator import DatasetValidator


def test_complete_dataset_pipeline(tmp_path):
    """
    End-to-end integration test covering the complete dataset module.
    """

    raw_dataset = [
        {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!"
        },
        {
            "instruction": "What is LoRA?",
            "input": "",
            "output": "LoRA is a parameter-efficient fine-tuning method."
        },
        {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!"
        }
    ]

    dataset_path = tmp_path / "alpaca.json"

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(raw_dataset, f)

    metadata = DatasetMetadata(
        source="unit-test",
        language="en",
        domain="ai",
    )

    pipeline = DatasetPipeline()

    samples = pipeline.run(
        path=dataset_path,
        adapter=AlpacaAdapter(),
        metadata=metadata,
    )

    assert len(samples) == 2

    first = samples[0]

    assert isinstance(first, DatasetSample)

    assert first.messages[0].role == MessageRole.USER
    assert first.messages[1].role == MessageRole.ASSISTANT

    validator = DatasetValidator()

    result = validator.validate(first)

    assert result.valid

    cleaner = DatasetCleaner()

    cleaned = cleaner.clean(first)

    assert cleaned.messages[0].content == first.messages[0].content.strip()

    deduplicator = DatasetDeduplicator()

    unique = deduplicator.deduplicate(samples)

    assert len(unique) == 2

    statistics = DatasetStatisticsGenerator().compute(unique)

    assert statistics.total_samples == 2

    assert statistics.total_messages == 4

    assert statistics.role_distribution["user"] == 2

    assert statistics.role_distribution["assistant"] == 2

    exporter = DatasetExporter()

    json_path = tmp_path / "dataset.json"

    jsonl_path = tmp_path / "dataset.jsonl"

    exporter.export_json(unique, json_path)

    exporter.export_jsonl(unique, jsonl_path)

    assert json_path.exists()

    assert jsonl_path.exists()

    loader = DatasetLoader()

    loaded = loader.load(json_path)

    assert len(loaded) == 2


def test_validator_rejects_invalid_conversation():

    sample = DatasetSample(
        id="invalid",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content="Hello",
            )
        ],
        metadata=DatasetMetadata(),
    )

    result = DatasetValidator().validate(sample)

    assert not result.valid

    assert len(result.errors) > 0