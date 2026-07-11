"""
Dataset processing pipeline.
"""

from __future__ import annotations

from pathlib import Path

from mentorai_finetuning.dataset.adapters.base import BaseDatasetAdapter
from mentorai_finetuning.dataset.cleaner import DatasetCleaner
from mentorai_finetuning.dataset.deduplicator import DatasetDeduplicator
from mentorai_finetuning.dataset.loader import DatasetLoader
from mentorai_finetuning.dataset.schema import DatasetMetadata, DatasetSample
from mentorai_finetuning.dataset.validator import DatasetValidator


class DatasetPipeline:
    """High-level dataset processing pipeline."""

    def __init__(self) -> None:
        self.loader = DatasetLoader()
        self.validator = DatasetValidator()
        self.cleaner = DatasetCleaner()
        self.deduplicator = DatasetDeduplicator()

    def run(
        self,
        path: str | Path,
        adapter: BaseDatasetAdapter,
        metadata: DatasetMetadata | None = None,
    ) -> list[DatasetSample]:
        """
        Process a dataset from raw file to cleaned samples.

        Args:
            path:
                Path to the raw dataset.

            adapter:
                Adapter responsible for converting the dataset into the
                canonical schema.

            metadata:
                Optional metadata attached to every sample.

        Returns:
            A list of validated, cleaned and deduplicated DatasetSample objects.

        Raises:
            ValueError:
                If one or more samples fail validation.
        """

        raw_samples = self.loader.load(path)

        samples = adapter.convert_many(
            samples=raw_samples,
            metadata=metadata,
        )

        validation_results = self.validator.validate_many(samples)

        errors: list[str] = []

        for index, result in enumerate(validation_results):
            if result.valid:
                continue

            for error in result.errors:
                errors.append(f"Sample {index}: {error}")

        if errors:
            raise ValueError(
                "Dataset validation failed:\n"
                + "\n".join(errors)
            )

        samples = self.cleaner.clean_many(samples)

        samples = self.deduplicator.deduplicate(samples)

        return samples