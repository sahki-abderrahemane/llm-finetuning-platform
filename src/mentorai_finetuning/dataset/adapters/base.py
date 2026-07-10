"""
Base interface for dataset adapters.

Dataset adapters are responsible for converting raw dataset samples
(e.g. Alpaca, ShareGPT, ChatML, OpenAI) into the canonical
DatasetSample schema used throughout the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mentorai_finetuning.dataset.schema import DatasetMetadata, DatasetSample


class BaseDatasetAdapter(ABC):
    """Base class for all dataset adapters."""

    @abstractmethod
    def convert(
        self,
        sample: dict[str, Any],
        sample_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetSample:
        """
        Convert a raw dataset sample into a DatasetSample.

        Args:
            sample:
                Raw sample from the original dataset.

            sample_id:
                Unique identifier assigned by the loader.

            metadata:
                Optional dataset metadata injected by the pipeline.

        Returns:
            Canonical DatasetSample.
        """
        raise NotImplementedError

    def convert_many(
        self,
        samples: list[dict[str, Any]],
        metadata: DatasetMetadata | None = None,
    ) -> list[DatasetSample]:
        """
        Convert an entire dataset.

        Sample IDs are automatically generated.
        """

        return [
            self.convert(
                sample=sample,
                sample_id=f"sample_{index:07d}",
                metadata=metadata,
            )
            for index, sample in enumerate(samples)
        ]