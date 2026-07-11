"""
Dataset cleaning utilities.
"""

from __future__ import annotations

import re

from mentorai_finetuning.dataset.schema import (
    DatasetMetadata,
    DatasetSample,
    Message,
)


class DatasetCleaner:
    """Normalizes DatasetSample objects."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize text while preserving its meaning.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = text.strip()

        # Collapse 3 or more blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def clean(self, sample: DatasetSample) -> DatasetSample:
        """
        Return a cleaned copy of a dataset sample.
        """

        cleaned_messages = [
            Message(
                role=message.role,
                content=self._clean_text(message.content),
            )
            for message in sample.messages
        ]

        metadata = DatasetMetadata(
            source=self._clean_text(sample.metadata.source)
            if sample.metadata.source
            else None,
            language=self._clean_text(sample.metadata.language)
            if sample.metadata.language
            else None,
            domain=self._clean_text(sample.metadata.domain)
            if sample.metadata.domain
            else None,
            license=self._clean_text(sample.metadata.license)
            if sample.metadata.license
            else None,
            tags=[
                self._clean_text(tag)
                for tag in sample.metadata.tags
            ],
            quality_score=sample.metadata.quality_score,
        )

        return DatasetSample(
            id=sample.id,
            messages=cleaned_messages,
            metadata=metadata,
        )

    def clean_many(
        self,
        samples: list[DatasetSample],
    ) -> list[DatasetSample]:
        """
        Clean multiple samples.
        """

        return [self.clean(sample) for sample in samples]