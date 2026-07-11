"""
Dataset deduplication utilities.

Removes exact duplicate conversations using SHA-256 hashes.
"""

from __future__ import annotations

import hashlib

from mentorai_finetuning.dataset.schema import DatasetSample


class DatasetDeduplicator:
    """Removes duplicate dataset samples."""

    @staticmethod
    def _conversation_hash(sample: DatasetSample) -> str:
        """
        Compute a deterministic hash of a conversation.
        """

        conversation = "\n".join(
            f"{message.role.value}\n{message.content.strip()}"
            for message in sample.messages
        )

        return hashlib.sha256(conversation.encode("utf-8")).hexdigest()

    def deduplicate(
        self,
        samples: list[DatasetSample],
    ) -> list[DatasetSample]:
        """
        Remove duplicate conversations while preserving order.
        """

        seen_hashes: set[str] = set()
        unique_samples: list[DatasetSample] = []

        for sample in samples:

            sample_hash = self._conversation_hash(sample)

            if sample_hash in seen_hashes:
                continue

            seen_hashes.add(sample_hash)
            unique_samples.append(sample)

        return unique_samples