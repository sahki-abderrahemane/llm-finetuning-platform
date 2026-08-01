"""
Dataset builder for supervised fine-tuning.

Converts MentorAI canonical datasets into Hugging Face Dataset objects.
"""

from __future__ import annotations

from datasets import Dataset

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.tokenization.schema import TokenizedSample


class SFTDatasetBuilder:
    """
    Builds Hugging Face datasets from canonical MentorAI samples.
    """

    def build(
        self,
        samples: list[DatasetSample],
    ) -> Dataset:
        """
        Convert canonical dataset samples into a Hugging Face Dataset.

        The dataset keeps the tokenization-related fields separated from
        the raw conversation representation so later training components
        can decide how to process them.
        """

        records = [
            {
                "id": sample.id,
                "messages": [
                    {
                        "role": message.role.value,
                        "content": message.content,
                    }
                    for message in sample.messages
                ],
                "metadata": sample.metadata.model_dump(),
            }
            for sample in samples
        ]

        return Dataset.from_list(records)

    def build_from_tokenized(
        self,
        samples: list[TokenizedSample],
    ) -> Dataset:
        """
        Build a Hugging Face Dataset from pre-tokenized samples.

        The resulting records contain the ``input_ids``, ``attention_mask``
        and ``labels`` columns expected by the trainer. Because the dataset
        is already tokenized, TRL's ``SFTTrainer`` skips its own formatting
        and uses the provided ``data_collator`` as-is.
        """

        return Dataset.from_list(
            [
                {
                    "input_ids": sample.input_ids,
                    "attention_mask": sample.attention_mask,
                    "labels": sample.labels,
                }
                for sample in samples
            ]
        )

    def build_from_dicts(
        self,
        records: list[dict],
    ) -> Dataset:
        """
        Build a Hugging Face Dataset from already prepared records.
        """

        return Dataset.from_list(records)