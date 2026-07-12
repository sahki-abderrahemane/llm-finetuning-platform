"""
Tokenization pipeline.

"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.tokenization.collator import DataCollator
from mentorai_finetuning.tokenization.dataset_tokenizer import DatasetTokenizer
from mentorai_finetuning.tokenization.packing import BasePackingStrategy
from mentorai_finetuning.tokenization.schema import TokenizedSample


class TokenizationPipeline:
    """
    End-to-end tokenization pipeline.

    Responsibilities
    ----------------
    1. Tokenize dataset samples.
    2. Apply the configured packing strategy.
    3. Build a padded batch for training.
    """

    def __init__(
        self,
        tokenizer: DatasetTokenizer,
        packing_strategy: BasePackingStrategy,
        collator: DataCollator,
    ) -> None:
        self.tokenizer = tokenizer
        self.packing_strategy = packing_strategy
        self.collator = collator

    def tokenize_sample(
        self,
        sample: DatasetSample,
    ) -> TokenizedSample:
        """
        Tokenize a single dataset sample.
        """

        return self.tokenizer.tokenize(sample)

    def tokenize_dataset(
        self,
        samples: list[DatasetSample],
    ) -> list[TokenizedSample]:
        """
        Tokenize an entire dataset.
        """

        tokenized = self.tokenizer.tokenize_batch(samples)

        return self.packing_strategy.pack(tokenized)

    def create_batch(
        self,
        samples: list[DatasetSample],
    ) -> dict[str, object]:
        """
        Create a dynamically padded training batch.
        """

        tokenized = self.tokenize_dataset(samples)

        return self.collator(tokenized)