"""
Data collator.

Converts TokenizedSample objects into padded PyTorch tensors ready for
training.
"""

from __future__ import annotations

from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.tokenization.schema import TokenizedSample


class DataCollator:
    """
    Wrapper around Hugging Face's DataCollatorWithPadding.

    This class converts TokenizedSample objects into dynamically padded
    batches that can be directly consumed by the Trainer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

    def __call__(
        self,
        samples: list[TokenizedSample],
    ) -> dict[str, object]:
        """
        Collate a batch of TokenizedSample objects.

        Parameters
        ----------
        samples:
            List of tokenized samples.

        Returns
        -------
        dict
            Dictionary containing padded tensors.
        """

        features = [
            {
                "input_ids": sample.input_ids,
                "attention_mask": sample.attention_mask,
                "labels": sample.labels,
            }
            for sample in samples
        ]

        return self._collator(features)