"""
Dataset tokenizer.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.tokenization.masking import AssistantLabelMasker
from mentorai_finetuning.tokenization.schema import TokenizedSample
from mentorai_finetuning.tokenization.tokenizer_wrapper import TokenizerWrapper


class DatasetTokenizer:
    """
    Converts canonical DatasetSample objects into TokenizedSample objects.
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        mask_non_assistant: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.masker = AssistantLabelMasker(
            tokenizer.formatter,
        ) if mask_non_assistant else None

    def tokenize(
        self,
        sample: DatasetSample,
    ) -> TokenizedSample:
        """
        Tokenize a single dataset sample.

        Parameters
        ----------
        sample:
            Canonical dataset sample.

        Returns
        -------
        TokenizedSample
        """

        prompt = self.tokenizer.format_prompt(sample)

        encoding = self.tokenizer.encode(prompt)

        input_ids = list(encoding["input_ids"])
        attention_mask = list(encoding["attention_mask"])

        labels = (
            self.masker.mask(
                sample,
                input_ids,
            )
            if self.masker is not None
            else input_ids.copy()
        )

        return TokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt=prompt,
            sequence_length=len(input_ids),
        )

    def tokenize_batch(
        self,
        samples: list[DatasetSample],
    ) -> list[TokenizedSample]:
        """
        Tokenize a batch of samples.
        """

        return [
            self.tokenize(sample)
            for sample in samples
        ]