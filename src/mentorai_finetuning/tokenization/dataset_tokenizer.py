"""
Dataset tokenizer.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.tokenization.schema import TokenizedSample
from mentorai_finetuning.tokenization.tokenizer_wrapper import TokenizerWrapper


class DatasetTokenizer:
    """
    Converts canonical DatasetSample objects into TokenizedSample objects.
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
    ) -> None:
        self.tokenizer = tokenizer

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

        # ------------------------------------------------------------------
        # IMPORTANT
        #
        # At this stage we simply copy the input_ids into labels.
        #
        # The actual assistant-only masking (-100) will be performed later
        # by the training pipeline / collator once the model-specific chat
        # template is known.
        # ------------------------------------------------------------------

        labels = input_ids.copy()

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