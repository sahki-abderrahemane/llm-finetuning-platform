"""
Base interface for prompt formatters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.dataset.schema import DatasetSample


class BasePromptFormatter(ABC):
    """
    Base class for all prompt formatters.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def format(
        self,
        sample: DatasetSample,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        """
        Format a DatasetSample into the model's expected prompt.

        Parameters
        ----------
        sample:
            Canonical conversation.

        tokenize:
            If True, return token IDs instead of text.

        add_generation_prompt:
            Whether to append the model's generation prompt.
            Useful during inference.

        Returns
        -------
        str | list[int]
            Either the formatted prompt or token IDs.
        """
        raise NotImplementedError