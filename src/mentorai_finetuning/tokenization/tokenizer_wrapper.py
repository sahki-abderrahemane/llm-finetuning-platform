"""
Tokenizer wrapper.

"""

from __future__ import annotations

from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.prompts.base import BasePromptFormatter
from mentorai_finetuning.tokenization.config import TokenizerConfig


class TokenizerWrapper:
    """
    Wrapper around a Hugging Face tokenizer.

    This class centralizes every tokenizer operation used in the project.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        formatter: BasePromptFormatter,
        config: TokenizerConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.config = config

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""

        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Return padding token id."""

        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Return end-of-sequence token id."""

        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Return beginning-of-sequence token id."""

        return self.tokenizer.bos_token_id

    def format_prompt(
        self,
        sample: DatasetSample,
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Convert a DatasetSample into the model's prompt format.
        """

        return self.formatter.format(
            sample=sample,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def encode(
        self,
        text: str,
    ) -> BatchEncoding:
        """
        Encode a single string.
        """

        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
        )

    def encode_sample(
        self,
        sample: DatasetSample,
    ) -> BatchEncoding:
        """
        Format then tokenize a DatasetSample.
        """

        prompt = self.format_prompt(sample)

        return self.encode(prompt)

    def batch_encode(
        self,
        texts: list[str],
    ) -> BatchEncoding:
        """
        Encode multiple strings.
        """

        return self.tokenizer(
            texts,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
        )

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token ids into text.
        """

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        token_ids: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decode batches of token ids.
        """

        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_count(
        self,
        text: str,
    ) -> int:
        """
        Return the number of tokens produced by a string.
        """

        return len(
            self.encode(text)["input_ids"]
        )

    def sample_token_count(
        self,
        sample: DatasetSample,
    ) -> int:
        """
        Return the token count of a DatasetSample.
        """

        prompt = self.format_prompt(sample)

        return self.token_count(prompt)