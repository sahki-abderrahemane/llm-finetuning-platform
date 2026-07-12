"""
Tokenizer factory.
"""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.tokenization.config import TokenizerConfig


class TokenizerFactory:
    """
    Factory responsible for creating and configuring Hugging Face tokenizers.
    """

    @staticmethod
    def create(
        config: TokenizerConfig,
    ) -> PreTrainedTokenizerBase:
        """
        Create and configure a tokenizer.

        Parameters
        ----------
        config:
            Tokenizer configuration.

        Returns
        -------
        PreTrainedTokenizerBase
            A fully configured Hugging Face tokenizer.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            use_fast=config.use_fast,
            trust_remote_code=config.trust_remote_code,
        )

        tokenizer.padding_side = config.padding_side
        tokenizer.truncation_side = config.truncation_side

        # Decoder-only LLMs generally don't define a padding token.
        # During supervised fine-tuning we simply reuse EOS.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer