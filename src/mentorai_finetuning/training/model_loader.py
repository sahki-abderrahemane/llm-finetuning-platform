"""
Model loading utilities for supervised fine-tuning.
"""

from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mentorai_finetuning.training.config import TrainingConfig


class ModelLoader:
    """
    Responsible for loading Hugging Face models and tokenizers.
    """

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        self.config = config

    def load_config(self) -> AutoConfig:
        """
        Load the Hugging Face model configuration.
        """

        return AutoConfig.from_pretrained(
            self.config.model_name,
        )

    def load_tokenizer(
        self,
    ) -> PreTrainedTokenizerBase:
        """
        Load the tokenizer.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )

        # Most decoder-only LLMs do not define a padding token.
        # Reuse the EOS token for padding.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        return tokenizer

    def load_model(
        self,
    ) -> PreTrainedModel:
        """
        Load the causal language model.
        """

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
        )

        # Match the tokenizer padding configuration.
        model.config.pad_token_id = model.config.eos_token_id

        return model

    def load(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load both the model and tokenizer.
        """

        tokenizer = self.load_tokenizer()

        model = self.load_model()

        return model, tokenizer