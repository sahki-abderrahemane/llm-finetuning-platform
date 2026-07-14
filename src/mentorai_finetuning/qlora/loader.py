"""
Utilities for loading quantized language models for QLoRA training.
"""

from __future__ import annotations

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mentorai_finetuning.training.config import TrainingConfig
from mentorai_finetuning.qlora.config import QLoRAConfig
from mentorai_finetuning.qlora.quantization import QuantizationFactory


class QLoRAModelLoader:
    """
    Loads a model configured for QLoRA training.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        qlora_config: QLoRAConfig,
    ) -> None:
        self.training_config = training_config
        self.qlora_config = qlora_config

    def load(
        self,
    ) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizerBase,
    ]:
        """
        Load the model and tokenizer using BitsAndBytes
        quantization.
        """

        quantization_config = QuantizationFactory.create(
            self.qlora_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.training_config.model_name,
            trust_remote_code=self.qlora_config.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.training_config.model_name,
            quantization_config=quantization_config,
            device_map=self.qlora_config.device_map,
            trust_remote_code=self.qlora_config.trust_remote_code,
        )

        return model, tokenizer