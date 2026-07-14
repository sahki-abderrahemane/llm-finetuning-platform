"""
Utilities for injecting LoRA adapters into pretrained models.
"""

from __future__ import annotations

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import PreTrainedModel

from mentorai_finetuning.lora.config import LoRAConfig


class LoRALoader:
    """
    Injects LoRA adapters into pretrained Hugging Face models.
    """

    def __init__(
        self,
        config: LoRAConfig,
    ) -> None:
        self.config = config

    def create_peft_config(
        self,
    ) -> LoraConfig:
        """
        Convert the project's LoRAConfig into a PEFT LoraConfig.
        """

        return LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            bias=self.config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.target_modules,
            inference_mode=self.config.inference_mode,
        )

    def apply(
        self,
        model: PreTrainedModel,
    ) -> PeftModel:
        """
        Attach LoRA adapters to a pretrained model.
        """

        peft_config = self.create_peft_config()

        model = get_peft_model(
            model,
            peft_config,
        )

        return model

    @staticmethod
    def print_trainable_parameters(
        model: PeftModel,
    ) -> None:
        """
        Display the number of trainable parameters.
        """

        model.print_trainable_parameters()