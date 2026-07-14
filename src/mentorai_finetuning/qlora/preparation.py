"""
Utilities for preparing quantized models for QLoRA training.
"""

from __future__ import annotations

from peft import prepare_model_for_kbit_training
from transformers import PreTrainedModel


class QLoRAPreparer:
    """
    Prepares a quantized model for QLoRA training.
    """

    @staticmethod
    def prepare(
        model: PreTrainedModel,
        *,
        use_gradient_checkpointing: bool = True,
    ) -> PreTrainedModel:
        """
        Prepare a quantized model for k-bit training.

        Parameters
        ----------
        model:
            Quantized Hugging Face model.

        use_gradient_checkpointing:
            Enable gradient checkpointing during preparation.
        """

        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )