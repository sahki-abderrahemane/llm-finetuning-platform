"""
Factory for creating BitsAndBytes quantization configurations.
"""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from mentorai_finetuning.qlora.config import QLoRAConfig


class QuantizationFactory:
    """
    Creates Hugging Face BitsAndBytesConfig objects.
    """

    _DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    @classmethod
    def create(
        cls,
        config: QLoRAConfig,
    ) -> BitsAndBytesConfig:
        """
        Convert a QLoRAConfig into a BitsAndBytesConfig.
        """

        compute_dtype = cls._DTYPE_MAP.get(
            config.compute_dtype.lower()
        )

        if compute_dtype is None:
            supported = ", ".join(cls._DTYPE_MAP.keys())

            raise ValueError(
                f"Unsupported compute dtype: "
                f"{config.compute_dtype!r}. "
                f"Supported values: {supported}."
            )

        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )