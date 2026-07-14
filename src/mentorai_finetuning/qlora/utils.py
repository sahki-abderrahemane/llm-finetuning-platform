"""
General utilities for working with QLoRA models.
"""

from __future__ import annotations

from collections import Counter

from transformers import PreTrainedModel


class QLoRAUtils:
    """
    Utility methods for inspecting quantized models.
    """

    @staticmethod
    def is_quantized(
        model: PreTrainedModel,
    ) -> bool:
        """
        Return True if the model has a quantization configuration.
        """

        config = getattr(model, "config", None)

        return getattr(
            config,
            "quantization_config",
            None,
        ) is not None

    @staticmethod
    def is_4bit(
        model: PreTrainedModel,
    ) -> bool:
        """
        Return True if the model was loaded in 4-bit.
        """

        quant_config = getattr(
            model.config,
            "quantization_config",
            None,
        )

        if quant_config is None:
            return False

        return getattr(
            quant_config,
            "load_in_4bit",
            False,
        )

    @staticmethod
    def gradient_checkpointing_enabled(
        model: PreTrainedModel,
    ) -> bool:
        """
        Check whether gradient checkpointing is enabled.
        """

        return getattr(
            model,
            "is_gradient_checkpointing",
            False,
        )

    @staticmethod
    def parameter_dtypes(
        model: PreTrainedModel,
    ) -> dict[str, int]:
        """
        Count parameters by dtype.
        """

        counter = Counter()

        for parameter in model.parameters():
            counter[str(parameter.dtype)] += parameter.numel()

        return dict(counter)

    @staticmethod
    def print_summary(
        model: PreTrainedModel,
    ) -> None:
        """
        Print a concise QLoRA summary.
        """

        print()

        print("=" * 60)
        print("QLoRA Summary")
        print("=" * 60)

        print(
            f"Quantized               : "
            f"{QLoRAUtils.is_quantized(model)}"
        )

        print(
            f"4-bit                   : "
            f"{QLoRAUtils.is_4bit(model)}"
        )

        print(
            f"Gradient Checkpointing  : "
            f"{QLoRAUtils.gradient_checkpointing_enabled(model)}"
        )

        print()

        print("Parameter DTypes")

        for dtype, count in QLoRAUtils.parameter_dtypes(
            model,
        ).items():
            print(f"  {dtype:<18} {count:,}")

        print("=" * 60)