"""
Utilities for inspecting QLoRA models.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel


@dataclass(slots=True)
class QuantizationStatistics:
    """
    Statistics describing a quantized model.
    """

    total_parameters: int
    trainable_parameters: int
    quantized: bool
    load_in_4bit: bool
    load_in_8bit: bool
    quantization_type: str | None
    compute_dtype: str | None


class QLoRAStatistics:
    """
    Utilities for inspecting quantized Hugging Face models.
    """

    @staticmethod
    def compute(
        model: PreTrainedModel,
    ) -> QuantizationStatistics:

        total = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        config = getattr(
            model,
            "config",
            None,
        )

        quant_config = getattr(
            config,
            "quantization_config",
            None,
        )

        if quant_config is None:
            return QuantizationStatistics(
                total_parameters=total,
                trainable_parameters=trainable,
                quantized=False,
                load_in_4bit=False,
                load_in_8bit=False,
                quantization_type=None,
                compute_dtype=None,
            )

        return QuantizationStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            quantized=True,
            load_in_4bit=getattr(
                quant_config,
                "load_in_4bit",
                False,
            ),
            load_in_8bit=getattr(
                quant_config,
                "load_in_8bit",
                False,
            ),
            quantization_type=getattr(
                quant_config,
                "bnb_4bit_quant_type",
                None,
            ),
            compute_dtype=str(
                getattr(
                    quant_config,
                    "bnb_4bit_compute_dtype",
                    None,
                )
            ),
        )

    @staticmethod
    def print_summary(
        model: PreTrainedModel,
    ) -> None:

        stats = QLoRAStatistics.compute(
            model,
        )

        print()
        print("=" * 60)
        print("QLoRA Statistics")
        print("=" * 60)

        print(
            f"Quantized            : {stats.quantized}"
        )

        print(
            f"4-bit                : {stats.load_in_4bit}"
        )

        print(
            f"8-bit                : {stats.load_in_8bit}"
        )

        print(
            f"Quantization Type    : {stats.quantization_type}"
        )

        print(
            f"Compute DType        : {stats.compute_dtype}"
        )

        print(
            f"Total Parameters     : {stats.total_parameters:,}"
        )

        print(
            f"Trainable Parameters : {stats.trainable_parameters:,}"
        )

        print("=" * 60)