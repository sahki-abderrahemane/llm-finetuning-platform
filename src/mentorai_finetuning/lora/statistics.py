"""
Utilities for inspecting LoRA models.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel


@dataclass(slots=True)
class ModelStatistics:
    """
    Statistics describing the trainable state of a model.
    """

    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float


class LoRAStatistics:
    """
    Computes statistics for LoRA-enabled models.
    """

    @staticmethod
    def compute(
        model: PreTrainedModel,
    ) -> ModelStatistics:
        """
        Compute parameter statistics.
        """

        total = 0
        trainable = 0

        for parameter in model.parameters():
            count = parameter.numel()

            total += count

            if parameter.requires_grad:
                trainable += count

        frozen = total - trainable

        percentage = (
            (trainable / total) * 100.0
            if total > 0
            else 0.0
        )

        return ModelStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            frozen_parameters=frozen,
            trainable_percentage=percentage,
        )

    @staticmethod
    def print_summary(
        model: PreTrainedModel,
    ) -> None:
        """
        Print a human-readable statistics summary.
        """

        stats = LoRAStatistics.compute(model)

        print()

        print("=" * 60)
        print("LoRA Model Statistics")
        print("=" * 60)

        print(
            f"Total Parameters      : {stats.total_parameters:,}"
        )

        print(
            f"Trainable Parameters  : {stats.trainable_parameters:,}"
        )

        print(
            f"Frozen Parameters     : {stats.frozen_parameters:,}"
        )

        print(
            f"Trainable Percentage  : "
            f"{stats.trainable_percentage:.4f}%"
        )

        print("=" * 60)