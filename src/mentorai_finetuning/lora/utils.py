"""
General utilities for working with LoRA / PEFT models.
"""

from __future__ import annotations

from peft import PeftModel
from transformers import PreTrainedModel


class LoRAUtils:
    """
    Utility methods for inspecting LoRA models.
    """

    @staticmethod
    def is_lora_model(
        model: PreTrainedModel,
    ) -> bool:
        """
        Returns True if the model is wrapped by PEFT.
        """

        return isinstance(model, PeftModel)

    @staticmethod
    def trainable_parameter_names(
        model: PreTrainedModel,
    ) -> list[str]:
        """
        Return the names of all trainable parameters.
        """

        return [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def frozen_parameter_names(
        model: PreTrainedModel,
    ) -> list[str]:
        """
        Return the names of all frozen parameters.
        """

        return [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ]

    @staticmethod
    def count_trainable_tensors(
        model: PreTrainedModel,
    ) -> int:
        """
        Return the number of trainable parameter tensors.
        """

        return sum(
            1
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def list_adapter_modules(
        model: PreTrainedModel,
    ) -> list[str]:
        """
        Return the module names that contain LoRA adapters.
        """

        modules: list[str] = []

        for name, module in model.named_modules():
            module_name = module.__class__.__name__.lower()

            if "lora" in module_name:
                modules.append(name)

        return modules

    @staticmethod
    def print_summary(
        model: PreTrainedModel,
    ) -> None:
        """
        Print a concise LoRA summary.
        """

        print()

        print("=" * 60)
        print("LoRA Summary")
        print("=" * 60)

        print(
            f"PEFT Model          : {LoRAUtils.is_lora_model(model)}"
        )

        print(
            f"Trainable Tensors   : "
            f"{LoRAUtils.count_trainable_tensors(model)}"
        )

        adapters = LoRAUtils.list_adapter_modules(model)

        print(
            f"Adapter Modules     : {len(adapters)}"
        )

        for module in adapters:
            print(f"  • {module}")

        print("=" * 60)