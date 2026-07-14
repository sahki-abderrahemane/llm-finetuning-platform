"""
Utilities for merging LoRA adapters into the base model.
"""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import PreTrainedModel


class LoRAMerger:
    """
    Merge trained LoRA adapters into the base model.
    """

    @staticmethod
    def merge(
        model: PeftModel,
    ) -> PreTrainedModel:
        """
        Merge adapters into the base model.

        Returns
        -------
        PreTrainedModel
            A standard Hugging Face model without PEFT wrappers.
        """

        return model.merge_and_unload()

    @staticmethod
    def save(
        model: PreTrainedModel,
        output_dir: Path | str,
    ) -> None:
        """
        Save the merged model.
        """

        model.save_pretrained(
            str(output_dir),
        )

    @staticmethod
    def merge_and_save(
        model: PeftModel,
        output_dir: Path | str,
    ) -> PreTrainedModel:
        """
        Merge adapters and immediately save the resulting model.
        """

        merged_model = model.merge_and_unload()

        merged_model.save_pretrained(
            str(output_dir),
        )

        return merged_model