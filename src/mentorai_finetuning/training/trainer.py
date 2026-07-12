"""
High-level training wrapper.

Provides a clean interface around the underlying Hugging Face/TRL trainer.
"""

from __future__ import annotations

from pathlib import Path

from trl import SFTTrainer


class SFTTrainingEngine:
    """
    Wrapper around SFTTrainer.
    """

    def __init__(
        self,
        trainer: SFTTrainer,
    ) -> None:
        self.trainer = trainer

    def train(self) -> None:
        """
        Start supervised fine-tuning.
        """

        self.trainer.train()

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model.
        """

        return self.trainer.evaluate()

    def save(
        self,
        output_dir: Path | str,
    ) -> None:
        """
        Save the trained model and tokenizer.
        """

        self.trainer.save_model(
            str(output_dir),
        )