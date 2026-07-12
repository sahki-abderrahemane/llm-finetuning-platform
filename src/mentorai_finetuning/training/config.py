"""
Training configuration.

"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """
    Configuration for supervised fine-tuning.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_dataset: Path = Path("data/processed/train.json")
    validation_dataset: Path = Path("data/processed/validation.json")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    output_dir: Path = Path("models/checkpoints")

    overwrite_output_dir: bool = True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    epochs: int = Field(
        default=3,
        ge=1,
    )

    learning_rate: float = Field(
        default=2e-5,
        gt=0,
    )

    train_batch_size: int = Field(
        default=2,
        ge=1,
    )

    eval_batch_size: int = Field(
        default=2,
        ge=1,
    )

    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
    )

    max_sequence_length: int = Field(
        default=2048,
        ge=128,
    )

    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
    )

    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    logging_steps: int = Field(
        default=10,
        ge=1,
    )

    save_steps: int = Field(
        default=100,
        ge=1,
    )

    evaluation_steps: int = Field(
        default=100,
        ge=1,
    )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    seed: int = 42

    # ------------------------------------------------------------------
    # Mixed Precision
    # ------------------------------------------------------------------

    fp16: bool = False
    bf16: bool = False

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    remove_unused_columns: bool = False

    model_config = ConfigDict(
        extra="forbid",
    )