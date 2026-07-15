"""
Experiment configuration models.

"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mentorai_finetuning.training.config import TrainingConfig


class ExperimentConfig(BaseModel):
    """
    Configuration describing a single training experiment.
    """

    name: str

    description: str = ""

    model_name: str

    dataset_name: str

    output_dir: Path

    training: TrainingConfig

    tags: list[str] = Field(
        default_factory=list,
    )

    experiment_id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    model_config = ConfigDict(
        extra="forbid",
    )