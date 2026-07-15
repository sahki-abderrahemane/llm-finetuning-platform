"""
Experiment result models.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExperimentResult(BaseModel):
    """
    Result of a completed experiment.
    """

    experiment_id: str

    experiment_name: str

    status: Literal[
        "SUCCESS",
        "FAILED",
    ]

    started_at: datetime

    finished_at: datetime

    duration_seconds: float

    checkpoint_dir: Path

    metrics: dict[str, float]

    model_config = ConfigDict(
        extra="forbid",
    )