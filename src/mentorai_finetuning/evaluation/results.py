"""
Evaluation result models.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """
    Result produced by a single evaluation metric.
    """

    name: str

    score: float

    model_config = ConfigDict(
        extra="forbid",
    )


class EvaluationResult(BaseModel):
    """
    Complete evaluation results for one experiment.
    """

    experiment_name: str

    checkpoint_dir: Path

    metrics: list[MetricResult] = Field(
        default_factory=list,
    )

    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    model_config = ConfigDict(
        extra="forbid",
    )