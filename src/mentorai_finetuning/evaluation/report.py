"""
Evaluation report models.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class EvaluationReport(BaseModel):
    """
    Complete evaluation report for a model.
    """

    model_name: str

    dataset_name: str

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    metrics: list[MetricResult]

    model_config = ConfigDict(
        extra="forbid",
    )

    def metric(
        self,
        name: str,
    ) -> MetricResult | None:
        """
        Retrieve a metric by name.
        """

        for metric in self.metrics:
            if metric.name == name:
                return metric

        return None

    def to_dict(
        self,
    ) -> dict:
        """
        Convert the report into a dictionary.
        """

        return self.model_dump()

    def to_json(
        self,
    ) -> str:
        """
        Serialize the report as JSON.
        """

        return self.model_dump_json(
            indent=4,
        )