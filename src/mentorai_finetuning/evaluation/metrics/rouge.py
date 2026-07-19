"""
ROUGE evaluation metric.
"""

from __future__ import annotations

import evaluate

from .base import (
    EvaluationMetric,
)
from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class ROUGEMetric(EvaluationMetric):
    """
    Computes the ROUGE-L score.
    """

    def __init__(
        self,
    ) -> None:
        self.metric = evaluate.load(
            "rouge",
        )

    @property
    def name(
        self,
    ) -> str:
        """
        Metric name.
        """

        return "ROUGE-L"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """
        Compute the ROUGE-L score.
        """

        scores = self.metric.compute(
            predictions=predictions,
            references=references,
        )

        return MetricResult(
            name=self.name,
            score=float(scores["rougeL"]),
        )