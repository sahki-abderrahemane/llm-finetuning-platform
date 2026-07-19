"""
BLEU evaluation metric.
"""

from __future__ import annotations

    
import evaluate

from .base import EvaluationMetric
from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class BLEUMetric(EvaluationMetric):
    """
    Computes the BLEU score.
    """

    def __init__(self) -> None:
        self.metric = evaluate.load(
            "bleu",
        )

    @property
    def name(self) -> str:
        """
        Metric name.
        """

        return "BLEU"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """
        Compute the BLEU score.
        """

        formatted_references = [
            [reference]
            for reference in references
        ]

        score = self.metric.compute(
            predictions=predictions,
            references=formatted_references,
        )

        return MetricResult(
            name=self.name,
            score=float(score["bleu"]),
        )