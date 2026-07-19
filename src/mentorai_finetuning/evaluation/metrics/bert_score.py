"""
BERTScore evaluation metric.
"""

from __future__ import annotations

import evaluate

from .base import (
    EvaluationMetric,
)
from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class BERTScoreMetric(EvaluationMetric):
    """
    Computes the BERTScore F1 metric.
    """

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
    ) -> None:
        self.metric = evaluate.load(
            "bertscore",
        )

        self.model_type = model_type

    @property
    def name(
        self,
    ) -> str:
        """
        Metric name.
        """

        return "BERTScore"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """
        Compute the BERTScore F1.
        """

        scores = self.metric.compute(
            predictions=predictions,
            references=references,
            model_type=self.model_type,
        )

        f1 = sum(
            scores["f1"]
        ) / len(
            scores["f1"]
        )

        return MetricResult(
            name=self.name,
            score=float(f1),
        )