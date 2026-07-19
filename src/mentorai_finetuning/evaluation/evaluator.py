"""
Evaluation engine.

Runs one or more evaluation metrics on model predictions.
"""

from __future__ import annotations

from mentorai_finetuning.evaluation.metrics.base import (
    EvaluationMetric,
)
from mentorai_finetuning.evaluation.report import (
    EvaluationReport,
)
from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class ModelEvaluator:
    """
    Executes a collection of evaluation metrics.
    """

    def __init__(
        self,
        metrics: list[EvaluationMetric],
    ) -> None:
        self.metrics = metrics

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        model_name: str,
        dataset_name: str,
    ) -> EvaluationReport:
        """
        Evaluate predictions using every configured metric.
        """

        results: list[MetricResult] = []

        for metric in self.metrics:
            results.append(
                metric.compute(
                    predictions,
                    references,
                )
            )

        return EvaluationReport(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=results,
        )