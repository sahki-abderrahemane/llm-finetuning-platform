"""
Base interface for evaluation metrics.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from mentorai_finetuning.evaluation.results import (
    MetricResult,
)


class EvaluationMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable metric name.
        """

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """
        Compute the metric score.
        """