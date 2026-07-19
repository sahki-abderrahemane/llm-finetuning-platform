"""
Utilities for comparing evaluation reports.
"""

from __future__ import annotations

from mentorai_finetuning.evaluation.report import (
    EvaluationReport,
)


class EvaluationComparison:
    """
    Compares multiple evaluation reports.
    """

    def __init__(
        self,
        reports: list[EvaluationReport],
    ) -> None:
        self.reports = reports

    def best_model(
        self,
        metric_name: str,
    ) -> EvaluationReport:
        """
        Return the report with the highest score for a metric.
        """

        return max(
            self.reports,
            key=lambda report: (
                report.metric(metric_name).score
                if report.metric(metric_name) is not None
                else float("-inf")
            ),
        )

    def summary(
        self,
    ) -> list[dict[str, float | str]]:
        """
        Return a summary of every report.
        """

        summaries: list[
            dict[str, float | str]
        ] = []

        for report in self.reports:

            row: dict[
                str,
                float | str,
            ] = {
                "model": report.model_name,
                "dataset": report.dataset_name,
            }

            for metric in report.metrics:
                row[
                    metric.name
                ] = metric.score

            summaries.append(
                row,
            )

        return summaries