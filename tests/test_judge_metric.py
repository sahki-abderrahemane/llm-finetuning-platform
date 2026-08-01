"""
Tests for the LLM-as-a-judge metric.
"""

from __future__ import annotations

from unittest import mock

from mentorai_finetuning.evaluation.metrics.judge import (
    LLMJudgeMetric,
)


def test_judge_metric_missing_key_skips():
    """Without an API key the metric must return 0.0 gracefully."""

    metric = LLMJudgeMetric(api_key=None)

    result = metric.compute(
        predictions=["answer"],
        references=["answer"],
    )

    assert result.name == "llm_judge"

    assert result.score == 0.0


def test_judge_metric_scores_average():
    """Scores from the judge API must be averaged."""

    metric = LLMJudgeMetric(api_key="test-key")

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "5",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "3",
                    }
                }
            ]
        },
    ]

    with mock.patch(
        "httpx.post",
    ) as post:
        post.side_effect = [
            mock.Mock(
                json=lambda body=r: body,
                raise_for_status=lambda: None,
            )
            for r in responses
        ]

        result = metric.compute(
            predictions=["good", "ok"],
            references=["reference", "reference"],
        )

    assert result.score == 4.0


def test_judge_metric_requires_equal_lengths():
    """Mismatched predictions/references must raise."""

    metric = LLMJudgeMetric(api_key="test-key")

    try:
        metric.compute(
            predictions=["a", "b"],
            references=["a"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")
