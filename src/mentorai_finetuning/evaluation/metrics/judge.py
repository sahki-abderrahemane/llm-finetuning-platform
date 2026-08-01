"""
LLM-as-a-judge evaluation metric.

Scores generated responses against references using a judge model
through the OpenAI-compatible chat completions API.
"""

from __future__ import annotations

import httpx
from loguru import logger

from mentorai_finetuning.common.config import (
    get_settings,
)
from mentorai_finetuning.evaluation.config import (
    LLMJudgeConfig,
)
from mentorai_finetuning.evaluation.results import (
    MetricResult,
)

from .base import EvaluationMetric

_JUDGE_SYSTEM = (
    "You are an impartial evaluator. Score the candidate answer "
    "against the reference answer on a scale of 0 to 5 for each "
    "criterion, where 5 is the best. Respond with a single integer "
    "score only."
)


class LLMJudgeMetric(EvaluationMetric):
    """
    LLM-as-a-judge metric using an OpenAI-compatible API.

    Requires ``OPENAI_API_KEY`` to be configured. When the key is
    missing the metric logs a warning and returns a score of 0.0.
    """

    def __init__(
        self,
        config: LLMJudgeConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or LLMJudgeConfig()
        self.api_key = (
            api_key
            if api_key is not None
            else get_settings().OPENAI_API_KEY
        )

    @property
    def name(self) -> str:
        """
        Metric name.
        """

        return "llm_judge"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """
        Score predictions against references using the judge model.
        """

        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY is not set; "
                "LLMJudgeMetric will return a score of 0.0."
            )

            return MetricResult(
                name=self.name,
                score=0.0,
            )

        if len(predictions) != len(references):
            raise ValueError(
                "predictions and references must have the same length."
            )

        scores: list[float] = []

        for prediction, reference in zip(
            predictions,
            references,
            strict=True,
        ):
            scores.append(
                self._score_single(
                    prediction,
                    reference,
                )
            )

        average = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        return MetricResult(
            name=self.name,
            score=average,
        )

    def _score_single(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """
        Score a single prediction/reference pair.
        """

        criteria = ", ".join(self.config.criteria)

        user_prompt = (
            f"Criterion: {criteria}\n"
            f"Reference: {reference}\n"
            f"Candidate: {prediction}\n"
            f"Score (0-5):"
        )

        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": _JUDGE_SYSTEM,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": (
                    f"Bearer {self.api_key}"
                ),
            },
            timeout=httpx.Timeout(60.0),
        )

        response.raise_for_status()

        data = response.json()

        content = (
            data["choices"][0]["message"]["content"]
        )

        return float(content.strip())
