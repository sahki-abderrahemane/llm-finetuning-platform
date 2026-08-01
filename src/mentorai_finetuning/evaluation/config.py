"""
Evaluation configuration models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMJudgeConfig(BaseModel):
    """
    Configuration for LLM-as-a-judge evaluation.
    """

    enabled: bool = False

    model_name: str = "gpt-4o-mini"

    judge: Literal["single", "pairwise"] = "single"

    max_tokens: int = 1024

    temperature: float = 0.0

    criteria: list[str] = Field(
        default_factory=lambda: [
            "coherence",
            "helpfulness",
            "accuracy",
        ],
    )
