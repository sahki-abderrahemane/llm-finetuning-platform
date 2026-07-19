"""
Deployment response models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GenerationResponse(BaseModel):
    """
    Response returned by the deployment pipeline.
    """

    prompt: str

    response: str

    model_name: str

    finish_reason: str = "stop"

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    generation_time: float

    model_config = ConfigDict(
        extra="forbid",
    )