"""
API request and response models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    """
    Text generation request.
    """

    prompt: str = Field(
        min_length=1,
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class GenerateResponse(BaseModel):
    """
    Text generation response.
    """

    response: str

    model_name: str

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    generation_time: float

    model_config = ConfigDict(
        extra="forbid",
    )


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str

    model_loaded: bool

    model_name: str

    model_config = ConfigDict(
        extra="forbid",
    )