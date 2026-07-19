"""
Inference request schemas.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class InferenceRole(str, Enum):
    """
    Supported inference roles.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class InferenceMessage(BaseModel):
    """
    A single inference message.
    """

    role: InferenceRole

    content: str = Field(
        min_length=1,
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class InferenceRequest(BaseModel):
    """
    Canonical inference request used by the deployment layer.

    Every API (REST, OpenAI, CLI, future WebSocket, etc.) converts its
    input into this schema before prompt construction.
    """

    messages: list[InferenceMessage] = Field(
        min_length=1,
    )

    model_config = ConfigDict(
        extra="forbid",
    )