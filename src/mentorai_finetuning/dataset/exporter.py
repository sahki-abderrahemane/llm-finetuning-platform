"""
Canonical dataset schema used throughout the MentorAI fine-tuning pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    """Supported conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message within a conversation."""

    role: MessageRole
    content: str = Field(..., min_length=1)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class DatasetMetadata(BaseModel):
    """Optional metadata associated with a dataset sample."""

    source: str | None = None
    language: str | None = None
    domain: str | None = None
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    quality_score: float | None = None

    model_config = ConfigDict(extra="allow")


class DatasetSample(BaseModel):
    """
    Canonical representation of a single training sample.

    Every supported dataset format is converted into this schema before
    entering the preprocessing and training pipeline.
    """

    id: str

    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Conversation messages in chronological order.",
    )

    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)

    model_config = ConfigDict(extra="forbid")


class Dataset(BaseModel):
    """A collection of dataset samples."""

    samples: list[DatasetSample]

    model_config = ConfigDict(extra="forbid")


DatasetDict = dict[str, Any]