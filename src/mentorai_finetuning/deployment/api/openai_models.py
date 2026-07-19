"""
OpenAI-compatible API models.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """
    A single chat message.
    """

    role: Literal[
        "system",
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    """

    model: str | None = None

    messages: list[ChatMessage] = Field(
        min_length=1,
    )

    max_tokens: int | None = None

    temperature: float | None = None

    top_p: float | None = None

    stream: bool = False

    model_config = ConfigDict(
        extra="ignore",
    )


class CompletionRequest(BaseModel):
    """
    OpenAI-compatible completion request.
    """

    model: str | None = None

    prompt: str = Field(
        min_length=1,
    )

    max_tokens: int | None = None

    temperature: float | None = None

    top_p: float |None = None

    stream: bool = False

    model_config = ConfigDict(
        extra="ignore",
    )


class ChatChoice(BaseModel):
    """
    Chat completion choice.
    """

    index: int

    message: ChatMessage

    finish_reason: str

    model_config = ConfigDict(
        extra="forbid",
    )


class CompletionChoice(BaseModel):
    """
    Completion choice.
    """

    index: int

    text: str

    finish_reason: str

    model_config = ConfigDict(
        extra="forbid",
    )


class Usage(BaseModel):
    """
    Token usage.
    """

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    model_config = ConfigDict(
        extra="forbid",
    )


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    """

    id: str

    object: Literal["chat.completion"] = "chat.completion"

    created: int = Field(
        default_factory=lambda: int(time.time()),
    )

    model: str

    choices: list[ChatChoice]

    usage: Usage

    model_config = ConfigDict(
        extra="forbid",
    )


class CompletionResponse(BaseModel):
    """
    OpenAI-compatible completion response.
    """

    id: str

    object: Literal["text_completion"] = "text_completion"

    created: int = Field(
        default_factory=lambda: int(time.time()),
    )

    model: str

    choices: list[CompletionChoice]

    usage: Usage

    model_config = ConfigDict(
        extra="forbid",
    )


class ModelInfo(BaseModel):
    """
    A served model.
    """

    id: str

    object: Literal["model"] = "model"

    owned_by: str = "mentorai"

    model_config = ConfigDict(
        extra="forbid",
    )


class ModelsResponse(BaseModel):
    """
    OpenAI-compatible models response.
    """

    object: Literal["list"] = "list"

    data: list[ModelInfo]

    model_config = ConfigDict(
        extra="forbid",
    )