"""
OpenAI-compatible API routes.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from mentorai_finetuning.deployment.api.dependencies import (
    get_pipeline,
)
from mentorai_finetuning.deployment.api.openai_models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)
from mentorai_finetuning.deployment.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceRole,
)
from mentorai_finetuning.deployment.pipeline import (
    DeploymentPipeline,
)

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI Compatible"],
)


def _chunk_text(
    text: str,
    size: int = 32,
) -> list[str]:
    """
    Split text into chunks of at most ``size`` characters.
    """

    if not text:
        return []

    return [
        text[index : index + size]
        for index in range(0, len(text), size)
    ]


def _sse(
    data: object,
) -> str:
    """
    Format an object as a server-sent-event data payload.
    """

    return f"data: {json.dumps(data)}\n\n"


def _stream_chat_completions(
    request: ChatCompletionRequest,
    pipeline: DeploymentPipeline,
) -> StreamingResponse:
    """
    Stream chat completions as server-sent events.
    """

    async def generate() -> AsyncGenerator[str, None]:

        inference_request = InferenceRequest(
            messages=[
                InferenceMessage(
                    role=InferenceRole(message.role),
                    content=message.content,
                )
                for message in request.messages
            ],
        )

        result = pipeline.generate(
            inference_request,
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        for chunk in _chunk_text(result.response):

            payload = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": result.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield _sse(payload)

        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": result.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": result.finish_reason,
                }
            ],
        }

        yield _sse(payload)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_completions(
    request: CompletionRequest,
    pipeline: DeploymentPipeline,
) -> StreamingResponse:
    """
    Stream completions as server-sent events.
    """

    async def generate() -> AsyncGenerator[str, None]:

        result = pipeline.generate_prompt(
            request.prompt,
        )

        response_id = f"cmpl-{uuid.uuid4().hex}"

        for chunk in _chunk_text(result.response):

            payload = {
                "id": response_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": result.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }
                ],
            }

            yield _sse(payload)

        payload = {
            "id": response_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": result.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": result.finish_reason,
                }
            ],
        }

        yield _sse(payload)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
)
def list_models(
    pipeline: DeploymentPipeline = Depends(
        get_pipeline,
    ),
) -> ModelsResponse:
    """
    List available models.
    """

    return ModelsResponse(
        data=[
            ModelInfo(
                id=pipeline.config.model_name,
            )
        ],
    )


@router.post(
    "/completions",
    response_model=CompletionResponse,
)
def completions(
    request: CompletionRequest,
    pipeline: DeploymentPipeline = Depends(
        get_pipeline,
    ),
) -> CompletionResponse | StreamingResponse:
    """
    OpenAI-compatible completion endpoint.
    """

    if request.stream:
        return _stream_completions(
            request,
            pipeline,
        )

    result = pipeline.generate_prompt(
        request.prompt,
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        model=result.model_name,
        choices=[
            CompletionChoice(
                index=0,
                text=result.response,
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
    )


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    request: ChatCompletionRequest,
    pipeline: DeploymentPipeline = Depends(
        get_pipeline,
    ),
) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completion endpoint.
    """

    if request.stream:
        return _stream_chat_completions(
            request,
            pipeline,
        )

    inference_request = InferenceRequest(
        messages=[
            InferenceMessage(
                role=InferenceRole(message.role),
                content=message.content,
            )
            for message in request.messages
        ],
    )

    result = pipeline.generate(
        inference_request,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=result.model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.response,
                ),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
    )
