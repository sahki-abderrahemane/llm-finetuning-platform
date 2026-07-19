"""
OpenAI-compatible API routes.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

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
) -> CompletionResponse:
    """
    OpenAI-compatible completion endpoint.
    """

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
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completion endpoint.
    """

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