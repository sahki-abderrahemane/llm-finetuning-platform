"""
API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from mentorai_finetuning.deployment.api.dependencies import (
    get_pipeline,
)
from mentorai_finetuning.deployment.api.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from mentorai_finetuning.deployment.pipeline import (
    DeploymentPipeline,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    pipeline: DeploymentPipeline = Depends(
        get_pipeline,
    ),
) -> HealthResponse:
    """
    Health check endpoint.
    """

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=pipeline.config.model_name,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(
    request: GenerateRequest,
    pipeline: DeploymentPipeline = Depends(
        get_pipeline,
    ),
) -> GenerateResponse:
    """
    Generate text from a single prompt.
    """

    result = pipeline.generate_prompt(
        request.prompt,
    )

    return GenerateResponse(
        response=result.response,
        model_name=result.model_name,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        generation_time=result.generation_time,
    )