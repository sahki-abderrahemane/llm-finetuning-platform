"""
FastAPI application.
"""

from __future__ import annotations

from fastapi import FastAPI

from mentorai_finetuning.deployment.api.openai_routes import (
    router as openai_router,
)
from mentorai_finetuning.deployment.api.routes import (
    router as api_router,
)

app = FastAPI(
    title="MentorAI Inference API",
    description=(
        "REST API for serving MentorAI language models."
    ),
    version="1.0.0",
)

app.include_router(
    api_router,
)

app.include_router(
    openai_router,
)