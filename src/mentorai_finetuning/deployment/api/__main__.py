"""
Entry point for running the MentorAI Inference API.

Usage: python -m mentorai_finetuning.deployment.api
"""

from __future__ import annotations

import uvicorn

from mentorai_finetuning.common.config import (
    get_settings,
)


def main() -> None:
    """
    Start the FastAPI application with uvicorn.
    """

    settings = get_settings()

    uvicorn.run(
        "mentorai_finetuning.deployment.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
