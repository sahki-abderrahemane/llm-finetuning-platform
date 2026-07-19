"""
FastAPI dependencies.
"""

from __future__ import annotations

from functools import lru_cache

from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)
from mentorai_finetuning.deployment.pipeline import (
    DeploymentPipeline,
)


@lru_cache(maxsize=1)
def get_pipeline() -> DeploymentPipeline:
    """
    Return the deployment pipeline singleton.
    """

    config = DeploymentConfig()

    return DeploymentPipeline(
        config,
    )