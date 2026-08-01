"""
FastAPI dependencies.
"""

from __future__ import annotations

from functools import lru_cache

from mentorai_finetuning.common.config import (
    get_settings,
)
from mentorai_finetuning.deployment.api.security import (
    ApiSecurity,
)
from mentorai_finetuning.deployment.backend.types import (
    BackendType,
)
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

    settings = get_settings()

    config = DeploymentConfig(
        model_name=settings.MODEL_NAME,
        backend=BackendType(settings.DEPLOY_BACKEND),
        adapter_path=settings.DEPLOY_ADAPTER_PATH,
        merged_model_path=settings.DEPLOY_MERGED_MODEL_PATH,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
    )

    return DeploymentPipeline(
        config,
    )


def get_api_security() -> ApiSecurity:
    """
    Build the API security instance from settings.
    """

    return ApiSecurity(
        api_key=get_settings().MENTORAI_API_KEY,
    )