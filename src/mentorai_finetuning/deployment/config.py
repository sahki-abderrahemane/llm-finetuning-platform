"""
Deployment configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mentorai_finetuning.deployment.backend.types import (
    BackendType,
)


class DeploymentConfig(BaseModel):
    """
    Configuration for model deployment and inference.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_name: str = (
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    adapter_path: Path | None = None

    merged_model_path: Path | None = None

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    backend: BackendType = (
        BackendType.VLLM
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device: str = "auto"

    torch_dtype: str = "auto"

    trust_remote_code: bool = False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    max_new_tokens: int = Field(
        default=512,
        ge=1,
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
    )

    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )

    top_k: int = Field(
        default=50,
        ge=0,
    )

    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
    )

    do_sample: bool = True

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    use_cache: bool = True

    seed: int = 42

    model_config = ConfigDict(
        extra="forbid",
    )