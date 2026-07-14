"""
Configuration for Low-Rank Adaptation (LoRA).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LoRAConfig(BaseModel):
    """
    Configuration for PEFT LoRA adapters.
    """

    rank: int = Field(
        default=16,
        gt=0,
        description="Rank (r) of the low-rank matrices.",
    )

    alpha: int = Field(
        default=32,
        gt=0,
        description="Scaling factor applied to LoRA updates.",
    )

    dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Dropout applied on the LoRA branch during training.",
    )

    bias: str = Field(
        default="none",
        description="Bias handling strategy.",
    )

    task_type: str = Field(
        default="CAUSAL_LM",
        description="PEFT task type.",
    )

    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        description="Transformer modules that receive LoRA adapters.",
    )

    inference_mode: bool = Field(
        default=False,
        description="Whether adapters are used only for inference.",
    )

    model_config = ConfigDict(
        extra="forbid",
    )