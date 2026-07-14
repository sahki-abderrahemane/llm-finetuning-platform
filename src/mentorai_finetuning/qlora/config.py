"""
Configuration for Quantized LoRA (QLoRA).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QLoRAConfig(BaseModel):
    """
    Configuration used when loading a model for QLoRA training.

    This configuration controls how the base model is quantized
    and loaded into memory. LoRA adapter settings remain in
    LoRAConfig.
    """

    load_in_4bit: bool = Field(
        default=True,
        description="Load the model using 4-bit quantization.",
    )

    quantization_type: str = Field(
        default="nf4",
        description="4-bit quantization type (nf4 or fp4).",
    )

    compute_dtype: str = Field(
        default="float16",
        description="Compute dtype used during forward passes.",
    )

    use_double_quant: bool = Field(
        default=True,
        description="Enable double quantization.",
    )

    device_map: str = Field(
        default="auto",
        description="Device placement strategy.",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Allow execution of remote modeling code.",
    )

    model_config = ConfigDict(
        extra="forbid",
    )