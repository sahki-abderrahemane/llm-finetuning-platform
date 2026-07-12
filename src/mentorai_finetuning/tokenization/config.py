"""
Tokenization configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    """
    Configuration for Hugging Face tokenizers.
    """

    model_name: str

    max_length: int = Field(
        default=2048,
        ge=1,
    )

    use_fast: bool = True

    trust_remote_code: bool = False

    padding_side: str = Field(
        default="right",
        pattern="^(left|right)$",
    )

    truncation_side: str = Field(
        default="right",
        pattern="^(left|right)$",
    )

    add_special_tokens: bool = True

    return_attention_mask: bool = True