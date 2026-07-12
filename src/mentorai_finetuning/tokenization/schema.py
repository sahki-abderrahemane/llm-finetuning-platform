"""
Schemas for tokenized samples.

"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TokenizedSample(BaseModel):
    """
    A single tokenized training example.

    This is the canonical representation exchanged between the
    tokenization pipeline and the training pipeline.
    """

    model_config = ConfigDict(frozen=True)

    input_ids: list[int] = Field(
        description="Token ids fed into the model.",
    )

    attention_mask: list[int] = Field(
        description="Attention mask corresponding to input_ids.",
    )

    labels: list[int] = Field(
        description=(
            "Training labels. Tokens with value -100 are ignored "
            "during loss computation."
        ),
    )

    prompt: str | None = Field(
        default=None,
        description="Formatted prompt used to generate this sample.",
    )

    sequence_length: int = Field(
        description="Number of tokens in the sequence.",
    )

    @property
    def num_tokens(self) -> int:
        """Alias for sequence_length."""

        return self.sequence_length

    @property
    def trainable_tokens(self) -> int:
        """
        Number of tokens contributing to the loss.
        """

        return sum(label != -100 for label in self.labels)

    @property
    def ignored_tokens(self) -> int:
        """
        Number of ignored tokens.
        """

        return sum(label == -100 for label in self.labels)

    def to_dict(self) -> dict[str, list[int]]:
        """
        Convert to the dictionary format expected by Hugging Face datasets
        and collators.
        """

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }