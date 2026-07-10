"""
Adapter for the Stanford Alpaca instruction dataset format.

Expected input format:

{
    "instruction": "...",
    "input": "...",
    "output": "..."
}
"""

from __future__ import annotations

from mentorai_finetuning.dataset.adapters.base import BaseDatasetAdapter
from mentorai_finetuning.dataset.schema import (
    DatasetMetadata,
    DatasetSample,
    Message,
    MessageRole,
)


class AlpacaAdapter(BaseDatasetAdapter):
    """Converts Alpaca samples into the canonical DatasetSample schema."""

    def convert(
        self,
        sample: dict,
        sample_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetSample:
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if input_text:
            user_prompt = (
                f"{instruction}\n\n"
                f"Input:\n"
                f"{input_text}"
            )
        else:
            user_prompt = instruction

        return DatasetSample(
            id=sample_id,
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=user_prompt,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=output,
                ),
            ],
            metadata=metadata or DatasetMetadata(),
        )