"""
Adapter for the OpenAI Messages dataset format.

Expected input format:

{
    "messages": [
        {
            "role": "system",
            "content": "..."
        },
        {
            "role": "user",
            "content": "..."
        },
        {
            "role": "assistant",
            "content": "..."
        }
    ]
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


class OpenAIAdapter(BaseDatasetAdapter):
    """Converts OpenAI Messages into the canonical DatasetSample schema."""

    ROLE_MAPPING = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "tool": MessageRole.TOOL,
    }

    def convert(
        self,
        sample: dict,
        sample_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetSample:

        messages = [
            Message(
                role=self.ROLE_MAPPING[message["role"].lower()],
                content=message["content"].strip(),
            )
            for message in sample.get("messages", [])
        ]

        return DatasetSample(
            id=sample_id,
            messages=messages,
            metadata=metadata or DatasetMetadata(),
        )