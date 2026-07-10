"""
Adapter for the ShareGPT conversation dataset format.

Expected input format:

{
    "conversations": [
        {
            "from": "human",
            "value": "..."
        },
        {
            "from": "gpt",
            "value": "..."
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


class ShareGPTAdapter(BaseDatasetAdapter):
    """Converts ShareGPT conversations into the canonical DatasetSample schema."""

    ROLE_MAPPING = {
        "system": MessageRole.SYSTEM,
        "human": MessageRole.USER,
        "user": MessageRole.USER,
        "gpt": MessageRole.ASSISTANT,
        "assistant": MessageRole.ASSISTANT,
        "tool": MessageRole.TOOL,
    }

    def convert(
        self,
        sample: dict,
        sample_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetSample:

        messages: list[Message] = []

        for message in sample.get("conversations", []):
            role = self.ROLE_MAPPING.get(message["from"].lower())

            if role is None:
                raise ValueError(
                    f"Unsupported ShareGPT role: {message['from']}"
                )

            messages.append(
                Message(
                    role=role,
                    content=message["value"].strip(),
                )
            )

        return DatasetSample(
            id=sample_id,
            messages=messages,
            metadata=metadata or DatasetMetadata(),
        )