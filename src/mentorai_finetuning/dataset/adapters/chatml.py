"""
Adapter for the ChatML dataset format.

ChatML renders each turn with ``<|im_start|>`` / ``<|im_end|>``
markers. A raw sample looks like:

{
    "text": "<|im_start|>system\\nYou are MentorAI.<|im_end|>\\n"
            "<|im_start|>user\\nHello<|im_end|>\\n"
            "<|im_start|>assistant\\nHi there<|im_end|>"
}

The adapter also accepts an already-structured ``messages`` array.
"""

from __future__ import annotations

import re

from mentorai_finetuning.dataset.adapters.base import BaseDatasetAdapter
from mentorai_finetuning.dataset.schema import (
    DatasetMetadata,
    DatasetSample,
    Message,
    MessageRole,
)

ROLE_MAPPING = {
    "system": MessageRole.SYSTEM,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
}

_TURN_PATTERN = re.compile(
    r"<\|im_start\|>\s*(\w+)\s*\n(.*?)<\|im_end\|>",
    re.DOTALL,
)


class ChatMLAdapter(BaseDatasetAdapter):
    """Converts ChatML samples into the canonical DatasetSample schema."""

    def convert(
        self,
        sample: dict,
        sample_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetSample:

        if "messages" in sample:
            messages = self._from_messages(sample["messages"])
        elif "text" in sample:
            messages = self._from_text(sample["text"])
        else:
            raise ValueError(
                "ChatML sample must contain a 'text' or 'messages' field."
            )

        return DatasetSample(
            id=sample_id,
            messages=messages,
            metadata=metadata or DatasetMetadata(),
        )

    def _from_messages(
        self,
        messages: list,
    ) -> list[Message]:
        """Convert a structured messages array."""

        converted: list[Message] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "ChatML messages must be objects."
                )

            role = message.get("role")

            if role not in ROLE_MAPPING:
                raise ValueError(
                    f"Unsupported ChatML role: {role}"
                )

            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError(
                    f"ChatML message '{role}' has non-string content."
                )

            converted.append(
                Message(
                    role=ROLE_MAPPING[role],
                    content=content.strip(),
                )
            )

        return converted

    def _from_text(
        self,
        text: str,
    ) -> list[Message]:
        """Parse a raw ChatML string into canonical messages."""

        messages: list[Message] = []

        for match in _TURN_PATTERN.finditer(text):
            role, content = match.group(1), match.group(2)

            if role not in ROLE_MAPPING:
                raise ValueError(
                    f"Unsupported ChatML role: {role}"
                )

            messages.append(
                Message(
                    role=ROLE_MAPPING[role],
                    content=content.strip(),
                )
            )

        if not messages:
            raise ValueError(
                "No valid ChatML turns found in text."
            )

        return messages
