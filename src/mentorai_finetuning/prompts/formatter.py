"""
Generic prompt formatter.

"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.prompts.base import BasePromptFormatter


class ChatTemplateFormatter(BasePromptFormatter):
    """
    Prompt formatter with a built-in fallback chat template.

    Modern Hugging Face tokenizers ship their official chat template,
    which is used as-is. If a tokenizer lacks one, the formatter falls
    back to the family template defined by ``CHAT_TEMPLATE``.
    """

    CHAT_TEMPLATE: str | None = None

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(tokenizer)

        self._ensure_template()

    def _ensure_template(self) -> None:
        """Install the fallback template when the tokenizer has none."""

        if self.tokenizer.chat_template is None and self.CHAT_TEMPLATE:
            self.tokenizer.chat_template = self.CHAT_TEMPLATE

    @staticmethod
    def _build_chat(
        sample: DatasetSample,
    ) -> list[dict[str, str]]:
        """
        Convert a DatasetSample into the conversation format expected by
        Hugging Face chat templates.
        """

        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in sample.messages
        ]

    def format(
        self,
        sample: DatasetSample,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        """
        Format a DatasetSample using the tokenizer's chat template.
        """

        return self.format_messages(
            self._build_chat(sample),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def format_messages(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        """
        Format a raw conversation using the tokenizer's chat template.
        """

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class PromptFormatter(ChatTemplateFormatter):
    """
    Generic prompt formatter compatible with modern Hugging Face chat models.
    """