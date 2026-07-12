"""
Generic prompt formatter.

"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.dataset.schema import DatasetSample
from mentorai_finetuning.prompts.base import BasePromptFormatter


class PromptFormatter(BasePromptFormatter):
    """
    Generic prompt formatter compatible with modern Hugging Face chat models.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(tokenizer)

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
        Format a DatasetSample using the tokenizer's official chat template.
        """

        conversation = self._build_chat(sample)

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )