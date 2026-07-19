"""
Inference request builder.

Converts canonical inference requests into the prompt format expected by
the target language model.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.deployment.inference import (
    InferenceRequest,
)


class InferenceRequestBuilder:
    """
    Builds model-ready prompts from canonical inference requests.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _build_chat(
        request: InferenceRequest,
    ) -> list[dict[str, str]]:
        """
        Convert an InferenceRequest into the conversation format expected
        by Hugging Face chat templates.
        """

        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]

    def build(
        self,
        request: InferenceRequest,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        """
        Build the prompt expected by the target model.

        Parameters
        ----------
        request:
            Canonical inference request.

        tokenize:
            Whether to return token ids instead of text.

        add_generation_prompt:
            Whether to append the assistant generation prompt.

        Returns
        -------
        str | list[int]
            Formatted prompt or token ids.
        """

        conversation = self._build_chat(
            request,
        )

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )