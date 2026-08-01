"""
Assistant-turn label masking.

During supervised fine-tuning only the assistant responses should
contribute to the loss. Every other token (system, user, tool) is
masked with the ``-100`` label used by Hugging Face loss functions.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    MessageRole,
)
from mentorai_finetuning.prompts.base import (
    BasePromptFormatter,
)
from mentorai_finetuning.tokenization.schema import (
    TokenizedSample,
)


class AssistantLabelMasker:
    """
    Builds training labels that ignore every non-assistant token.

    Boundaries are derived from the tokenizer's own chat template by
    formatting message prefixes. For every assistant turn ``i`` the
    start boundary is the length of the prefix rendered with
    ``add_generation_prompt=True`` and the end boundary is the length
    of the prefix that includes the assistant message itself.
    """

    def __init__(
        self,
        formatter: BasePromptFormatter,
    ) -> None:
        self.formatter = formatter

    def mask(
        self,
        sample: DatasetSample,
        input_ids: list[int],
    ) -> list[int]:
        """
        Return labels where non-assistant tokens are replaced by -100.

        Parameters
        ----------
        sample:
            Canonical conversation being tokenized.

        input_ids:
            Token IDs of the fully rendered conversation. The returned
            labels are guaranteed to have the same length.

        Returns
        -------
        list[int]
            Training labels aligned with ``input_ids``.
        """

        if not sample.messages:
            return [-100] * len(input_ids)

        conversation = [
            {"role": message.role.value, "content": message.content}
            for message in sample.messages
        ]

        boundaries = self._assistant_boundaries(conversation)

        if not boundaries:
            return [-100] * len(input_ids)

        labels = [-100] * len(input_ids)

        for start, end in boundaries:
            for index in range(start, min(end, len(input_ids))):
                labels[index] = input_ids[index]

        return labels

    def mask_sample(
        self,
        tokenized: TokenizedSample,
    ) -> TokenizedSample:
        """
        Return a copy of ``tokenized`` with masked labels.

        The tokenized sample must carry its formatted ``prompt`` so the
        assistant boundaries can be recovered from the tokenizer's chat
        template.
        """

        return tokenized.model_copy(
            update={"labels": self.mask(tokenized, tokenized.input_ids)},
        )

    def _assistant_boundaries(
        self,
        conversation: list[dict[str, str]],
    ) -> list[tuple[int, int]]:
        """
        Compute ``(start, end)`` token boundaries for every assistant
        message in ``conversation``.
        """

        boundaries: list[tuple[int, int]] = []

        for index, message in enumerate(conversation):
            if message["role"] != MessageRole.ASSISTANT.value:
                continue

            start = self._render_length(
                conversation[:index],
                add_generation_prompt=True,
            )

            end = self._render_length(
                conversation[: index + 1],
                add_generation_prompt=False,
            )

            if end > start:
                boundaries.append((start, end))

        return boundaries

    def _render_length(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> int:
        """
        Tokenize a message prefix and return its length.

        Returns 0 for an empty prefix.
        """

        if not conversation and not add_generation_prompt:
            return 0

        rendered = self.formatter.format_messages(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )

        if isinstance(rendered, list):
            return len(rendered)

        # Modern transformers returns a BatchEncoding when the template
        # requests tokenization.
        input_ids = getattr(rendered, "input_ids", None)

        if input_ids is not None:
            return len(input_ids)

        raise TypeError(
            "Expected tokenized output from the prompt formatter."
        )
