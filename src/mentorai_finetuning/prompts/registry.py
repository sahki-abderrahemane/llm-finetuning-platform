"""
Prompt formatter registry.

Maps model names/families to the appropriate prompt formatter.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)
from mentorai_finetuning.prompts.gemma import GemmaPromptFormatter
from mentorai_finetuning.prompts.llama import LlamaPromptFormatter
from mentorai_finetuning.prompts.mistral import MistralPromptFormatter
from mentorai_finetuning.prompts.phi import PhiPromptFormatter
from mentorai_finetuning.prompts.qwen import QwenPromptFormatter


class PromptFormatterRegistry:
    """
    Registry responsible for selecting the appropriate prompt formatter.

    Every supported chat model uses ``apply_chat_template`` through a
    formatter subclass. The subclass only matters when a tokenizer does
    not ship an official chat template, in which case it provides a
    built-in fallback for its model family.
    """

    _FORMATTERS: dict[str, type[ChatTemplateFormatter]] = {
        "default": ChatTemplateFormatter,
        "qwen": QwenPromptFormatter,
        "llama": LlamaPromptFormatter,
        "mistral": MistralPromptFormatter,
        "phi": PhiPromptFormatter,
        "gemma": GemmaPromptFormatter,
    }

    _FAMILY_MARKERS: list[tuple[tuple[str, ...], str]] = [
        (("qwen",), "qwen"),
        (("llama", "llama3", "llama-2"), "llama"),
        (("mistral", "mixtral"), "mistral"),
        (("phi",), "phi"),
        (("gemma",), "gemma"),
    ]

    @classmethod
    def register(
        cls,
        model_type: str,
        formatter: type[ChatTemplateFormatter],
    ) -> None:
        """
        Register a new formatter.
        """

        cls._FORMATTERS[model_type.lower()] = formatter

    @classmethod
    def _detect_family(
        cls,
        model_type: str,
    ) -> str:
        """
        Detect the model family from a model name.
        """

        name = model_type.lower()

        for markers, family in cls._FAMILY_MARKERS:
            if any(marker in name for marker in markers):
                return family

        return "default"

    @classmethod
    def create(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "default",
    ) -> ChatTemplateFormatter:
        """
        Create the appropriate formatter.
        """

        family = cls._detect_family(model_type)

        formatter_class = cls._FORMATTERS.get(
            family,
            cls._FORMATTERS["default"],
        )

        return formatter_class(tokenizer)
