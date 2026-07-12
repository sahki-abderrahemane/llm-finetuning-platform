"""
Prompt formatter registry.

Maps model names to the appropriate prompt formatter.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.prompts.formatter import PromptFormatter

# from mentorai_finetuning.prompts.qwen import QwenPromptFormatter
# from mentorai_finetuning.prompts.llama import LlamaPromptFormatter
# from mentorai_finetuning.prompts.gemma import GemmaPromptFormatter
# from mentorai_finetuning.prompts.mistral import MistralPromptFormatter
# from mentorai_finetuning.prompts.phi import PhiPromptFormatter


class PromptFormatterRegistry:
    """
    Registry responsible for selecting the appropriate prompt formatter.

    At the moment every supported chat model uses the generic
    PromptFormatter, since modern Hugging Face tokenizers already expose
    their official chat template through `apply_chat_template()`.

    If a future model requires custom formatting, only this registry
    needs to change.
    """

    _FORMATTERS: dict[str, type[PromptFormatter]] = {
        "default": PromptFormatter,
    }

    @classmethod
    def register(
        cls,
        model_type: str,
        formatter: type[PromptFormatter],
    ) -> None:
        """
        Register a new formatter.
        """

        cls._FORMATTERS[model_type.lower()] = formatter

    @classmethod
    def create(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "default",
    ) -> PromptFormatter:
        """
        Create the appropriate formatter.
        """

        formatter_class = cls._FORMATTERS.get(
            model_type.lower(),
            cls._FORMATTERS["default"],
        )

        return formatter_class(tokenizer)