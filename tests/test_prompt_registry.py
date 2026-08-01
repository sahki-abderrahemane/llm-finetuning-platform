"""
Tests for the prompt formatter registry.
"""

from __future__ import annotations

from types import SimpleNamespace

from mentorai_finetuning.prompts.formatter import (
    ChatTemplateFormatter,
)
from mentorai_finetuning.prompts.gemma import GemmaPromptFormatter
from mentorai_finetuning.prompts.llama import LlamaPromptFormatter
from mentorai_finetuning.prompts.mistral import MistralPromptFormatter
from mentorai_finetuning.prompts.phi import PhiPromptFormatter
from mentorai_finetuning.prompts.qwen import QwenPromptFormatter
from mentorai_finetuning.prompts.registry import PromptFormatterRegistry


def make_tokenizer():
    """Create a minimal tokenizer stub with a chat template."""

    return SimpleNamespace(
        chat_template="{{ messages }}",
        apply_chat_template=lambda conversation, **kwargs: "<prompt>",
    )


def test_default_formatter():
    """Unknown model families must fall back to the generic formatter."""

    formatter = PromptFormatterRegistry.create(
        make_tokenizer(),
        model_type="unknown-model",
    )

    assert isinstance(formatter, ChatTemplateFormatter)


def test_model_family_detection():
    """Model names must map to their family formatter."""

    cases = [
        ("Qwen/Qwen2.5-1.5B-Instruct", QwenPromptFormatter),
        ("meta-llama/Llama-3.1-8B-Instruct", LlamaPromptFormatter),
        ("mistralai/Mistral-7B-Instruct-v0.3", MistralPromptFormatter),
        ("microsoft/Phi-3-mini-4k-instruct", PhiPromptFormatter),
        ("google/gemma-2-9b-it", GemmaPromptFormatter),
    ]

    for model_type, expected in cases:
        formatter = PromptFormatterRegistry.create(
            make_tokenizer(),
            model_type=model_type,
        )

        assert isinstance(formatter, expected), model_type


def test_fallback_template_installed():
    """A formatter without a tokenizer template must install its own."""

    tokenizer = SimpleNamespace(chat_template=None)

    formatter = QwenPromptFormatter(tokenizer)

    assert tokenizer.chat_template is not None

    assert isinstance(formatter, ChatTemplateFormatter)


def test_existing_template_not_overridden():
    """An existing tokenizer template must be left untouched."""

    tokenizer = SimpleNamespace(chat_template="official-template")

    QwenPromptFormatter(tokenizer)

    assert tokenizer.chat_template == "official-template"
