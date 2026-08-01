"""
Tests for the ChatML dataset adapter.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.adapters.chatml import ChatMLAdapter
from mentorai_finetuning.dataset.schema import MessageRole


def test_chatml_parses_text_format():
    """Raw ChatML text must be parsed into canonical messages."""

    sample = {
        "text": (
            "<|im_start|>system\nYou are MentorAI.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>"
        )
    }

    result = ChatMLAdapter().convert(sample, "chatml-1")

    roles = [message.role for message in result.messages]

    assert roles == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]

    assert result.messages[0].content == "You are MentorAI."
    assert result.messages[1].content == "Hello"
    assert result.messages[2].content == "Hi there"


def test_chatml_accepts_messages_array():
    """A structured messages array must be converted as well."""

    sample = {
        "messages": [
            {"role": "user", "content": "What is QLoRA?"},
            {"role": "assistant", "content": "4-bit quantization + LoRA."},
        ]
    }

    result = ChatMLAdapter().convert(sample, "chatml-2")

    assert result.messages[0].role == MessageRole.USER
    assert result.messages[1].role == MessageRole.ASSISTANT


def test_chatml_rejects_unsupported_role():
    """Unsupported roles must raise a clear error."""

    sample = {
        "messages": [
            {"role": "developer", "content": "Hello"},
        ]
    }

    try:
        ChatMLAdapter().convert(sample, "chatml-3")
    except ValueError as exc:
        assert "developer" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")


def test_chatml_rejects_missing_fields():
    """Samples without text or messages must raise."""

    try:
        ChatMLAdapter().convert({"foo": "bar"}, "chatml-4")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")
