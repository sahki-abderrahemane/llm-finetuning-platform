"""
Tests for assistant-turn label masking.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.prompts.formatter import PromptFormatter
from mentorai_finetuning.tokenization.config import TokenizerConfig
from mentorai_finetuning.tokenization.masking import AssistantLabelMasker
from mentorai_finetuning.tokenization.tokenizer_factory import TokenizerFactory


def build_masked_labels(messages, *, mask_non_assistant=True):
    """Tokenize a conversation and return input_ids + masked labels."""

    config = TokenizerConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )

    tokenizer = TokenizerFactory.create(config)

    sample = DatasetSample(
        id="mask-test",
        messages=messages,
    )

    formatter = PromptFormatter(tokenizer)

    prompt = formatter.format(sample, tokenize=False)

    input_ids = list(tokenizer(prompt)["input_ids"])

    if mask_non_assistant:
        labels = AssistantLabelMasker(formatter).mask(
            sample,
            input_ids,
        )
    else:
        labels = input_ids.copy()

    return input_ids, labels, tokenizer


def test_masking_keeps_only_assistant_tokens():
    """Only assistant response tokens should be trainable."""

    input_ids, labels, tokenizer = build_masked_labels(
        [
            Message(role=MessageRole.SYSTEM, content="You are MentorAI."),
            Message(role=MessageRole.USER, content="Explain LoRA."),
            Message(
                role=MessageRole.ASSISTANT,
                content="LoRA is a parameter-efficient fine-tuning method.",
            ),
        ]
    )

    assert len(labels) == len(input_ids)

    trainable = [
        token
        for label, token in zip(labels, input_ids, strict=True)
        if label != -100
    ]

    decoded = tokenizer.decode(trainable)

    assert "LoRA is a parameter-efficient fine-tuning method." in decoded

    assert "Explain LoRA." not in decoded


def test_masking_multi_turn():
    """Every assistant turn should remain trainable."""

    input_ids, labels, tokenizer = build_masked_labels(
        [
            Message(role=MessageRole.USER, content="What is LoRA?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="LoRA is parameter efficient.",
            ),
            Message(role=MessageRole.USER, content="And QLoRA?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="QLoRA adds 4-bit quantization.",
            ),
        ]
    )

    trainable = [
        token
        for label, token in zip(labels, input_ids, strict=True)
        if label != -100
    ]

    decoded = tokenizer.decode(trainable)

    assert "LoRA is parameter efficient" in decoded

    assert "QLoRA adds 4-bit quantization" in decoded

    assert "What is LoRA?" not in decoded


def test_no_assistant_turns_masks_everything():
    """A conversation with no assistant turn should produce all -100."""

    _, labels, _ = build_masked_labels(
        [
            Message(role=MessageRole.SYSTEM, content="You are MentorAI."),
            Message(role=MessageRole.USER, content="Hello."),
        ]
    )

    assert all(label == -100 for label in labels)
