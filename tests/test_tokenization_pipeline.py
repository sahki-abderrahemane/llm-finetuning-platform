"""
Integration tests for the tokenization pipeline.
"""

from __future__ import annotations

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.prompts.formatter import PromptFormatter
from mentorai_finetuning.tokenization.collator import (
    DataCollator,
)
from mentorai_finetuning.tokenization.config import TokenizerConfig
from mentorai_finetuning.tokenization.dataset_tokenizer import DatasetTokenizer
from mentorai_finetuning.tokenization.packing import NoPackingStrategy
from mentorai_finetuning.tokenization.pipeline import TokenizationPipeline
from mentorai_finetuning.tokenization.tokenizer_factory import TokenizerFactory
from mentorai_finetuning.tokenization.tokenizer_wrapper import TokenizerWrapper


def build_pipeline() -> TokenizationPipeline:
    """
    Create a complete tokenization pipeline.
    """

    config = TokenizerConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )

    tokenizer = TokenizerFactory.create(config)

    formatter = PromptFormatter(tokenizer)

    wrapper = TokenizerWrapper(
        tokenizer=tokenizer,
        formatter=formatter,
        config=config,
    )

    dataset_tokenizer = DatasetTokenizer(wrapper)

    collator = DataCollator(tokenizer)

    return TokenizationPipeline(
        tokenizer=dataset_tokenizer,
        packing_strategy=NoPackingStrategy(),
        collator=collator,
    )


def build_sample() -> DatasetSample:
    """
    Create a sample conversation.
    """

    return DatasetSample(
        id="sample-1",
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are MentorAI.",
            ),
            Message(
                role=MessageRole.USER,
                content="Explain LoRA.",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="LoRA is a parameter-efficient fine-tuning method.",
            ),
        ],
    )


def test_tokenize_single_sample() -> None:
    """
    Tokenize one sample.
    """

    pipeline = build_pipeline()

    sample = build_sample()

    tokenized = pipeline.tokenize_sample(sample)

    assert tokenized.sequence_length > 0

    assert len(tokenized.input_ids) == len(
        tokenized.attention_mask
    )

    assert len(tokenized.labels) == len(
        tokenized.input_ids
    )


def test_tokenize_dataset() -> None:
    """
    Tokenize multiple samples.
    """

    pipeline = build_pipeline()

    dataset = [
        build_sample(),
        build_sample(),
        build_sample(),
    ]

    tokenized = pipeline.tokenize_dataset(dataset)

    assert len(tokenized) == 3

    for sample in tokenized:
        assert sample.sequence_length > 0


def test_create_batch() -> None:
    """
    Create a padded training batch.
    """

    pipeline = build_pipeline()

    dataset = [
        build_sample(),
        build_sample(),
    ]

    batch = pipeline.create_batch(dataset)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[0] == 2