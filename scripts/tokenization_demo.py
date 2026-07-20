"""
End-to-end demonstration of the MentorAI tokenization pipeline.
"""

from __future__ import annotations

from pprint import pprint

from mentorai_finetuning.common.config import get_settings
from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.prompts.formatter import PromptFormatter
from mentorai_finetuning.tokenization.collator import DataCollator
from mentorai_finetuning.tokenization.config import TokenizerConfig
from mentorai_finetuning.tokenization.dataset_tokenizer import DatasetTokenizer
from mentorai_finetuning.tokenization.packing import NoPackingStrategy
from mentorai_finetuning.tokenization.pipeline import TokenizationPipeline
from mentorai_finetuning.tokenization.tokenizer_factory import TokenizerFactory
from mentorai_finetuning.tokenization.tokenizer_wrapper import TokenizerWrapper


def build_pipeline() -> tuple[TokenizationPipeline, TokenizerWrapper]:
    config = TokenizerConfig(
        model_name=get_settings().MODEL_NAME,
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

    pipeline = TokenizationPipeline(
        tokenizer=dataset_tokenizer,
        packing_strategy=NoPackingStrategy(),
        collator=collator,
    )

    return pipeline, wrapper


def build_sample() -> DatasetSample:
    return DatasetSample(
        id="mentorai-demo",
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are MentorAI, an AI mentor specialized in AI and software engineering.",
            ),
            Message(
                role=MessageRole.USER,
                content="Explain LoRA in one paragraph.",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    "LoRA (Low-Rank Adaptation) is a parameter-efficient "
                    "fine-tuning technique that freezes the original model "
                    "weights and learns only a small set of trainable low-rank "
                    "matrices, dramatically reducing memory usage while "
                    "maintaining strong performance."
                ),
            ),
        ],
    )


def main() -> None:
    pipeline, wrapper = build_pipeline()

    sample = build_sample()

    print("=" * 80)
    print("RAW SAMPLE")
    print("=" * 80)
    pprint(sample.model_dump())

    print()

    print("=" * 80)
    print("FORMATTED PROMPT")
    print("=" * 80)

    prompt = wrapper.format_prompt(sample)

    print(prompt)

    print()

    print("=" * 80)
    print("TOKENIZED SAMPLE")
    print("=" * 80)

    tokenized = pipeline.tokenize_sample(sample)

    print(f"Sequence length : {tokenized.sequence_length}")
    print(f"Trainable tokens: {tokenized.trainable_tokens}")
    print(f"Ignored tokens  : {tokenized.ignored_tokens}")

    print()

    print("=" * 80)
    print("FIRST 40 TOKEN IDS")
    print("=" * 80)

    print(tokenized.input_ids[:40])

    print()

    print("=" * 80)
    print("DECODED TEXT")
    print("=" * 80)

    print(
        wrapper.decode(
            tokenized.input_ids,
        )
    )

    print()

    print("=" * 80)
    print("TRAINING BATCH")
    print("=" * 80)

    batch = pipeline.create_batch([sample, sample])

    for key, value in batch.items():
        print(f"{key}: {tuple(value.shape)}")

    print()

    print("=" * 80)
    print("TOKENIZATION PIPELINE VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()