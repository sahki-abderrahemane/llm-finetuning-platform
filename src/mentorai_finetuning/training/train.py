"""
Main entry point for supervised fine-tuning.

Connects the dataset engineering, tokenization and training pipelines
into a single command-line interface.

Example:
    python -m mentorai_finetuning.training.train \
        --dataset data/raw/train.jsonl \
        --adapter alpaca \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --method sft
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.common.seed import set_seed
from mentorai_finetuning.dataset.adapters.alpaca import AlpacaAdapter
from mentorai_finetuning.dataset.adapters.base import BaseDatasetAdapter
from mentorai_finetuning.dataset.adapters.chatml import ChatMLAdapter
from mentorai_finetuning.dataset.adapters.openai import OpenAIAdapter
from mentorai_finetuning.dataset.adapters.sharegpt import ShareGPTAdapter
from mentorai_finetuning.dataset.pipeline import DatasetPipeline
from mentorai_finetuning.lora.config import LoRAConfig
from mentorai_finetuning.prompts.registry import PromptFormatterRegistry
from mentorai_finetuning.qlora.config import QLoRAConfig
from mentorai_finetuning.qlora.trainer import QLoRATrainer
from mentorai_finetuning.tokenization.collator import DataCollator
from mentorai_finetuning.tokenization.config import TokenizerConfig
from mentorai_finetuning.tokenization.dataset_tokenizer import DatasetTokenizer
from mentorai_finetuning.tokenization.packing import (
    ConstantLengthPackingStrategy,
    NoPackingStrategy,
    PackingStrategy,
)
from mentorai_finetuning.tokenization.pipeline import TokenizationPipeline
from mentorai_finetuning.tokenization.tokenizer_factory import TokenizerFactory
from mentorai_finetuning.tokenization.tokenizer_wrapper import TokenizerWrapper
from mentorai_finetuning.training.arguments import TrainingArgumentsFactory
from mentorai_finetuning.training.config import TrainingConfig
from mentorai_finetuning.training.dataset_builder import SFTDatasetBuilder
from mentorai_finetuning.training.model_loader import ModelLoader
from mentorai_finetuning.training.trainer import SFTTrainingEngine
from mentorai_finetuning.training.trainer_factory import TrainerFactory

ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {
    "alpaca": AlpacaAdapter,
    "chatml": ChatMLAdapter,
    "openai": OpenAIAdapter,
    "sharegpt": ShareGPTAdapter,
}

METHODS = ("sft", "qlora")


@dataclass
class TokenizedData:
    """
    Pre-tokenized training data plus the tokenizer that produced it.
    """

    tokenizer: PreTrainedTokenizerBase
    dataset: Dataset
    collator: DataCollator
    eval_dataset: Dataset | None = None


def build_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Build the training CLI argument parser."""

    if parser is None:
        parser = argparse.ArgumentParser(
            description="Fine-tune a language model on MentorAI data.",
        )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to a raw JSON or JSONL dataset.",
    )

    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=None,
        help="Optional path to a raw validation JSON or JSONL dataset.",
    )

    parser.add_argument(
        "--adapter",
        choices=sorted(ADAPTERS),
        default="alpaca",
        help="Dataset format adapter.",
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        default="sft",
        help="Training method.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory where the trained model is saved.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size.",
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Per-device evaluation batch size.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision training.",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision training.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Peak learning rate.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens.",
    )

    parser.add_argument(
        "--packing",
        choices=[strategy.value for strategy in PackingStrategy],
        default=PackingStrategy.NONE.value,
        help="Sequence packing strategy.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank.",
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )

    return parser


def tokenize_dataset(
    config: TrainingConfig,
    adapter: BaseDatasetAdapter,
    packing: PackingStrategy,
    validation_dataset: Path | None = None,
) -> TokenizedData:
    """
    Run the dataset and tokenization pipelines.

    Loads the raw dataset, converts it into canonical samples, applies
    assistant-turn label masking and returns a pre-tokenized Hugging Face
    dataset.
    """

    samples = DatasetPipeline().run(
        path=config.train_dataset,
        adapter=adapter,
    )

    tokenizer_config = TokenizerConfig(
        model_name=config.model_name,
        max_length=config.max_sequence_length,
    )

    tokenizer = TokenizerFactory.create(tokenizer_config)

    formatter = PromptFormatterRegistry.create(
        tokenizer,
        model_type=config.model_name,
    )

    wrapper = TokenizerWrapper(
        tokenizer=tokenizer,
        formatter=formatter,
        config=tokenizer_config,
    )

    if packing == PackingStrategy.CONSTANT_LENGTH:
        packing_strategy = ConstantLengthPackingStrategy(
            sequence_length=config.max_sequence_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        packing_strategy = NoPackingStrategy()

    dataset_tokenizer = DatasetTokenizer(
        wrapper,
        mask_non_assistant=True,
    )

    pipeline = TokenizationPipeline(
        tokenizer=dataset_tokenizer,
        packing_strategy=packing_strategy,
        collator=DataCollator(tokenizer),
    )

    tokenized = pipeline.tokenize_dataset(samples)

    dataset = SFTDatasetBuilder().build_from_tokenized(tokenized)

    eval_dataset: Dataset | None = None

    if validation_dataset is not None:
        validation_samples = DatasetPipeline().run(
            path=validation_dataset,
            adapter=adapter,
        )

        validation_tokenized = pipeline.tokenize_dataset(
            validation_samples,
        )

        eval_dataset = SFTDatasetBuilder().build_from_tokenized(
            validation_tokenized,
        )

    return TokenizedData(
        tokenizer=tokenizer,
        dataset=dataset,
        collator=DataCollator(tokenizer),
        eval_dataset=eval_dataset,
    )


def build_trainer(
    config: TrainingConfig,
    method: str,
    data: TokenizedData,
    lora_config: LoRAConfig,
) -> SFTTrainingEngine:
    """
    Build the training engine for the requested method.
    """

    training_args = TrainingArgumentsFactory.create(config)

    if method == "qlora":
        trainer = QLoRATrainer(
            training_config=config,
            qlora_config=QLoRAConfig(),
            lora_config=lora_config,
        ).create_trainer(
            train_dataset=data.dataset,
            tokenizer=data.tokenizer,
            eval_dataset=data.eval_dataset,
        )
    else:
        model = ModelLoader(config).load_model()

        trainer = TrainerFactory.create(
            model=model,
            tokenizer=data.tokenizer,
            train_dataset=data.dataset,
            eval_dataset=data.eval_dataset,
            training_args=training_args,
            config=config,
            data_collator=data.collator,
        )

    return SFTTrainingEngine(trainer)


def main(
    args: argparse.Namespace | None = None,
) -> None:
    """Execute supervised fine-tuning."""

    if args is None:
        args = build_parser().parse_args()

    set_seed(args.seed)

    config = TrainingConfig(
        model_name=args.model,
        train_dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        max_sequence_length=args.max_length,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    packing = PackingStrategy(args.packing)

    data = tokenize_dataset(
        config,
        adapter=ADAPTERS[args.adapter](),
        packing=packing,
        validation_dataset=args.val_dataset,
    )

    engine = build_trainer(
        config,
        method=args.method,
        data=data,
        lora_config=lora_config,
    )

    engine.train()

    engine.save(config.output_dir)


if __name__ == "__main__":
    main()
