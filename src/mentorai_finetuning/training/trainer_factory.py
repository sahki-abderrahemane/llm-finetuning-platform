"""
Factory for creating supervised fine-tuning trainers.
"""

from __future__ import annotations

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from mentorai_finetuning.training.config import TrainingConfig


class TrainerFactory:
    """
    Creates the SFT training engine.
    """

    @staticmethod
    def create(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        training_args: TrainingArguments,
        config: TrainingConfig,
        data_collator: object | None,
    ) -> SFTTrainer:
        """
        Create a TRL SFTTrainer instance.
        """

        sft_config = SFTConfig(
            output_dir=training_args.output_dir,

            num_train_epochs=config.epochs,

            learning_rate=config.learning_rate,

            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,

            gradient_accumulation_steps=config.gradient_accumulation_steps,

            logging_steps=config.logging_steps,
            save_steps=config.save_steps,

            max_length=config.max_sequence_length,

            fp16=config.fp16,
            bf16=config.bf16,

            report_to="none",
        )

        return SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )