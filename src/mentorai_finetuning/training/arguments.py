"""
Utilities for converting the project's TrainingConfig into Hugging Face
TrainingArguments.
"""

from __future__ import annotations

from transformers import TrainingArguments

from mentorai_finetuning.training.config import TrainingConfig


class TrainingArgumentsFactory:
    """
    Factory responsible for creating Hugging Face TrainingArguments.
    """

    @staticmethod
    def create(
        config: TrainingConfig,
    ) -> TrainingArguments:
        """
        Convert a TrainingConfig into Hugging Face TrainingArguments.
        """

        return TrainingArguments(
            output_dir=str(config.output_dir),

            num_train_epochs=config.epochs,

            learning_rate=config.learning_rate,

            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,

            gradient_accumulation_steps=config.gradient_accumulation_steps,

            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,

            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.evaluation_steps,

            eval_strategy="steps",
            save_strategy="steps",

            seed=config.seed,

            fp16=config.fp16,
            bf16=config.bf16,

            remove_unused_columns=config.remove_unused_columns,

            report_to="none",
        )