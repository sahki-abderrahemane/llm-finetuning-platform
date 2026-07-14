"""
QLoRA training pipeline.
"""

from __future__ import annotations

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from mentorai_finetuning.lora.config import LoRAConfig
from mentorai_finetuning.lora.loader import LoRALoader

from mentorai_finetuning.qlora.config import QLoRAConfig
from mentorai_finetuning.qlora.loader import QLoRAModelLoader
from mentorai_finetuning.qlora.preparation import QLoRAPreparer

from mentorai_finetuning.training.arguments import (
    TrainingArgumentsFactory,
)
from mentorai_finetuning.training.config import TrainingConfig
from mentorai_finetuning.training.trainer_factory import (
    TrainerFactory,
)


class QLoRATrainer:
    """
    High-level orchestration class for QLoRA training.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        qlora_config: QLoRAConfig,
        lora_config: LoRAConfig,
    ) -> None:

        self.training_config = training_config
        self.qlora_config = qlora_config
        self.lora_config = lora_config

    def create_trainer(
        self,
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """
        Build a complete QLoRA trainer.
        """

        loader = QLoRAModelLoader(
            self.training_config,
            self.qlora_config,
        )

        model, _ = loader.load()

        model = QLoRAPreparer.prepare(
            model,
        )

        lora_loader = LoRALoader(
            self.lora_config,
        )

        model = lora_loader.apply(
            model,
        )

        training_args = (
            TrainingArgumentsFactory.create(
                self.training_config,
            )
        )

        trainer = TrainerFactory.create(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            training_args=training_args,
            config=self.training_config,
            data_collator=None,
        )

        return trainer