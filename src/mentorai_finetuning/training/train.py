"""
Main entry point for supervised fine-tuning.
"""

from __future__ import annotations

from mentorai_finetuning.training.arguments import (
    TrainingArgumentsFactory,
)
from mentorai_finetuning.training.config import TrainingConfig
from mentorai_finetuning.training.dataset_builder import (
    SFTDatasetBuilder,
)
from mentorai_finetuning.training.model_loader import (
    ModelLoader,
)
from mentorai_finetuning.training.trainer import (
    SFTTrainingEngine,
)
from mentorai_finetuning.training.trainer_factory import (
    TrainerFactory,
)


def load_dataset():
    """
    Temporary dataset loader.

    This will later be replaced by the complete dataset loading pipeline.
    """
    raise NotImplementedError(
        "Dataset loading will be connected after the SFT pipeline integration."
    )


def main() -> None:
    """
    Execute supervised fine-tuning.
    """

    config = TrainingConfig()

    model_loader = ModelLoader(config)

    model, tokenizer = model_loader.load()

    samples = load_dataset()

    dataset_builder = SFTDatasetBuilder()

    train_dataset = dataset_builder.build(samples)

    training_args = TrainingArgumentsFactory.create(
        config,
    )

    trainer = TrainerFactory.create(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        training_args=training_args,
        config=config,
        data_collator=None,
    )

    engine = SFTTrainingEngine(
        trainer,
    )

    engine.train()

    engine.save(
        config.output_dir,
    )


if __name__ == "__main__":
    main()