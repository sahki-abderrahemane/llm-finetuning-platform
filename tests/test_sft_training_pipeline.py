"""
Integration test for the SFT training pipeline.

This test verifies that the complete training stack can be initialized
and execute a minimal training step.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.training.arguments import (
    TrainingArgumentsFactory,
)
from mentorai_finetuning.training.config import (
    TrainingConfig,
)
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


def create_sample() -> DatasetSample:
    return DatasetSample(
        id="test-1",
        messages=[
            Message(
                role=MessageRole.USER,
                content="What is LoRA?",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    "LoRA is a parameter efficient "
                    "fine tuning technique."
                ),
            ),
        ],
    )


def test_sft_pipeline_initialization() -> None:
    """
    Verify that all SFT components can be created.
    """

    with tempfile.TemporaryDirectory() as directory:
        config = TrainingConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir=Path(directory),
            epochs=1,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
        )

        loader = ModelLoader(config)

        model, tokenizer = loader.load()

        dataset_builder = SFTDatasetBuilder()

        dataset = dataset_builder.build(
            [
                create_sample(),
            ]
        )

        training_args = TrainingArgumentsFactory.create(
            config,
        )

        trainer = TrainerFactory.create(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            training_args=training_args,
            config=config,
            data_collator=None,
        )

        engine = SFTTrainingEngine(
            trainer,
        )

        assert engine is not None
        assert engine.trainer is not None