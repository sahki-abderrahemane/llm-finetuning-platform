"""
End-to-end SFT training demonstration.

Runs a minimal supervised fine-tuning job using a small dataset.
"""

from __future__ import annotations

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
from mentorai_finetuning.common.config import (
    get_settings,
)

def create_dataset() -> list[DatasetSample]:
    return [
        DatasetSample(
            id="demo-1",
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are MentorAI.",
                ),
                Message(
                    role=MessageRole.USER,
                    content="What is supervised fine tuning?",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "Supervised fine tuning trains a language model "
                        "using examples containing instructions and "
                        "desired responses."
                    ),
                ),
            ],
        ),
        DatasetSample(
            id="demo-2",
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="What is LoRA?",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "LoRA is a parameter efficient fine tuning method "
                        "that trains additional low rank adapters."
                    ),
                ),
            ],
        ),
    ]


def main() -> None:
    output_dir = Path(
        "models/demo-sft-checkpoint"
    )

    config = TrainingConfig(
        model_name=get_settings().MODEL_NAME,
        output_dir=output_dir,
        epochs=1,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_steps=10,
    )

    print("Loading model...")

    loader = ModelLoader(config)

    model, tokenizer = loader.load()

    print("Building dataset...")

    dataset_builder = SFTDatasetBuilder()

    dataset = dataset_builder.build(
        create_dataset()
    )

    print("Creating training arguments...")

    training_args = TrainingArgumentsFactory.create(
        config,
    )

    print("Creating trainer...")

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

    print("Starting training...")

    engine.train()

    print("Saving checkpoint...")

    engine.save(
        output_dir,
    )

    print("SFT demo completed successfully.")


if __name__ == "__main__":
    main()