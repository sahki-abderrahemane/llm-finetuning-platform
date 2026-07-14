"""
End-to-end QLoRA training demonstration.

Loads a model in 4-bit, prepares it for k-bit training,
injects LoRA adapters, and performs a minimal supervised
fine-tuning run.
"""

from __future__ import annotations

from pathlib import Path

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.lora.config import (
    LoRAConfig,
)
from mentorai_finetuning.lora.loader import (
    LoRALoader,
)
from mentorai_finetuning.qlora.config import (
    QLoRAConfig,
)
from mentorai_finetuning.qlora.loader import (
    QLoRAModelLoader,
)
from mentorai_finetuning.qlora.preparation import (
    QLoRAPreparer,
)
from mentorai_finetuning.qlora.statistics import (
    QLoRAStatistics,
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
from mentorai_finetuning.training.trainer import (
    SFTTrainingEngine,
)
from mentorai_finetuning.training.trainer_factory import (
    TrainerFactory,
)

from mentorai_finetuning.common.device import (
    DeviceDetector,
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
                    content="Explain QLoRA.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "QLoRA combines 4-bit quantization with "
                        "LoRA adapters to efficiently fine-tune "
                        "large language models."
                    ),
                ),
            ],
        ),
        DatasetSample(
            id="demo-2",
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="Why use NF4?",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "NF4 is a 4-bit quantization format designed "
                        "for normally distributed neural network "
                        "weights and provides better accuracy than "
                        "standard FP4."
                    ),
                ),
            ],
        ),
    ]


def main() -> None:
    output_dir = Path(
        "models/demo-qlora-checkpoint"
    )

    device = DeviceDetector.detect()

    DeviceDetector.print_summary(device)

    training_config = TrainingConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir=output_dir,
    epochs=1,
    train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    fp16=device.fp16,
    bf16=device.bf16,
    logging_steps=1,
    save_steps=10,
)

    qlora_config = QLoRAConfig()

    lora_config = LoRAConfig()

    print("Loading quantized model...")

    loader = QLoRAModelLoader(
        training_config,
        qlora_config,
    )

    model, tokenizer = loader.load()

    print("Preparing model for k-bit training...")

    model = QLoRAPreparer.prepare(
        model,
    )

    print("Injecting LoRA adapters...")

    lora_loader = LoRALoader(
        lora_config,
    )

    model = lora_loader.apply(
        model,
    )

    print()

    LoRALoader.print_trainable_parameters(
        model,
    )

    QLoRAStatistics.print_summary(
        model,
    )

    print("Building dataset...")

    dataset = SFTDatasetBuilder().build(
        create_dataset()
    )

    print("Creating training arguments...")

    training_args = TrainingArgumentsFactory.create(
        training_config,
    )

    print("Creating trainer...")

    trainer = TrainerFactory.create(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        training_args=training_args,
        config=training_config,
        data_collator=None,
    )

    engine = SFTTrainingEngine(
        trainer,
    )

    print("Starting training...")

    engine.train()

    print("Saving adapter...")

    engine.save(
        output_dir,
    )

    print()

    print("QLoRA demo completed successfully.")


if __name__ == "__main__":
    main()