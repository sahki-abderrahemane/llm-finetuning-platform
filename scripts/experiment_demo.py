"""
Runs multiple QLoRA experiments with different learning rates.
"""

from __future__ import annotations

from pathlib import Path

from mentorai_finetuning.common.config import get_settings
from mentorai_finetuning.common.device import DeviceDetector
from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)
from mentorai_finetuning.experiments.config import (
    ExperimentConfig,
)
from mentorai_finetuning.experiments.runner import (
    ExperimentRunner,
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
                    content="Why is NF4 used?",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "NF4 is a quantization data type designed "
                        "for normally distributed neural network "
                        "weights and provides better accuracy than "
                        "standard FP4."
                    ),
                ),
            ],
        ),
    ]


def run_experiment(
    learning_rate: float,
) -> None:

    print()
    print("=" * 80)
    print(f"Running Experiment (learning_rate={learning_rate})")
    print("=" * 80)

    device = DeviceDetector.detect()

    output_dir = Path(
        f"models/experiments/lr_{learning_rate:.0e}"
    )

    training_config = TrainingConfig(
        model_name=get_settings().MODEL_NAME,
        output_dir=output_dir,
        epochs=1,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        fp16=device.fp16,
        bf16=device.bf16,
        logging_steps=1,
        save_steps=10,
    )

    qlora_config = QLoRAConfig()
    lora_config = LoRAConfig()

    print("Loading model...")

    model, tokenizer = QLoRAModelLoader(
        training_config,
        qlora_config,
    ).load()

    print("Preparing model for k-bit training...")

    model = QLoRAPreparer.prepare(
        model,
    )

    print("Injecting LoRA adapters...")

    model = LoRALoader(
        lora_config,
    ).apply(
        model,
    )

    LoRALoader.print_trainable_parameters(
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

    experiment = ExperimentConfig(
        name=f"lr-{learning_rate:.0e}",
        description=(
            f"QLoRA learning-rate sweep "
            f"(lr={learning_rate})"
        ),
        model_name=training_config.model_name,
        dataset_name="demo_dataset",
        output_dir=output_dir,
        training=training_config,
        tags=[
            "demo",
            "qlora",
            "learning-rate-sweep",
            f"lr={learning_rate:.0e}",
        ],
    )

    runner = ExperimentRunner(
        config=experiment,
        trainer=engine,
    )

    result = runner.run()

    print()
    print("=" * 80)
    print("Experiment Result")
    print("=" * 80)
    print(f"Experiment : {result.experiment_name}")
    print(f"Status     : {result.status}")
    print(f"Duration   : {result.duration_seconds:.2f} sec")
    print(f"Checkpoint : {result.checkpoint_dir}")
    print()


def main() -> None:

    device = DeviceDetector.detect()

    DeviceDetector.print_summary(
        device,
    )

    learning_rates = [
        1e-5,
        2e-5,
        5e-5,
    ]

    for learning_rate in learning_rates:
        run_experiment(
            learning_rate,
        )

    print("=" * 80)
    print("All experiments completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()