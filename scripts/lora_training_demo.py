"""
End-to-end LoRA fine-tuning demonstration.
"""

from __future__ import annotations

from pathlib import Path

from mentorai_finetuning.common.config import get_settings
from mentorai_finetuning.lora.config import LoRAConfig
from mentorai_finetuning.lora.loader import LoRALoader
from mentorai_finetuning.lora.merger import LoRAMerger
from mentorai_finetuning.lora.statistics import LoRAStatistics
from mentorai_finetuning.lora.utils import LoRAUtils

from mentorai_finetuning.training.config import TrainingConfig
from mentorai_finetuning.training.model_loader import ModelLoader

from mentorai_finetuning.dataset.schema import (
    DatasetSample,
    Message,
    MessageRole,
)


def create_demo_dataset() -> list[DatasetSample]:
    return [
        DatasetSample(
            id="sample-1",
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are MentorAI.",
                ),
                Message(
                    role=MessageRole.USER,
                    content="Explain LoRA.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        "LoRA is a parameter-efficient fine-tuning "
                        "technique that learns low-rank adapters while "
                        "keeping the original model frozen."
                    ),
                ),
            ],
        )
    ]


def main() -> None:
    print("=" * 70)
    print("MentorAI - LoRA Demonstration")
    print("=" * 70)

    training_config = TrainingConfig(
        model_name=get_settings().MODEL_NAME,
        output_dir=Path("models/lora-demo"),
    )

    lora_config = LoRAConfig()

    print("\nLoading base model...")

    loader = ModelLoader(training_config)

    model, tokenizer = loader.load()

    print("\nApplying LoRA adapters...")

    lora_loader = LoRALoader(lora_config)

    model = lora_loader.apply(model)

    print("\nPrinting trainable parameters...\n")

    lora_loader.print_trainable_parameters(model)

    print()

    LoRAStatistics.print_summary(model)

    print()

    LoRAUtils.print_summary(model)

    print()

    print("Dataset samples:", len(create_demo_dataset()))

    print()

    print(
        "LoRA successfully attached.\n"
        "Continue training using the existing SFT pipeline."
    )

    print()

    print("After training you would execute:")

    print()

    print(
        "merged_model = LoRAMerger.merge(model)"
    )

    print(
        'LoRAMerger.save('
    )

    print(
        '    merged_model,'
    )

    print(
        '    "models/lora-merged",'
    )

    print(
        ')'
    )

    print()

    print("Demo completed successfully.")


if __name__ == "__main__":
    main()