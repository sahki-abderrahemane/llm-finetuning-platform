"""
Demonstrates the deployment pipeline.
"""
from __future__ import annotations

import multiprocessing as mp

from mentorai_finetuning.common.config import get_settings



def main() -> None:

    from mentorai_finetuning.deployment.config import ( 
    DeploymentConfig,
)
    from mentorai_finetuning.deployment.pipeline import (
    DeploymentPipeline,
)


    config = DeploymentConfig(
        model_name=get_settings().MODEL_NAME,
        max_new_tokens=get_settings().MAX_NEW_TOKENS,
        temperature=get_settings().TEMPERATURE,
        top_p=get_settings().TOP_P,
    )

    print("=" * 80)
    print("MentorAI Deployment Demo")
    print("=" * 80)
    print()

    print("Loading deployment pipeline...")
    print()

    pipeline = DeploymentPipeline(
        config,
    )

    prompts = [
        "Explain QLoRA in simple terms.",
        "What are LoRA adapters?",
        "Why is quantization useful?",
    ]

    for prompt in prompts:
        print("-" * 80)
        print(f"Prompt: {prompt}")
        print()

        response = pipeline.generate(
            prompt,
        )

        print("Response:")
        print(response.response)
        print()

        print("Statistics")
        print(f"Model              : {response.model_name}")
        print(f"Prompt Tokens      : {response.prompt_tokens}")
        print(f"Completion Tokens  : {response.completion_tokens}")
        print(f"Total Tokens       : {response.total_tokens}")
        print(f"Generation Time    : {response.generation_time:.2f} sec")
        print()

    print("=" * 80)
    print("Deployment demo completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method(
        "spawn",
        force=True,
    )

    main()