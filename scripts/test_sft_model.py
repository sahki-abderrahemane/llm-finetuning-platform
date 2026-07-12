"""
Test loading the fine-tuned SFT checkpoint.

Allows passing a custom prompt from the command line.
"""

from __future__ import annotations

import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_PATH = "models/demo-sft-checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test MentorAI SFT model inference."
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt to send to the fine-tuned model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
    )

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()