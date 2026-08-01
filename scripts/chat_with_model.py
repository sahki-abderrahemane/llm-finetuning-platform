"""
Interactive REPL chat with a fine-tuned MentorAI model.

Usage:
    python scripts/chat_with_model.py --path models/qwen-0.5b-qlora-medium

Type prompts at the >>>  prompt. Exit with Ctrl-D or "quit".
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a fine-tuned MentorAI model."
    )

    parser.add_argument(
        "--path",
        type=str,
        default="models/demo-sft-checkpoint",
        help="Path to the fine-tuned model directory.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per reply.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
    )

    history: list[dict[str, str]] = []

    print(f"Loaded model from {args.path}")
    print("Chat ready. Ctrl-D or 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt or prompt.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": prompt})

        if tokenizer.chat_template is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        outputs = cast(Any, model).generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )

        response = cast(
            str,
            tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True,
            ),
        ).strip()

        history.append({"role": "assistant", "content": response})

        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
