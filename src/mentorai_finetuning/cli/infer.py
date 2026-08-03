"""
``mentorai-cli infer`` -- run one-off inference against a checkpoint.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from mentorai_finetuning.cli.typing import SubParsersAction


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``infer`` subcommand.
    """

    parser = subparsers.add_parser(
        "infer",
        help="Generate a response from a fine-tuned checkpoint.",
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt to send to the fine-tuned model.",
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
        default=100,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.set_defaults(handler=handler)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Run a single inference pass.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
    )

    messages = [
        {
            "role": "user",
            "content": args.prompt,
        }
    ]

    if tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = args.prompt

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

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True,
    )

    print("\n=== Response ===")
    print(response)

    return 0
