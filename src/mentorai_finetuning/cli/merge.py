"""
``mentorai-cli merge`` -- merge a LoRA/QLoRA adapter into its base model.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from mentorai_finetuning.cli.typing import SubParsersAction
from mentorai_finetuning.common.config import get_settings
from mentorai_finetuning.lora.merger import LoRAMerger


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``merge`` subcommand.
    """

    parser = subparsers.add_parser(
        "merge",
        help="Merge a LoRA/QLoRA adapter into its base model.",
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default="models/qwen-7b-qlora",
        help="Path to the trained adapter directory.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model identifier or path (default: MODEL_NAME from .env).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/merged/qwen-7b",
        help="Output directory for the merged model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Dtype used to load the base model (default: fp16).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to load the base model for merging (default: cpu).",
    )

    parser.set_defaults(handler=handler)


def _resolve_dtype(
    dtype: str,
) -> torch.dtype:
    """
    Map a dtype name to a torch dtype.
    """

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Merge the LoRA adapter into a full-precision base model.
    """

    model_name = args.base_model or get_settings().MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_resolve_dtype(args.dtype),
        device_map=args.device,
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
    )

    LoRAMerger.merge_and_save(
        peft_model,
        args.output,
    )

    tokenizer.save_pretrained(
        args.output,
    )

    print(f"Merged model saved to: {args.output}")
    return 0
