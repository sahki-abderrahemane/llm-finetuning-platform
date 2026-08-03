"""
``mentorai-cli train`` -- supervised fine-tuning (SFT / LoRA / QLoRA).
"""

from __future__ import annotations

import argparse

from mentorai_finetuning.cli.typing import SubParsersAction
from mentorai_finetuning.training.train import (
    build_parser as build_train_parser,
)
from mentorai_finetuning.training.train import (
    main as run_train,
)


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``train`` subcommand.
    """

    parser = subparsers.add_parser(
        "train",
        help="Fine-tune a model using SFT, LoRA, or QLoRA.",
    )
    build_train_parser(parser)
    parser.set_defaults(handler=handler)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Run supervised fine-tuning.
    """

    run_train(args)
    return 0
