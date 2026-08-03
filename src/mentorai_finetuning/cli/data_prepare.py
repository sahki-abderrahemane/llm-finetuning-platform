"""
``mentorai-cli data prepare`` -- download an HF dataset and write JSONL.
"""

from __future__ import annotations

import argparse

from mentorai_finetuning.cli.typing import SubParsersAction
from mentorai_finetuning.dataset.hf_prepare import (
    build_parser as build_prepare_parser,
)
from mentorai_finetuning.dataset.hf_prepare import (
    main as run_prepare,
)


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``prepare`` subcommand.
    """

    parser = subparsers.add_parser(
        "prepare",
        help="Download a Hugging Face dataset and split into train/validation JSONL.",
    )
    build_prepare_parser(parser)
    parser.set_defaults(handler=handler)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Run dataset preparation.
    """

    run_prepare(args)
    return 0
