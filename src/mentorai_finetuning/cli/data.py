"""
``mentorai-cli data`` -- dataset preparation utilities.
"""

from __future__ import annotations

import argparse
from typing import cast

from mentorai_finetuning.cli.data_prepare import (
    build_subparser as build_prepare_subparser,
)
from mentorai_finetuning.cli.typing import SubParsersAction


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``data`` subcommand group.
    """

    parser = subparsers.add_parser(
        "data",
        help="Dataset preparation utilities.",
    )

    data_subparsers = parser.add_subparsers(
        dest="data_command",
        metavar="data-command",
        required=True,
    )

    build_prepare_subparser(data_subparsers)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Dispatch to the nested data subcommand handler.
    """

    nested = args.handler
    return cast(int, nested(args))
