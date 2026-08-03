"""
Root parser and subcommand dispatch for the ``mentorai-cli`` command.
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

from mentorai_finetuning.cli import (
    chat,
    data,
    eval_cmd,
    infer,
    merge,
    registry,
    serve,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser with all subcommands."""

    parser = argparse.ArgumentParser(
        prog="mentorai-cli",
        description="MentorAI LLM fine-tuning toolkit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="command",
        required=True,
    )

    train.build_subparser(subparsers)
    serve.build_subparser(subparsers)
    data.build_subparser(subparsers)
    infer.build_subparser(subparsers)
    chat.build_subparser(subparsers)
    eval_cmd.build_subparser(subparsers)
    merge.build_subparser(subparsers)
    registry.build_subparser(subparsers)

    return parser


def dispatch(
    args: argparse.Namespace,
) -> int:
    """
    Route to the handler registered on the parsed subcommand.
    """

    handler = args.handler
    return cast(int, handler(args))


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for ``mentorai-cli``.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
