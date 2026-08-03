"""
``mentorai-cli registry`` -- manage the local model registry.
"""

from __future__ import annotations

import argparse

from mentorai_finetuning.cli.typing import SubParsersAction


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``registry`` subcommand.
    """

    parser = subparsers.add_parser(
        "registry",
        help="Manage the local model registry (not yet implemented).",
    )

    parser.add_argument(
        "action",
        choices=["list", "register", "show", "remove"],
        help="Registry action.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id (required for register/show/remove).",
    )

    parser.set_defaults(handler=handler)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Placeholder for the registry command.
    """

    print("registry: not yet implemented")
    return 1
