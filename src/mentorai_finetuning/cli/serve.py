"""
``mentorai-cli serve`` -- run the inference API server.
"""

from __future__ import annotations

import argparse
import sys

from mentorai_finetuning.cli.typing import SubParsersAction
from mentorai_finetuning.deployment.api.__main__ import (
    main as run_server,
)


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``serve`` subcommand.
    """

    parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI inference server.",
    )
    parser.set_defaults(handler=handler)


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Start the uvicorn server.
    """

    run_server()
    return 0


if __name__ == "__main__":
    sys.exit(handler(argparse.Namespace()))
