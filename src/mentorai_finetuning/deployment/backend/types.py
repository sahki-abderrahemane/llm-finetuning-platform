"""
Supported inference backends.
"""

from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    """
    Supported deployment backends.
    """

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    OLLAMA = "ollama"