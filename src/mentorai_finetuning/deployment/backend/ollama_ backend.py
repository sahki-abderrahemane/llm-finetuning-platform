"""
Ollama inference backend.
"""

from __future__ import annotations

from mentorai_finetuning.deployment.backend.base import (
    BaseInferenceBackend,
)
from mentorai_finetuning.deployment.inference import (
    InferenceRequest,
)


class OllamaBackend(
    BaseInferenceBackend,
):
    """
    Ollama inference backend.

    This backend will communicate with an Ollama
    server through its HTTP API.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the backend.

        Implementation will be added during
        Ollama integration.
        """

    def generate(
        self,
        request: InferenceRequest,
    ) -> str:
        """
        Generate a response.

        Not yet implemented.
        """

        raise NotImplementedError(
            "Ollama backend has not been implemented."
        )