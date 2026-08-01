"""
Ollama inference backend.

Communicates with a running Ollama server through its HTTP API.
"""

from __future__ import annotations

from typing import Any

import httpx

from mentorai_finetuning.deployment.backend.base import (
    BaseInferenceBackend,
)
from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)
from mentorai_finetuning.deployment.inference import (
    InferenceRequest,
)


class OllamaBackend(BaseInferenceBackend):
    """
    Ollama inference backend.

    Delegates generation to a remote Ollama server, which manages its
    own model loading and templating.
    """

    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:
        self.config = config
        self.base_url = config.ollama_host.rstrip("/")
        self.model = config.ollama_model or config.model_name

    def generate(
        self,
        request: InferenceRequest,
    ) -> str:
        """
        Generate a response through the Ollama chat API.
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.max_new_tokens,
                "seed": self.config.seed,
            },
        }

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(600.0),
        )

        response.raise_for_status()

        data = response.json()

        message = self._extract_message(data)

        return message

    def _extract_message(
        self,
        data: dict[str, Any],
    ) -> str:
        """
        Extract the assistant message content from an Ollama response.
        """

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Ollama response is missing the 'message' field."
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(
                "Ollama response is missing message content."
            )

        return content
