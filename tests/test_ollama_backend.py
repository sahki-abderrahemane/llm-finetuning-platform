"""
Tests for the Ollama inference backend.
"""

from __future__ import annotations

from unittest import mock

from mentorai_finetuning.deployment.backend.ollama_backend import (
    OllamaBackend,
)
from mentorai_finetuning.deployment.config import DeploymentConfig
from mentorai_finetuning.deployment.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceRole,
)


def make_request() -> InferenceRequest:
    """Create a canonical inference request."""

    return InferenceRequest(
        messages=[
            InferenceMessage(
                role=InferenceRole.USER,
                content="Hello MentorAI.",
            )
        ]
    )


def make_response() -> dict:
    """Create a realistic Ollama chat response."""

    return {
        "model": "qwen2.5:0.5b",
        "message": {
            "role": "assistant",
            "content": "Hi there!",
        },
        "done": True,
    }


def test_ollama_backend_generates():
    """The backend must POST to /api/chat and return the content."""

    config = DeploymentConfig(
        backend="ollama",
        ollama_host="http://localhost:11434",
        ollama_model="qwen2.5:0.5b",
    )

    backend = OllamaBackend(config)

    with mock.patch(
        "httpx.post",
    ) as post:
        post.return_value = mock.Mock(
            json=lambda: make_response(),
            raise_for_status=lambda: None,
        )

        response = backend.generate(make_request())

    assert response == "Hi there!"

    payload = post.call_args.kwargs["json"]

    assert payload["model"] == "qwen2.5:0.5b"

    assert payload["stream"] is False

    assert payload["messages"][0]["role"] == "user"

    assert payload["messages"][0]["content"] == "Hello MentorAI."

    assert payload["options"]["temperature"] == config.temperature


def test_ollama_backend_defaults_model_to_model_name():
    """Without ollama_model, the base model name must be used."""

    config = DeploymentConfig(
        backend="ollama",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )

    backend = OllamaBackend(config)

    assert backend.model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_ollama_backend_raises_on_http_error():
    """HTTP failures must propagate."""

    config = DeploymentConfig(
        backend="ollama",
    )

    backend = OllamaBackend(config)

    with mock.patch(
        "httpx.post",
    ) as post:
        post.return_value = mock.Mock(
            raise_for_status=mock.Mock(
                side_effect=RuntimeError("connection refused"),
            ),
        )

        try:
            backend.generate(make_request())
        except RuntimeError as exc:
            assert "connection refused" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError.")
