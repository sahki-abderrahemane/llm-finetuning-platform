"""
Tests for OpenAI-compatible streaming responses.
"""

from __future__ import annotations

from unittest import mock

from fastapi.testclient import TestClient

from mentorai_finetuning.deployment.api.app import (
    app,
)
from mentorai_finetuning.deployment.api.dependencies import (
    get_pipeline,
)


def make_pipeline():
    """Create a fake pipeline returning a canned result."""

    result = mock.Mock()
    result.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    result.response = "Hello world!"
    result.finish_reason = "stop"
    result.prompt_tokens = 5
    result.completion_tokens = 3
    result.total_tokens = 8

    pipeline = mock.Mock()
    pipeline.generate.return_value = result
    pipeline.generate_prompt.return_value = result

    return pipeline


def test_chat_completions_streaming():
    """stream=True must return SSE chunks and [DONE]."""

    pipeline = make_pipeline()

    app.dependency_overrides[get_pipeline] = lambda: pipeline

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello",
                        }
                    ],
                    "stream": True,
                },
            )

        assert response.status_code == 200

        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        body = response.text

        assert "Hello" in body

        assert "data: [DONE]" in body

        assert 'chat.completion.chunk' in body
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


def test_completions_streaming():
    """The /v1/completions endpoint must also stream."""

    pipeline = make_pipeline()

    app.dependency_overrides[get_pipeline] = lambda: pipeline

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "prompt": "Hello",
                    "stream": True,
                },
            )

        assert response.status_code == 200

        assert "data: [DONE]" in response.text
    finally:
        app.dependency_overrides.pop(get_pipeline, None)
