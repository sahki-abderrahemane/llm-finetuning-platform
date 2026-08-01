"""
Tests for the API security layer.
"""

from __future__ import annotations

from unittest import mock

from mentorai_finetuning.deployment.api.security import (
    ApiSecurity,
)


def make_request(token: str | None = None):
    """Create a fake request with an optional Authorization header."""

    request = mock.Mock()
    request.headers = {}

    if token is not None:
        request.headers["authorization"] = f"Bearer {token}"

    return request


def test_disabled_security_allows_all():
    """With no API key configured, all requests must pass."""

    security = ApiSecurity(api_key=None)

    assert security.enabled is False

    assert security.authenticate(make_request(token=None)) is True


def test_valid_token_accepted():
    """A matching bearer token must authenticate."""

    security = ApiSecurity(api_key="secret-key")

    assert security.enabled is True

    assert security.authenticate(
        make_request(token="secret-key")
    ) is True


def test_invalid_token_rejected():
    """A wrong token must be rejected."""

    security = ApiSecurity(api_key="secret-key")

    assert security.authenticate(
        make_request(token="wrong-key")
    ) is False


def test_missing_header_rejected():
    """A missing Authorization header must be rejected."""

    security = ApiSecurity(api_key="secret-key")

    assert security.authenticate(make_request(token=None)) is False


def test_non_bearer_scheme_rejected():
    """A non-bearer scheme must be rejected."""

    security = ApiSecurity(api_key="secret-key")

    request = mock.Mock()
    request.headers = {
        "authorization": "Basic abc123",
    }

    assert security.authenticate(request) is False
