"""
API security helpers.

Validates bearer tokens against a configured API key.
"""

from __future__ import annotations

import secrets
from typing import Any


class ApiSecurity:
    """
    Bearer-token authentication for the serving API.

    When no API key is configured every request is allowed; when a key
    is configured, requests without a valid ``Authorization: Bearer
    <key>`` header are rejected.
    """

    def __init__(
        self,
        api_key: str | None,
    ) -> None:
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        """
        Whether authentication is enforced.
        """

        return self.api_key is not None

    def authenticate(
        self,
        request: Any,
    ) -> bool:
        """
        Validate the bearer token attached to a request.
        """

        if not self.enabled:
            return True

        authorization = request.headers.get(
            "authorization",
        )

        if authorization is None:
            return False

        scheme, _, token = authorization.partition(" ")

        if scheme.lower() != "bearer":
            return False

        assert self.api_key is not None

        return secrets.compare_digest(
            token,
            self.api_key,
        )
