"""
General-purpose I/O utilities.

Provides atomic JSON serialization and simple version helpers used
across the framework.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class VersionedUtils:
    """
    JSON and version helpers.
    """

    @staticmethod
    def save_json(
        path: Path | str,
        data: Any,
        indent: int = 4,
        atomic: bool = True,
    ) -> Path:
        """
        Serialize ``data`` to a JSON file.

        When ``atomic`` is True the file is written to a temporary
        location first and then moved into place, preventing partial
        writes on failure.
        """

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
        )

        if not atomic:
            destination.write_text(
                serialized,
                encoding="utf-8",
            )

            return destination

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            delete=False,
        ) as file:
            file.write(serialized)
            temp_path = file.name

        os.replace(
            temp_path,
            destination,
        )

        return destination

    @staticmethod
    def load_json(
        path: Path | str,
    ) -> Any:
        """
        Load and deserialize a JSON file.
        """

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def is_valid_version(
        version: str,
    ) -> bool:
        """
        Return True if ``version`` matches a ``major.minor.patch``
        format.
        """

        return bool(
            _VERSION_PATTERN.match(version)
        )

    def load(
        self,
        path: str,
    ) -> Any:
        """
        Load a JSON file.
        """

        return self.load_json(path)
