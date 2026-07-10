"""
Dataset loading utilities.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetLoader:
    """Loads raw datasets from supported file formats."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    @classmethod
    def load(cls, path: str | Path) -> list[dict[str, Any]]:
        """
        Load a dataset from disk.

        Args:
            path: Path to a JSON or JSONL dataset.

        Returns:
            List of raw dataset samples.

        Raises:
            FileNotFoundError:
                If the file does not exist.

            ValueError:
                If the file extension is unsupported or the JSON structure is
                invalid.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format '{path.suffix}'. "
                f"Supported formats: {sorted(cls.SUPPORTED_EXTENSIONS)}"
            )

        if path.suffix == ".json":
            return cls._load_json(path)

        return cls._load_jsonl(path)

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        """Load a JSON dataset."""

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

        raise ValueError("JSON dataset must contain an object or a list.")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load a JSONL dataset."""

        samples: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}."
                    ) from exc

        return samples