
"""
Dataset export utilities.

"""

from __future__ import annotations

import json
from pathlib import Path

from mentorai_finetuning.dataset.schema import DatasetSample


class DatasetExporter:
    """Exports processed datasets."""

    def export_json(
        self,
        samples: list[DatasetSample],
        output_path: str | Path,
    ) -> None:
        """
        Export the dataset as a JSON file.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                [sample.model_dump() for sample in samples],
                file,
                indent=2,
                ensure_ascii=False,
            )

    def export_jsonl(
        self,
        samples: list[DatasetSample],
        output_path: str | Path,
    ) -> None:
        """
        Export the dataset as a JSONL file.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for sample in samples:
                json.dump(
                    sample.model_dump(),
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")