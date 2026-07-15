"""
Experiment tracking utilities.
"""

from __future__ import annotations

import json
from pathlib import Path

from mentorai_finetuning.experiments.config import (
    ExperimentConfig,
)
from mentorai_finetuning.experiments.results import (
    ExperimentResult,
)


class ExperimentTracker:
    """
    Tracks experiment artifacts on disk.
    """

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        self.config = config

        self.root = (
            config.output_dir
            / config.experiment_id
        )

        self.checkpoints = (
            self.root / "checkpoints"
        )

        self.logs = (
            self.root / "logs"
        )

        self.metrics = (
            self.root / "metrics"
        )

    def initialize(self) -> None:
        """
        Create the experiment directory structure.
        """

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints.mkdir(
            exist_ok=True,
        )

        self.logs.mkdir(
            exist_ok=True,
        )

        self.metrics.mkdir(
            exist_ok=True,
        )

    def save_config(self) -> None:
        """
        Save the experiment configuration.
        """

        config_path = (
            self.root / "config.json"
        )

        config_path.write_text(
            self.config.model_dump_json(
                indent=4,
            ),
            encoding="utf-8",
        )

    def metrics_file(self) -> Path:
        """
        Return the metrics file path.
        """

        return (
            self.metrics
            / "metrics.json"
        )

    def save_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        """
        Save experiment metrics.
        """

        with self.metrics_file().open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

    def result_file(self) -> Path:
        """
        Return the experiment result file path.
        """

        return (
            self.root
            / "result.json"
        )

    def save_result(
        self,
        result: ExperimentResult,
    ) -> None:
        """
        Save the final experiment result.
        """

        self.result_file().write_text(
            result.model_dump_json(
                indent=4,
            ),
            encoding="utf-8",
        )

    def checkpoint_dir(self) -> Path:
        """
        Return the checkpoint directory.
        """

        return self.checkpoints