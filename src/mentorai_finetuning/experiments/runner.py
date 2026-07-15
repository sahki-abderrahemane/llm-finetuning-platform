"""
Experiment runner.

Coordinates experiment execution.
"""

from __future__ import annotations

from datetime import datetime

from mentorai_finetuning.experiments.config import (
    ExperimentConfig,
)
from mentorai_finetuning.experiments.results import (
    ExperimentResult,
)
from mentorai_finetuning.experiments.tracker import (
    ExperimentTracker,
)
from mentorai_finetuning.training.trainer import (
    SFTTrainingEngine,
)


class ExperimentRunner:
    """
    Executes a complete experiment.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        trainer: SFTTrainingEngine,
    ) -> None:
        self.config = config
        self.trainer = trainer

        self.tracker = ExperimentTracker(
            config,
        )

    def run(
        self,
    ) -> ExperimentResult:

        started = datetime.utcnow()

        print("=" * 60)
        print(f"Experiment : {self.config.name}")
        print(f"ID         : {self.config.experiment_id}")
        print("=" * 60)

        self.tracker.initialize()

        self.tracker.save_config()

        self.trainer.train()

        self.trainer.save(
            self.tracker.checkpoint_dir(),
        )

        finished = datetime.utcnow()

        result = ExperimentResult(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.name,
            status="SUCCESS",
            started_at=started,
            finished_at=finished,
            duration_seconds=(
                finished - started
            ).total_seconds(),
            checkpoint_dir=self.tracker.checkpoint_dir(),
            metrics={},
        )

        self.tracker.save_metrics(
            result.metrics,
        )

        self.tracker.save_result(
            result,
        )

        return result