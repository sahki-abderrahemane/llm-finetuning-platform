"""
``mentorai-cli eval`` -- evaluate predictions against references.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mentorai_finetuning.cli.typing import SubParsersAction
from mentorai_finetuning.common.config import get_settings
from mentorai_finetuning.evaluation.comparison import (
    EvaluationComparison,
)
from mentorai_finetuning.evaluation.evaluator import (
    ModelEvaluator,
)
from mentorai_finetuning.evaluation.metrics.base import (
    EvaluationMetric,
)
from mentorai_finetuning.evaluation.metrics.bert_score import (
    BERTScoreMetric,
)
from mentorai_finetuning.evaluation.metrics.bleu import (
    BLEUMetric,
)
from mentorai_finetuning.evaluation.metrics.judge import (
    LLMJudgeMetric,
)
from mentorai_finetuning.evaluation.metrics.rouge import (
    ROUGEMetric,
)

_METRICS: dict[str, Callable[[], EvaluationMetric]] = {
    "bleu": BLEUMetric,
    "rouge": ROUGEMetric,
    "bert": BERTScoreMetric,
    "llm_judge": LLMJudgeMetric,
}


def build_subparser(
    subparsers: SubParsersAction,
) -> None:
    """
    Register the ``eval`` subcommand.
    """

    parser = subparsers.add_parser(
        "eval",
        help="Evaluate predictions against reference answers.",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="models/qwen-7b-qlora",
        help="Path to the fine-tuned model directory.",
    )

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help=(
            "JSONL file where each line is {\"prediction\": \"...\", "
            "\"reference\": \"...\"}."
        ),
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated metrics: bleu, rouge, bert, llm_judge "
            "(default: all)."
        ),
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path for the JSON report (default: reports/eval-<timestamp>.json).",
    )

    parser.set_defaults(handler=handler)


def _load_predictions(
    path: Path,
) -> tuple[list[str], list[str]]:
    """
    Read prediction/reference pairs from a JSONL file.
    """

    predictions: list[str] = []
    references: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            predictions.append(str(record["prediction"]))
            references.append(str(record["reference"]))

    return predictions, references


def _select_metrics(
    selection: str,
) -> list[EvaluationMetric]:
    """
    Build the requested metric instances.
    """

    if selection == "all":
        names = ["bleu", "rouge", "bert"]
        if get_settings().OPENAI_API_KEY:
            names.append("llm_judge")
    else:
        names = [name.strip() for name in selection.split(",")]

    unknown = [name for name in names if name not in _METRICS]
    if unknown:
        raise SystemExit(
            f"eval: unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_METRICS))}."
        )

    return [_METRICS[name]() for name in names]


def handler(
    args: argparse.Namespace,
) -> int:
    """
    Evaluate predictions against references.
    """

    predictions_path = Path(args.predictions)
    predictions, references = _load_predictions(predictions_path)

    metrics = _select_metrics(args.metrics)

    evaluator = ModelEvaluator(metrics)
    report = evaluator.evaluate(
        predictions,
        references,
        model_name=args.path,
        dataset_name=predictions_path.stem,
    )

    comparison = EvaluationComparison([report])
    summary = comparison.summary()

    for row in summary:
        print(json.dumps(row, indent=2))

    report_path = Path(
        args.report
        or f"reports/eval-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_json(), encoding="utf-8")

    print(f"\nReport written to: {report_path}")

    return 0
