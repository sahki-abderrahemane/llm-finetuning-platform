"""
End-to-end evaluation demonstration.
"""

from __future__ import annotations

from mentorai_finetuning.evaluation.comparison import (
    EvaluationComparison,
)
from mentorai_finetuning.evaluation.evaluator import (
    ModelEvaluator,
)
from mentorai_finetuning.evaluation.metrics.bert_score import (
    BERTScoreMetric,
)
from mentorai_finetuning.evaluation.metrics.bleu import (
    BLEUMetric,
)
from mentorai_finetuning.evaluation.metrics.rouge import (
    ROUGEMetric,
)


def main() -> None:

    predictions_a = [
        "QLoRA fine-tunes large language models using low-rank adapters.",
        "NF4 is a quantization data type optimized for neural network weights.",
    ]

    predictions_b = [
        "QLoRA trains adapter matrices while keeping base weights frozen.",
        "NF4 is a four-bit quantization format.",
    ]

    references = [
        "QLoRA fine-tunes large language models using LoRA adapters.",
        "NF4 is a four-bit quantization format designed for neural network weights.",
    ]

    evaluator = ModelEvaluator(
        metrics=[
            BLEUMetric(),
            ROUGEMetric(),
            BERTScoreMetric(),
        ],
    )

    report_a = evaluator.evaluate(
        predictions=predictions_a,
        references=references,
        model_name="Qwen2.5-0.5B",
        dataset_name="MentorAI Demo",
    )

    report_b = evaluator.evaluate(
        predictions=predictions_b,
        references=references,
        model_name="Qwen2.5-1.5B",
        dataset_name="MentorAI Demo",
    )

    comparison = EvaluationComparison(
        [
            report_a,
            report_b,
        ],
    )

    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    for row in comparison.summary():

        print()

        for key, value in row.items():
            print(f"{key:15}: {value}")

    print()

    best = comparison.best_model(
        "BERTScore",
    )

    print("=" * 80)
    print("Best Model")
    print("=" * 80)
    print(best.model_name)


if __name__ == "__main__":
    main()