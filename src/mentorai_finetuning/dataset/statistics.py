"""
Dataset statistics utilities.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from mentorai_finetuning.dataset.schema import DatasetSample, MessageRole


class DatasetStatistics(BaseModel):
    """Summary statistics for a processed dataset."""

    total_samples: int

    total_messages: int

    average_messages_per_sample: float

    role_distribution: dict[str, int]

    language_distribution: dict[str, int]

    source_distribution: dict[str, int]

    domain_distribution: dict[str, int]

    average_characters_per_message: float


class DatasetStatisticsGenerator:
    """Computes descriptive statistics for datasets."""

    def compute(
        self,
        samples: list[DatasetSample],
    ) -> DatasetStatistics:

        role_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()

        total_messages = 0
        total_characters = 0

        for sample in samples:

            if sample.metadata.language:
                language_counter[sample.metadata.language] += 1

            if sample.metadata.source:
                source_counter[sample.metadata.source] += 1

            if sample.metadata.domain:
                domain_counter[sample.metadata.domain] += 1

            total_messages += len(sample.messages)

            for message in sample.messages:

                role_counter[message.role.value] += 1

                total_characters += len(message.content)

        average_messages = (
            total_messages / len(samples)
            if samples
            else 0.0
        )

        average_characters = (
            total_characters / total_messages
            if total_messages
            else 0.0
        )

        return DatasetStatistics(
            total_samples=len(samples),
            total_messages=total_messages,
            average_messages_per_sample=average_messages,
            role_distribution=dict(role_counter),
            language_distribution=dict(language_counter),
            source_distribution=dict(source_counter),
            domain_distribution=dict(domain_counter),
            average_characters_per_message=average_characters,
        )

    def print_summary(
        self,
        statistics: DatasetStatistics,
    ) -> None:
        """Print a readable dataset summary."""

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Samples: {statistics.total_samples}")
        print(f"Messages: {statistics.total_messages}")

        print(
            f"Average Messages / Sample: "
            f"{statistics.average_messages_per_sample:.2f}"
        )

        print(
            f"Average Characters / Message: "
            f"{statistics.average_characters_per_message:.2f}"
        )

        print("\nRole Distribution")

        for role, count in statistics.role_distribution.items():
            print(f"  {role:<12} {count}")

        if statistics.language_distribution:

            print("\nLanguages")

            for language, count in statistics.language_distribution.items():
                print(f"  {language:<12} {count}")

        if statistics.source_distribution:

            print("\nSources")

            for source, count in statistics.source_distribution.items():
                print(f"  {source:<25} {count}")

        if statistics.domain_distribution:

            print("\nDomains")

            for domain, count in statistics.domain_distribution.items():
                print(f"  {domain:<20} {count}")

        print("=" * 60)