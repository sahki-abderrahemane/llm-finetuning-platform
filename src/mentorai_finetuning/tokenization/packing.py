"""
Packing strategies for tokenized samples.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from mentorai_finetuning.tokenization.schema import TokenizedSample


class PackingStrategy(Enum):
    """
    Supported packing strategies.
    """

    NONE = "none"
    CONSTANT_LENGTH = "constant_length"


class BasePackingStrategy(ABC):
    """
    Base interface for all packing strategies.
    """

    @abstractmethod
    def pack(
        self,
        samples: list[TokenizedSample],
    ) -> list[TokenizedSample]:
        """
        Pack a collection of tokenized samples.
        """
        raise NotImplementedError


class NoPackingStrategy(BasePackingStrategy):
    """
    Leaves every sample unchanged.

    This is the recommended strategy for supervised instruction tuning
    and QLoRA unless maximizing GPU throughput becomes a priority.
    """

    def pack(
        self,
        samples: list[TokenizedSample],
    ) -> list[TokenizedSample]:
        return samples


class ConstantLengthPackingStrategy(BasePackingStrategy):
    """
    Concatenates short samples into fixed-length training sequences.

    Samples are chained together in order, separated by the EOS token,
    and split into chunks of ``sequence_length`` tokens. The final chunk
    is padded with the pad token. Labels are concatenated verbatim so
    the ``-100`` assistant masking is preserved across sample
    boundaries.
    """

    def __init__(
        self,
        sequence_length: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def pack(
        self,
        samples: list[TokenizedSample],
    ) -> list[TokenizedSample]:
        """
        Pack samples into fixed-length sequences.
        """

        if not samples:
            return []

        input_ids: list[int] = []
        attention_mask: list[int] = []
        labels: list[int] = []

        for sample in samples:
            if input_ids:
                input_ids.append(self.eos_token_id)
                attention_mask.append(1)
                labels.append(-100)

            input_ids.extend(sample.input_ids)
            attention_mask.extend(sample.attention_mask)
            labels.extend(sample.labels)

        if not input_ids:
            return []

        chunks = [
            input_ids[index : index + self.sequence_length]
            for index in range(0, len(input_ids), self.sequence_length)
        ]

        packed: list[TokenizedSample] = []

        for offset, chunk in enumerate(chunks):
            start = offset * self.sequence_length
            end = start + len(chunk)

            chunk_labels = labels[start:end]
            chunk_mask = attention_mask[start:end]

            pad_length = self.sequence_length - len(chunk)

            if pad_length > 0:
                chunk = chunk + [self.pad_token_id] * pad_length
                chunk_labels = chunk_labels + [-100] * pad_length
                chunk_mask = chunk_mask + [0] * pad_length

            packed.append(
                TokenizedSample(
                    input_ids=chunk,
                    attention_mask=chunk_mask,
                    labels=chunk_labels,
                    prompt=None,
                    sequence_length=len(chunk),
                )
            )

        return packed


class PackingFactory:
    """
    Factory responsible for creating packing strategies.
    """

    _STRATEGIES: dict[
        PackingStrategy,
        type[BasePackingStrategy],
    ] = {
        PackingStrategy.NONE: NoPackingStrategy,
        PackingStrategy.CONSTANT_LENGTH: ConstantLengthPackingStrategy,
    }

    @classmethod
    def create(
        cls,
        strategy: PackingStrategy,
        *,
        sequence_length: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> BasePackingStrategy:
        """
        Create a packing strategy.

        Parameters
        ----------
        strategy:
            Packing strategy to instantiate.

        sequence_length, eos_token_id, pad_token_id:
            Required only for ``CONSTANT_LENGTH`` packing.
        """

        if strategy != PackingStrategy.CONSTANT_LENGTH:
            return cls._STRATEGIES[strategy]()

        if (
            sequence_length is None
            or eos_token_id is None
            or pad_token_id is None
        ):
            raise ValueError(
                "Constant-length packing requires sequence_length, "
                "eos_token_id and pad_token_id."
            )

        return cls._STRATEGIES[strategy](
            sequence_length=sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )