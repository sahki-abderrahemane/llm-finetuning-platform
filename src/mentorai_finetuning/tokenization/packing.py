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
    Placeholder for constant-length sequence packing.

    This strategy will be implemented later when benchmarking
    different packing algorithms.
    """

    def pack(
        self,
        samples: list[TokenizedSample],
    ) -> list[TokenizedSample]:
        raise NotImplementedError(
            "Constant-length packing will be implemented "
            "during the optimization phase."
        )


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
    ) -> BasePackingStrategy:
        """
        Create a packing strategy.
        """

        return cls._STRATEGIES[strategy]()