from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from mentorai_finetuning.deployment.inference import (
    InferenceRequest,
)


class BaseInferenceBackend(ABC):
    """
    Base interface implemented by every inference backend.
    """

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
    ) -> str:
        """
        Generate a response for an inference request.
        """
        raise NotImplementedError