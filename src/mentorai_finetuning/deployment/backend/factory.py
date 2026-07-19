"""
Inference backend factory.
"""

from __future__ import annotations

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mentorai_finetuning.deployment.backend.base import (
    BaseInferenceBackend,
)
from mentorai_finetuning.deployment.backend.transformers_backend import (
    TransformersBackend,
)
from mentorai_finetuning.deployment.backend.types import (
    BackendType,
)
from mentorai_finetuning.deployment.backend.vllm_backend import VLLMBackend
from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)


class BackendFactory:
    """
    Factory responsible for creating inference backends.
    """

    @staticmethod
    def create(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: DeploymentConfig,
    ) -> BaseInferenceBackend:
        """
        Create the configured inference backend.
        """

        match config.backend:

            case BackendType.TRANSFORMERS:
                return TransformersBackend(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                )

            case BackendType.VLLM:
                 return VLLMBackend(
                             config=config,
                             )
            case BackendType.OLLAMA:
                raise NotImplementedError(
                    "Ollama backend is not implemented yet."
                )

            case _:
                raise ValueError(
                    f"Unsupported backend: {config.backend}"
                )