"""
High-level deployment pipeline.
"""

from __future__ import annotations

import time

from mentorai_finetuning.deployment.backend.factory import (
    BackendFactory,
)
from mentorai_finetuning.deployment.backend.types import (
    BackendType,
)
from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)
from mentorai_finetuning.deployment.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceRole,
)
from mentorai_finetuning.deployment.loader import (
    DeploymentLoader,
)
from mentorai_finetuning.deployment.request_builder import (
    InferenceRequestBuilder,
)
from mentorai_finetuning.deployment.response import (
    GenerationResponse,
)


class DeploymentPipeline:
    """
    High-level deployment interface.

    Loads a model once and serves inference requests.
    """

    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:

        self.config = config

        self.model = None
        self.tokenizer = None

        #
        # vLLM manages model loading internally, and Ollama runs the
        # model on its own server. Loading a transformers model first
        # would be wasteful (and breaks CUDA multiprocessing for vLLM).
        #
        if config.backend in (BackendType.VLLM, BackendType.OLLAMA):

            self.model = None
            self.tokenizer = None
            self.request_builder = None

        else:

            self.model, self.tokenizer = (
                DeploymentLoader(
                    config,
                ).load()
            )

            self.request_builder = (
                InferenceRequestBuilder(
                    self.tokenizer,
                )
            )

        self.backend = (
            BackendFactory.create(
                model=self.model,
                tokenizer=self.tokenizer,
                config=config,
            )
        )


    def generate(
        self,
        request: InferenceRequest,
    ) -> GenerationResponse:
        """
        Generate a response from a canonical inference request.
        """

        started = time.perf_counter()


        response = self.backend.generate(
            request,
        )


        finished = time.perf_counter()


        if self.request_builder:

            formatted_prompt = (
                self.request_builder.build(
                    request,
                )
            )

            prompt_tokens = len(
                self.tokenizer.encode(
                    formatted_prompt,
                )
            )

            completion_tokens = len(
                self.tokenizer.encode(
                    response,
                )
            )

        else:

            formatted_prompt = ""

            prompt_tokens = 0
            completion_tokens = 0


        return GenerationResponse(
            prompt=formatted_prompt,
            response=response,
            model_name=self.config.model_name,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens
                + completion_tokens
            ),
            generation_time=(
                finished - started
            ),
        )


    def generate_prompt(
        self,
        prompt: str,
    ) -> GenerationResponse:
        """
        Convenience wrapper for single-prompt inference.
        """

        request = InferenceRequest(
            messages=[
                InferenceMessage(
                    role=InferenceRole.USER,
                    content=prompt,
                )
            ]
        )

        return self.generate(
            request,
        )