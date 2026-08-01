"""
vLLM inference backend.
"""

from __future__ import annotations

import multiprocessing as mp

from mentorai_finetuning.deployment.backend.base import (
    BaseInferenceBackend,
)

from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)

from mentorai_finetuning.deployment.inference import (
    InferenceRequest,
)

from mentorai_finetuning.deployment.request_builder import (
    InferenceRequestBuilder,
)


class VLLMBackend(
    BaseInferenceBackend,
):
    """
    vLLM inference backend.
    """


    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:


        self.config = config


        #
        # vLLM is a heavy optional dependency. Import it lazily so the
        # package and the FastAPI app import cleanly on machines
        # without vLLM / a GPU installed.
        #
        from vllm import LLM  # type: ignore[import-not-found]

        #
        # Must happen before vLLM creates workers.
        #
        try:
            mp.set_start_method(
                "spawn",
                force=True,
            )

        except RuntimeError:
            pass


        self.llm = LLM(
            model=config.model_name,
            trust_remote_code=config.trust_remote_code,
            dtype=config.torch_dtype,
            enforce_eager=True,
            gpu_memory_utilization=(
                config.gpu_memory_utilization
            ),
        )


        self.request_builder = (
            InferenceRequestBuilder(
                self.llm.get_tokenizer(),
            )
        )


    def generate(
        self,
        request: InferenceRequest,
    ) -> str:
        """
        Generate text using vLLM.
        """


        prompt = (
            self.request_builder.build(
                request,
            )
        )

        from vllm import SamplingParams  # type: ignore[import-not-found]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_new_tokens,
            repetition_penalty=(
                self.config.repetition_penalty
            ),
            seed=self.config.seed,
        )


        outputs = self.llm.generate(
            [prompt],
            sampling_params,
        )


        return (
            outputs[0]
            .outputs[0]
            .text
        )