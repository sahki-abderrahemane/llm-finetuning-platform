"""
Transformers inference backend.
"""

from __future__ import annotations

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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


class TransformersBackend(
    BaseInferenceBackend,
):
    """
    Hugging Face Transformers inference backend.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: DeploymentConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.request_builder = (
            InferenceRequestBuilder(
                tokenizer,
            )
        )

    @torch.inference_mode()
    def generate(
        self,
        request: InferenceRequest,
    ) -> str:
        """
        Generate a response from an inference request.
        """

        formatted_prompt = (
            self.request_builder.build(
                request,
            )
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.model.device,
            )
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            use_cache=self.config.use_cache,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[-1]:,
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )