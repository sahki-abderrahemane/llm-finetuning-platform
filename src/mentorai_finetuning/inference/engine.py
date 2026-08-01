"""
Batched inference engine.

Runs batched text generation against a Hugging Face model.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)


class InferenceEngine:
    """
    Batched text generation using a causal language model.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: DeploymentConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DeploymentConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config: DeploymentConfig | None = None,
        trust_remote_code: bool = False,
    ) -> InferenceEngine:
        """
        Load a model and tokenizer from a local directory.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Generate a response for every prompt.

        Prompts are grouped into batches to limit peak memory usage.
        """

        max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.config.max_new_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else self.config.temperature
        )

        top_p = (
            top_p
            if top_p is not None
            else self.config.top_p
        )

        top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        do_sample = (
            do_sample
            if do_sample is not None
            else self.config.do_sample
        )

        results: list[str] = []

        for start in range(0, len(prompts), batch_size):

            batch = prompts[start : start + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            inputs = {
                key: value.to(
                    self.model.device,
                )
                for key, value in inputs.items()
            }

            model = cast(
                Any,
                self.model,
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                use_cache=self.config.use_cache,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = outputs[
                :,
                inputs["input_ids"].shape[-1]:,
            ]

            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            results.extend(decoded)

        return results

    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Generate a response for a single prompt.
        """

        return self.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )[0]
