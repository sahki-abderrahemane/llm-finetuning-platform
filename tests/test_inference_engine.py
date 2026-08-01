"""
Tests for the batched inference engine.
"""

from __future__ import annotations

import torch

from mentorai_finetuning.deployment.config import (
    DeploymentConfig,
)
from mentorai_finetuning.inference.engine import (
    InferenceEngine,
)


class FakeTokenizer:
    """Minimal tokenizer stub for the inference engine."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.call_count = 0
        self._last_batch_size = 0

    def __call__(self, texts, **kwargs):
        self.call_count += 1
        self._last_batch_size = len(texts)
        return {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    def batch_decode(self, sequences, **kwargs):
        return [
            "response"
            for _ in range(self._last_batch_size)
        ]


class FakeModel:
    """Minimal model stub for the inference engine."""

    def __init__(self) -> None:
        self.device = "cpu"

    def generate(self, **kwargs):
        return torch.ones(1, 8, dtype=torch.long)


def make_engine(batch_size: int = 2) -> InferenceEngine:
    """Create an engine wired to stub model/tokenizer."""

    config = DeploymentConfig(
        backend="transformers",
        max_new_tokens=16,
        temperature=0.5,
        top_p=0.9,
        top_k=10,
    )

    return InferenceEngine(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        config=config,
    )


def test_generate_batches_prompts():
    """Prompts must be processed in batches and returned."""

    engine = make_engine(batch_size=2)

    results = engine.generate(
        ["prompt-a", "prompt-b", "prompt-c"],
        batch_size=2,
    )

    assert results == ["response", "response", "response"]

    assert engine.tokenizer.call_count == 2


def test_generate_one():
    """generate_one() must return a single string."""

    engine = make_engine()

    result = engine.generate_one("hello")

    assert result == "response"


def test_generate_empty_batch():
    """An empty prompt list must produce no results."""

    engine = make_engine()

    assert engine.generate([]) == []
