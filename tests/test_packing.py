"""
Tests for constant-length sequence packing.
"""

from __future__ import annotations

from mentorai_finetuning.tokenization.packing import (
    ConstantLengthPackingStrategy,
    PackingFactory,
    PackingStrategy,
)
from mentorai_finetuning.tokenization.schema import TokenizedSample


def make_sample(token_count: int) -> TokenizedSample:
    """Create a tokenized sample with all-trainable labels."""

    input_ids = list(range(token_count))

    return TokenizedSample(
        input_ids=input_ids,
        attention_mask=[1] * token_count,
        labels=input_ids,
        prompt=None,
        sequence_length=token_count,
    )


def test_constant_length_packing_creates_fixed_chunks():
    """Packed sequences must all have the configured length."""

    strategy = ConstantLengthPackingStrategy(
        sequence_length=8,
        eos_token_id=99,
        pad_token_id=0,
    )

    packed = strategy.pack(
        [
            make_sample(5),
            make_sample(5),
            make_sample(5),
        ]
    )

    assert len(packed) == 3

    for sample in packed:
        assert sample.sequence_length == 8

        assert len(sample.input_ids) == 8

        assert len(sample.attention_mask) == 8

        assert len(sample.labels) == 8


def test_packing_preserves_assistant_labels():
    """Labels (including -100 masking) must survive packing."""

    strategy = ConstantLengthPackingStrategy(
        sequence_length=10,
        eos_token_id=99,
        pad_token_id=0,
    )

    first = TokenizedSample(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        labels=[-100, -100, 3],
        prompt=None,
        sequence_length=3,
    )

    second = TokenizedSample(
        input_ids=[4, 5],
        attention_mask=[1, 1],
        labels=[-100, 5],
        prompt=None,
        sequence_length=2,
    )

    packed = strategy.pack([first, second])

    assert len(packed) == 1

    sample = packed[0]

    assert sample.labels[0] == -100
    assert sample.labels[1] == -100
    assert sample.labels[2] == 3

    assert sample.labels[3] == -100

    assert sample.labels[4] == -100
    assert sample.labels[5] == 5


def test_packing_pads_last_chunk():
    """The final chunk must be padded with the pad token."""

    strategy = ConstantLengthPackingStrategy(
        sequence_length=6,
        eos_token_id=99,
        pad_token_id=0,
    )

    packed = strategy.pack([make_sample(4)])

    assert len(packed) == 1

    sample = packed[0]

    assert sample.sequence_length == 6

    assert sample.input_ids[-2:] == [0, 0]

    assert sample.attention_mask[-2:] == [0, 0]

    assert sample.labels[-2:] == [-100, -100]


def test_empty_input_returns_empty():
    """Packing an empty list must return an empty list."""

    strategy = ConstantLengthPackingStrategy(
        sequence_length=8,
        eos_token_id=99,
        pad_token_id=0,
    )

    assert strategy.pack([]) == []


def test_factory_creates_constant_length_strategy():
    """PackingFactory must require the extra arguments."""

    strategy = PackingFactory.create(
        PackingStrategy.CONSTANT_LENGTH,
        sequence_length=8,
        eos_token_id=99,
        pad_token_id=0,
    )

    assert isinstance(strategy, ConstantLengthPackingStrategy)

    try:
        PackingFactory.create(PackingStrategy.CONSTANT_LENGTH)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError when arguments are missing."
        )
