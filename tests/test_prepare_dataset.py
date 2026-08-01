"""
Tests for the Hugging Face dataset preparation script.
"""

from __future__ import annotations

import json

import pytest
from prepare_hf_dataset import apply_column_map, parse_column_map, write_jsonl

from mentorai_finetuning.dataset.adapters.alpaca import AlpacaAdapter


def test_parse_column_map_empty():
    """An empty mapping must parse to an empty dict."""

    assert parse_column_map("") == {}
    assert parse_column_map("  ") == {}


def test_parse_column_map_entries():
    """Entries must be split on '=' and ','."""

    mapping = parse_column_map(
        "instruction=instruction,input=context,output=response"
    )

    assert mapping == {
        "instruction": "instruction",
        "input": "context",
        "output": "response",
    }


def test_apply_column_map_identity_when_empty():
    """Samples must pass through unchanged without a mapping."""

    sample = {"instruction": "hi", "output": "yo"}

    assert apply_column_map([sample], {}) == [sample]


def test_apply_column_map_renames_keys():
    """Mapped keys must be renamed and unmapped keys preserved."""

    samples = [
        {"instruction": "hello", "input": "world", "output": "yo"}
    ]

    renamed = apply_column_map(
        samples,
        {"input": "context"},
    )

    assert renamed == [
        {"instruction": "hello", "context": "world", "output": "yo"}
    ]


def test_apply_column_map_does_not_fail_on_missing_key():
    """Mapping to an absent HF column must be a no-op."""

    samples = [{"output": "yo"}]

    renamed = apply_column_map(
        samples,
        {"missing": "input"},
    )

    assert renamed == [{"output": "yo"}]


def test_write_jsonl_one_object_per_line(tmp_path):
    """Records must be written as newline-delimited JSON."""

    path = tmp_path / "train.jsonl"

    write_jsonl(
        path,
        [{"id": "sample_0", "text": "a"}, {"id": "sample_1", "text": "b"}],
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2

    assert json.loads(lines[0]) == {"id": "sample_0", "text": "a"}
    assert json.loads(lines[1]) == {"id": "sample_1", "text": "b"}


def test_alpaca_record_from_prepared_shape_is_convertible():
    """Alpaca-style records written by the converter must load cleanly."""

    record = {
        "id": "sample_0000000",
        "instruction": "Explain gravity",
        "output": "Gravity pulls masses together.",
    }

    sample = AlpacaAdapter().convert(record, sample_id="0")

    assert sample.messages[0].role.name == "USER"
    assert sample.messages[-1].role.name == "ASSISTANT"


def test_alpaca_adapter_rejects_empty_instruction():
    """A missing or blank instruction must raise a clear error."""

    adapter = AlpacaAdapter()

    for sample in (
        {"instruction": "", "output": "yo"},
        {"instruction": "   ", "output": "yo"},
        {"output": "yo"},
    ):
        with pytest.raises(ValueError, match="empty 'instruction'"):
            adapter.convert(sample, sample_id="0")


def test_alpaca_adapter_rejects_empty_output():
    """A missing or blank output must raise a clear error."""

    adapter = AlpacaAdapter()

    for sample in (
        {"instruction": "hi", "output": ""},
        {"instruction": "hi", "output": "  "},
        {"instruction": "hi"},
    ):
        with pytest.raises(ValueError, match="empty 'output'"):
            adapter.convert(sample, sample_id="0")
