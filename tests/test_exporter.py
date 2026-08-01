"""
Tests for the model exporter.
"""

from __future__ import annotations

from unittest import mock

from mentorai_finetuning.deployment.export import (
    ModelExporter,
)


def make_model():
    """Create a fake model with save/push hooks."""

    return mock.Mock(
        save_pretrained=mock.Mock(),
        push_to_hub=mock.Mock(),
    )


def make_tokenizer():
    """Create a fake tokenizer."""

    return mock.Mock(
        save_pretrained=mock.Mock(),
        push_to_hub=mock.Mock(),
    )


def test_exporter_saves_locally(tmp_path):
    """export() must save the model and tokenizer."""

    model = make_model()
    tokenizer = make_tokenizer()

    exporter = ModelExporter()

    output_dir = tmp_path / "model"

    result = exporter.export(
        model=model,
        output_dir=output_dir,
        tokenizer=tokenizer,
    )

    assert result == output_dir

    model.save_pretrained.assert_called_once()

    tokenizer.save_pretrained.assert_called_once()

    model.push_to_hub.assert_not_called()


def test_exporter_pushes_to_hub(tmp_path):
    """export() must push to the hub when repo_id is given."""

    model = make_model()
    tokenizer = make_tokenizer()

    exporter = ModelExporter()

    exporter.export(
        model=model,
        output_dir=tmp_path / "model",
        tokenizer=tokenizer,
        repo_id="mentorai/demo",
    )

    model.push_to_hub.assert_called_once()

    tokenizer.push_to_hub.assert_called_once()
