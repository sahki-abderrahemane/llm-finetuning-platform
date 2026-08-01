"""
Tests for the versioned I/O utilities.
"""

from __future__ import annotations

import json

from mentorai_finetuning.utils.io import VersionedUtils


def test_save_and_load_json_round_trip(tmp_path):
    """Data written with save_json must reload identically."""

    path = tmp_path / "nested" / "data.json"

    VersionedUtils.save_json(
        path,
        {"a": [1, 2, 3], "b": "text"},
    )

    loaded = VersionedUtils.load_json(path)

    assert loaded == {"a": [1, 2, 3], "b": "text"}


def test_atomic_write_creates_valid_json(tmp_path):
    """Atomic writes must produce parseable JSON."""

    path = tmp_path / "data.json"

    VersionedUtils.save_json(
        path,
        {"ok": True},
        atomic=True,
    )

    with path.open("r", encoding="utf-8") as file:
        assert json.load(file) == {"ok": True}


def test_load_returns_same_as_load_json(tmp_path):
    """The load helper must delegate to load_json."""

    path = tmp_path / "data.json"

    VersionedUtils.save_json(path, {"x": 1})

    assert VersionedUtils().load(str(path)) == {"x": 1}


def test_is_valid_version():
    """Version strings must match major.minor.patch."""

    assert VersionedUtils.is_valid_version("1.2.3")
    assert VersionedUtils.is_valid_version("0.1.0")
    assert not VersionedUtils.is_valid_version("1.2")
    assert not VersionedUtils.is_valid_version("latest")
