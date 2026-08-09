from pathlib import Path

import pytest

from a_memorix.paths import resolve_data_dir, resolve_runtime_path


def test_data_dir_is_explicit_and_resolved(tmp_path: Path) -> None:
    assert resolve_data_dir(tmp_path / "memory") == (tmp_path / "memory").resolve()


def test_runtime_path_cannot_escape_data_dir(tmp_path: Path) -> None:
    assert resolve_runtime_path(tmp_path, "imports/input.json") == (tmp_path / "imports/input.json").resolve()
    with pytest.raises(ValueError, match="escapes data directory"):
        resolve_runtime_path(tmp_path, "../outside.json")
