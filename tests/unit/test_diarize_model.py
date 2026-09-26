"""Unit tests for backend.diarize.model — local-only model resolution + pinned revision."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from backend.diarize import model
from backend.diarize.errors import DiarizationConfigError
from backend.diarize.model import DIARIZATION_REVISION, DIARIZATION_SHA256, resolve_model_dir


def _snapshot(root: Path) -> Path:
    for name in ("config.yaml", *DIARIZATION_SHA256):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    return root


def _never(_cache_dir: str | None) -> Path | None:
    raise AssertionError("cache must not be consulted")


def test_env_dir_wins_over_cache(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
    env = {"STT_DIARIZATION_MODEL_DIR": str(tmp_path)}
    assert resolve_model_dir(env, cached_snapshot=_never) == tmp_path


def test_env_dir_without_config_names_the_variable(tmp_path: Path) -> None:
    with pytest.raises(DiarizationConfigError, match="STT_DIARIZATION_MODEL_DIR"):
        resolve_model_dir({"STT_DIARIZATION_MODEL_DIR": str(tmp_path)}, cached_snapshot=_never)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"HF_HUB_CACHE": "/c", "HF_HOME": "/h"}, "/c"),
        ({"HF_HOME": "/h"}, str(Path("/h") / "hub")),
        ({}, None),
    ],
)
def test_cached_snapshot_gets_the_env_cache_dir(tmp_path: Path, env: dict[str, str], expected: str | None) -> None:
    seen: list[str | None] = []

    def cached(cache_dir: str | None) -> Path:
        seen.append(cache_dir)
        return _snapshot(tmp_path)

    assert resolve_model_dir(env, cached_snapshot=cached) == tmp_path
    assert seen == [expected]


def test_incomplete_snapshot_is_not_installed(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    (tmp_path / "plda" / "plda.npz").unlink()
    with pytest.raises(DiarizationConfigError, match="Speaker model not installed"):
        resolve_model_dir({}, cached_snapshot=lambda _c: tmp_path)


def test_nothing_installed_names_both_fixes() -> None:
    with pytest.raises(DiarizationConfigError) as exc_info:
        resolve_model_dir({}, cached_snapshot=lambda _c: None)
    msg = str(exc_info.value)
    assert "make diarization-model" in msg
    assert "run setup again" in msg


def test_every_revision_literal_is_the_pinned_revision() -> None:
    """Bandit B615 needs the SHA spelled inside each call; this keeps every copy equal to the constant."""
    tree = ast.parse(inspect.getsource(model))
    literals = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "revision" and isinstance(kw.value, ast.Constant)
    ]
    assert literals
    assert set(literals) == {DIARIZATION_REVISION}
