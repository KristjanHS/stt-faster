"""Unit tests for backend.diarize.model — local-only model resolution + pinned revision."""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.diarize import model
from backend.diarize.errors import DiarizationConfigError
from backend.diarize.model import (
    DIARIZATION_INCLUDE,
    DIARIZATION_REPO,
    DIARIZATION_REVISION,
    DIARIZATION_SHA256,
    fetch_model,
    resolve_model_dir,
    verify_snapshot,
)


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


def _downloader(root: Path, calls: list[dict[str, object]]) -> Callable[..., str]:
    """Writes each weight file with its own name as content (hashes pinned by `_NAME_HASHES`)."""

    def download(repo_id: str, **kwargs: object) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        for name in DIARIZATION_SHA256:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(name.encode())
        return str(root)

    return download


_NAME_HASHES = {name: hashlib.sha256(name.encode()).hexdigest() for name in DIARIZATION_SHA256}


def test_fetch_model_verifies_and_sends_no_token(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    assert fetch_model(downloader=_downloader(tmp_path, calls), cache_dir="/c", sha256=_NAME_HASHES) == tmp_path
    assert calls == [
        {
            "repo_id": DIARIZATION_REPO,
            "revision": DIARIZATION_REVISION,
            "allow_patterns": list(DIARIZATION_INCLUDE),
            "token": False,
            "cache_dir": "/c",
        }
    ]


def test_fetch_model_bad_hash_removes_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    tampered = {**_NAME_HASHES, "plda/plda.npz": "0" * 64}
    with pytest.raises(DiarizationConfigError, match=r"corrupt download \(plda/plda.npz\)"):
        fetch_model(downloader=_downloader(snap, []), sha256=tampered)
    assert not snap.exists()


def test_verify_snapshot_deletes_cache_blobs_but_not_linked_in_files(tmp_path: Path) -> None:
    snap = tmp_path / "repo" / "snapshots" / "rev"
    blob = tmp_path / "repo" / "blobs" / "b1"
    external = tmp_path / "shared" / "plda.npz"
    for path in (snap, blob.parent, external.parent):
        path.mkdir(parents=True)
    blob.write_bytes(b"x")
    external.write_bytes(b"y")
    (snap / "a.bin").symlink_to(blob)
    (snap / "b.bin").symlink_to(external)
    with pytest.raises(DiarizationConfigError):
        verify_snapshot(snap, {"a.bin": "0" * 64, "b.bin": "0" * 64})
    assert not blob.exists()
    assert external.exists()
    assert not snap.exists()
