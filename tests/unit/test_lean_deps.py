"""Lean install (no cpu/cu130 extra): no torch in the base closure; `--diarize` degrades."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.cli import transcription_commands

LOCK = Path(__file__).resolve().parents[2] / "uv.lock"
TORCH_STACK = {"torch", "torchaudio", "pyannote-audio"}


def _base_closure() -> set[str]:
    packages: list[dict[str, Any]] = tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]
    deps_by_name: dict[str, list[str]] = {}
    for pkg in packages:
        deps_by_name.setdefault(pkg["name"], []).extend(d["name"] for d in pkg.get("dependencies", []))
    seen: set[str] = set()
    todo = list(deps_by_name["stt-faster"])
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo.extend(deps_by_name.get(name, []))
    return seen


def test_base_closure_has_no_torch() -> None:
    closure = _base_closure()
    assert "ctranslate2" in closure
    assert TORCH_STACK.isdisjoint(closure), sorted(TORCH_STACK & closure)


def _run_process(tmp_path: Path, *, pyannote: bool) -> argparse.Namespace:
    args = argparse.Namespace(input_folder=str(tmp_path), diarize=True, verbose=False, variant=61, variants=None)
    captured: list[argparse.Namespace] = []

    def fake_single(a: argparse.Namespace, *_: object) -> int:
        captured.append(a)
        return 0

    with (
        patch.object(transcription_commands, "_pyannote_installed", return_value=pyannote),
        patch.object(transcription_commands, "_process_single_variant", side_effect=fake_single),
    ):
        assert transcription_commands.cmd_process(args) == 0
    return captured[0]


def test_diarize_skipped_without_pyannote(tmp_path: Path) -> None:
    assert _run_process(tmp_path, pyannote=False).diarize is False


def test_diarize_kept_with_pyannote(tmp_path: Path) -> None:
    assert _run_process(tmp_path, pyannote=True).diarize is True
