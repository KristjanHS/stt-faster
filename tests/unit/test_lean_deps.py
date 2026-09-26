"""Lean install (no cpu/cu130 extra): no torch in the base closure; `--diarize` degrades."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from backend.cli import transcription_commands

LOCK = Path(__file__).resolve().parents[2] / "uv.lock"
TORCH_STACK = {"torch", "torchaudio", "pyannote-audio"}


def _lean_closure(packages: list[dict[str, Any]]) -> set[str]:
    """Names reachable from `uv sync --no-dev --extra gui`; markers ignored (over-approximates)."""
    deps: dict[str, list[dict[str, Any]]] = {}
    extras: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pkg in packages:
        deps.setdefault(pkg["name"], []).extend(pkg.get("dependencies", []))
        for extra, entries in pkg.get("optional-dependencies", {}).items():
            extras.setdefault(pkg["name"], {}).setdefault(extra, []).extend(entries)
    seen: set[str] = set()
    todo = deps["stt-faster"] + extras["stt-faster"]["gui"]
    while todo:
        entry = todo.pop()
        name = entry["name"]
        for extra in entry.get("extra", []):
            todo.extend(extras.get(name, {}).get(extra, []))
        if name not in seen:
            seen.add(name)
            todo.extend(deps.get(name, []))
    return seen


def test_base_closure_has_no_torch() -> None:
    closure = _lean_closure(tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"])
    assert "ctranslate2" in closure
    assert TORCH_STACK.isdisjoint(closure), sorted(TORCH_STACK & closure)


def _run_process(tmp_path: Path, *, pyannote: bool) -> argparse.Namespace:
    args = argparse.Namespace(input_folder=str(tmp_path), diarize=True, verbose=False, variant=61, variants=None)
    captured: list[argparse.Namespace] = []

    def fake_single(a: argparse.Namespace, *_: object, **__: object) -> int:
        captured.append(a)
        return 0

    exit_code = transcription_commands.cmd_process(
        args, pyannote_installed=lambda: pyannote, process_single_variant=fake_single
    )
    assert exit_code == 0
    return captured[0]


def test_diarize_skipped_without_pyannote(tmp_path: Path) -> None:
    assert _run_process(tmp_path, pyannote=False).diarize is False


def test_diarize_kept_with_pyannote(tmp_path: Path) -> None:
    assert _run_process(tmp_path, pyannote=True).diarize is True
