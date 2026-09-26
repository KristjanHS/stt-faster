"""scripts/select_variant.sh: linked worktrees inherit the main checkout's variant."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _variant(checkout: Path) -> str:
    result = subprocess.run(
        ["bash", str(checkout / "scripts" / "select_variant.sh")],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def main_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    (main / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "select_variant.sh", main / "scripts")
    _git(main, "init", "-q")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "init")
    worktree = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(worktree))
    return main, worktree


def test_worktree_inherits_main_checkout_variant(main_and_worktree: tuple[Path, Path]) -> None:
    main, worktree = main_and_worktree
    (main / ".stt-variant.local").write_text("cu130\n")

    assert _variant(worktree) == "cu130"


def test_worktree_own_variant_overrides_main(main_and_worktree: tuple[Path, Path]) -> None:
    main, worktree = main_and_worktree
    (main / ".stt-variant.local").write_text("cu130\n")
    (worktree / ".stt-variant.local").write_text("cpu\n")

    assert _variant(worktree) == "cpu"


def test_no_variant_file_anywhere_defaults_to_cpu(main_and_worktree: tuple[Path, Path]) -> None:
    main, worktree = main_and_worktree

    assert _variant(main) == "cpu"
    assert _variant(worktree) == "cpu"
