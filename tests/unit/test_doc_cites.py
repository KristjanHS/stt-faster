"""Every dated plan-doc cite (`YYYY-MM-DD-<slug>.md`) in a tracked file resolves to a tracked file.

Deleting or renaming a dated doc without updating its cites fails here. Known dangles are pinned
in `BASELINE_DANGLES`; the baseline may only shrink (a pinned name that resolves again fails too).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[2]
DATED_CITE = re.compile(r"\d{4}-\d{2}-\d{2}-[\w.-]+\.md")

# Cited dated doc names that resolve to no tracked file today. Shrink only.
BASELINE_DANGLES: frozenset[str] = frozenset()


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name for name in out.decode("utf-8").split("\0") if name]


def _dangling_cites() -> dict[str, set[str]]:
    tracked = [f for f in _tracked_files() if (REPO / f).is_file()]
    present = {PurePosixPath(f).name for f in tracked}
    dangles: dict[str, set[str]] = {}
    for rel in tracked:
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for cite in DATED_CITE.findall(text):
            if cite not in present:
                dangles.setdefault(cite, set()).add(rel)
    return dangles


def test_dated_doc_cites_resolve() -> None:
    dangles = _dangling_cites()
    new = {cite: sorted(files) for cite, files in dangles.items() if cite not in BASELINE_DANGLES}
    assert not new, f"dated doc cites with no tracked target: {new}"
    stale = sorted(BASELINE_DANGLES - dangles.keys())
    assert not stale, f"BASELINE_DANGLES entries now resolve or are uncited; drop them: {stale}"
