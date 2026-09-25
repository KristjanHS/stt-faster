"""scripts/installer_reuse.sh against fake git/gh binaries: rebuild the exe only when installer/ changed."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the repo's own release script
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "installer_reuse.sh"

FAKE_GIT = r"""#!/usr/bin/env bash
echo "git $*" >> "$FAKE_LOG"
[[ "$1" != "diff" ]] || exit "${FAKE_DIFF_RC:-0}"
exit 0
"""

FAKE_GH = r"""#!/usr/bin/env bash
echo "gh $*" >> "$FAKE_LOG"
case "$1 $2" in
  "release list") [[ -z "${FAKE_LIST_RC:-}" ]] || exit "$FAKE_LIST_RC"; printf '%s\n' ${FAKE_RELEASES:-} ;;
  "release view") [[ " ${FAKE_WITH_EXE:-} " != *" $3 "* ]] || printf 'source.zip\nTranscribe-Setup.exe\n' ;;
esac
exit 0
"""


def _run(
    tmp_path: Path, fake_env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], list[str], dict[str, str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("git", FAKE_GIT), ("gh", FAKE_GH)):
        fake = bin_dir / name
        fake.write_text(body)
        fake.chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    github_env = tmp_path / "github_env"
    github_env.touch()
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GITHUB_ENV": str(github_env)}
    env.pop("FORCE_REBUILD", None)
    env.update(FAKE_LOG=str(log), **fake_env)
    result = subprocess.run(
        ["bash", str(SCRIPT), "v1.3.0", "dist"], env=env, capture_output=True, text=True, check=False
    )  # nosec B603
    written = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    return result, log.read_text().splitlines(), written


def _downloads(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("gh release download")]


def test_unchanged_installer_reuses_newest_earlier_exe(tmp_path: Path) -> None:
    # v1.3.0 (the release being built) is skipped even if a re-run already attached an exe to it.
    fake = {"FAKE_RELEASES": "v1.3.0 v1.2.0 v1.1.0", "FAKE_WITH_EXE": "v1.3.0 v1.2.0 v1.1.0"}
    result, calls, written = _run(tmp_path, fake)
    assert result.returncode == 0, result.stderr
    assert written == {"REBUILD": "false", "REUSE_FROM": "v1.2.0"}
    assert "git diff --quiet v1.2.0 v1.3.0 -- installer/" in calls
    assert _downloads(calls) == ["gh release download v1.2.0 --pattern Transcribe-Setup.exe --dir dist --clobber"]


def test_skips_releases_without_an_exe(tmp_path: Path) -> None:
    result, calls, written = _run(tmp_path, {"FAKE_RELEASES": "v1.2.0 v1.1.0", "FAKE_WITH_EXE": "v1.1.0"})
    assert result.returncode == 0, result.stderr
    assert written["REUSE_FROM"] == "v1.1.0"


@pytest.mark.parametrize(
    ("fake_env", "reason"),
    [
        ({"FAKE_RELEASES": "v1.2.0", "FAKE_WITH_EXE": "v1.2.0", "FAKE_DIFF_RC": "1"}, "installer/ changed"),
        ({"FAKE_RELEASES": "v1.3.0 v1.2.0", "FAKE_WITH_EXE": "v1.3.0"}, "no earlier release"),
        ({"FAKE_RELEASES": "v1.2.0", "FAKE_WITH_EXE": "v1.2.0", "FORCE_REBUILD": "true"}, "rebuild forced"),
    ],
)
def test_rebuilds_without_downloading(tmp_path: Path, fake_env: dict[str, str], reason: str) -> None:
    result, calls, written = _run(tmp_path, fake_env)
    assert result.returncode == 0, result.stderr
    assert written == {"REBUILD": "true", "REUSE_FROM": ""}
    assert reason in result.stdout
    assert _downloads(calls) == []


def test_diff_error_fails_instead_of_guessing(tmp_path: Path) -> None:
    result, calls, written = _run(
        tmp_path, {"FAKE_RELEASES": "v1.2.0", "FAKE_WITH_EXE": "v1.2.0", "FAKE_DIFF_RC": "128"}
    )
    assert result.returncode != 0
    assert "are both tags fetched" in result.stderr
    assert written == {}
    assert _downloads(calls) == []


def test_failed_release_listing_fails_instead_of_rebuilding(tmp_path: Path) -> None:
    result, calls, written = _run(tmp_path, {"FAKE_RELEASES": "v1.2.0", "FAKE_WITH_EXE": "v1.2.0", "FAKE_LIST_RC": "1"})
    assert result.returncode != 0
    assert written == {}
    assert _downloads(calls) == []
