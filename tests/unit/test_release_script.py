"""scripts/release.sh against fake git/gh/uv binaries: preflight aborts and the publish sequence."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the repo's own release script
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release.sh"

FAKE_GIT = r"""#!/usr/bin/env bash
echo "git $*" >> "$FAKE_LOG"
case "$*" in
  "rev-parse --show-toplevel") echo "$FAKE_ROOT" ;;
  "rev-parse --abbrev-ref HEAD") echo "${FAKE_BRANCH:-main}" ;;
  "status --porcelain") printf '%s' "${FAKE_STATUS:-}" ;;
  "rev-parse -q --verify refs/tags/"*) exit "${FAKE_TAG_LOCAL_RC:-1}" ;;
  "ls-remote "*) exit "${FAKE_LS_REMOTE_RC:-2}" ;;
esac
exit 0
"""

FAKE_GH = r"""#!/usr/bin/env bash
all="$*"
echo "gh ${all//$'\n'/ }" >> "$FAKE_LOG"
if [[ "$1 $2" == "release download" ]]; then
  [[ "${FAKE_GH_HAS_ASSET:-0}" == 1 ]] || exit 1
  touch "${@: -1}/Transcribe-Setup.exe"
fi
exit 0
"""

FAKE_UV = r"""#!/usr/bin/env bash
echo "uv $*" >> "$FAKE_LOG"
"""


def _run(tmp_path: Path, *, version: str = "1.1.0", with_dist_exe: bool = True, fake_env: dict[str, str] | None = None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("git", FAKE_GIT), ("gh", FAKE_GH), ("uv", FAKE_UV)):
        fake = bin_dir / name
        fake.write_text(body)
        fake.chmod(0o755)
    root = tmp_path / "repo"
    root.mkdir()
    pyproject = '[project]\nname = "stt-faster"\nversion = "0.1.0"\n\n[tool.x]\nversion = "9.9.9"\n'
    (root / "pyproject.toml").write_text(pyproject)
    if with_dist_exe:
        (root / "dist").mkdir()
        (root / "dist" / "Transcribe-Setup.exe").write_bytes(b"MZ")
    log = tmp_path / "calls.log"
    log.touch()
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.update(FAKE_LOG=str(log), FAKE_ROOT=str(root), **(fake_env or {}))
    result = subprocess.run(["bash", str(SCRIPT), version], env=env, capture_output=True, text=True, check=False)  # nosec B603
    return result, log.read_text().splitlines(), root


def _mutations(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith(("git commit", "git tag", "git push", "uv ", "gh release create"))]


@pytest.mark.parametrize(
    ("version", "fake_env", "message"),
    [
        ("1.1", {}, "usage"),
        ("1.1.0", {"FAKE_BRANCH": "feat/x"}, "not on main"),
        ("1.1.0", {"FAKE_STATUS": " M backend/gui.py\n"}, "not clean"),
        ("1.1.0", {"FAKE_TAG_LOCAL_RC": "0"}, "exists locally"),
        ("1.1.0", {"FAKE_LS_REMOTE_RC": "0"}, "already on origin"),
        ("1.1.0", {"FAKE_LS_REMOTE_RC": "128"}, "unreachable"),
    ],
)
def test_preflight_aborts_before_any_change(
    tmp_path: Path, version: str, fake_env: dict[str, str], message: str
) -> None:
    result, calls, root = _run(tmp_path, version=version, fake_env=fake_env)
    assert result.returncode != 0
    assert message in result.stderr
    assert _mutations(calls) == []
    assert 'version = "0.1.0"' in (root / "pyproject.toml").read_text()


def test_no_exe_anywhere_aborts_before_any_change(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path, with_dist_exe=False)
    assert result.returncode != 0
    assert "build_installer.bat" in result.stderr
    assert _mutations(calls) == []


def test_happy_path_with_rebuilt_exe(tmp_path: Path) -> None:
    result, calls, root = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    pyproject = (root / "pyproject.toml").read_text()
    assert 'version = "1.1.0"' in pyproject
    assert 'version = "9.9.9"' in pyproject  # only the first `version =` line is bumped
    mutations = _mutations(calls)
    assert mutations[:5] == [
        "uv lock",
        "git commit -m chore(release): v1.1.0 -- pyproject.toml uv.lock",
        "git tag -a v1.1.0 -m v1.1.0",
        "git push origin main",
        "git push origin v1.1.0",
    ]
    create = mutations[5]
    assert create.startswith("gh release create v1.1.0 dist/Transcribe-Setup.exe --title v1.1.0 --generate-notes")
    assert "--verify-tag" in create
    assert "SmartScreen" in create
    assert not any(c.startswith("gh release download") for c in calls)


def test_happy_path_reattaches_latest_release_asset(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path, with_dist_exe=False, fake_env={"FAKE_GH_HAS_ASSET": "1"})
    assert result.returncode == 0, result.stderr
    download = next(i for i, c in enumerate(calls) if c.startswith("gh release download"))
    create = next(i for i, c in enumerate(calls) if c.startswith("gh release create"))
    assert download < next(i for i, c in enumerate(calls) if c == "uv lock")
    assert "/Transcribe-Setup.exe" in calls[create]
    assert "dist/" not in calls[create]


def test_rerun_after_bump_skips_commit(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path, version="0.1.0")
    assert result.returncode == 0, result.stderr
    assert not any(c.startswith(("uv lock", "git commit")) for c in calls)
    assert "git tag -a v0.1.0 -m v0.1.0" in calls
