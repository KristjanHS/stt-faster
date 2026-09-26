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
  "rev-list --count HEAD..origin/main") echo "${FAKE_BEHIND:-0}" ;;
  "push "*) exit "${FAKE_PUSH_RC:-0}" ;;
esac
exit 0
"""

FAKE_GH = r"""#!/usr/bin/env bash
all="$*"
echo "gh ${all//$'\n'/ }" >> "$FAKE_LOG"
[[ "$1 $2" != "auth status" ]] || exit "${FAKE_GH_AUTH_RC:-0}"
[[ "$1 $2" != "release create" ]] || exit "${FAKE_GH_CREATE_RC:-0}"
if [[ "$1 $2" == "release list" ]]; then
  [[ -z "${FAKE_RELEASES:-}" ]] || printf '%s\n' $FAKE_RELEASES
  exit "${FAKE_GH_LIST_RC:-0}"
fi
if [[ "$1 $2" == "release view" ]]; then
  [[ " ${FAKE_EXE_TAGS:-} " != *" $3 "* ]] || echo "Transcribe-Setup.exe"
  echo "notes.txt"
fi
[[ "$1 $2" != "release delete-asset" ]] || exit "${FAKE_GH_DELETE_RC:-0}"
exit 0
"""

FAKE_UV = r"""#!/usr/bin/env bash
echo "uv $*" >> "$FAKE_LOG"
"""


def _run(tmp_path: Path, *, version: str = "1.1.0", fake_env: dict[str, str] | None = None):
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
    (root / "dist").mkdir()  # a leftover local build must not be attached
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
        ("1.1.0", {"FAKE_BEHIND": "2"}, "behind origin/main"),
        ("1.1.0", {"FAKE_GH_AUTH_RC": "1"}, "gh auth login"),
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


def test_happy_path_publishes_without_an_exe(tmp_path: Path) -> None:
    result, calls, root = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    pyproject = (root / "pyproject.toml").read_text()
    assert 'version = "1.1.0"' in pyproject
    assert 'version = "9.9.9"' in pyproject  # only the first `version =` line is bumped
    mutations = _mutations(calls)
    assert mutations[:4] == [
        "uv lock",
        "git commit -m chore(release): v1.1.0 -- pyproject.toml uv.lock",
        "git tag -a v1.1.0 -m v1.1.0",
        "git push --atomic origin main v1.1.0",
    ]
    create = mutations[4]
    assert create.startswith("gh release create v1.1.0 --title v1.1.0 --generate-notes")  # no asset argument
    assert "--verify-tag" in create
    assert "dist/" not in create
    assert "Run anyway" in create
    assert "Keep anyway" in create
    assert not any(c.startswith(("gh release download", "gh release upload")) for c in calls)


def test_rerun_after_bump_skips_commit(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path, version="0.1.0")
    assert result.returncode == 0, result.stderr
    assert not any(c.startswith(("uv lock", "git commit")) for c in calls)
    assert "git tag -a v0.1.0 -m v0.1.0" in calls


@pytest.mark.parametrize(
    ("fake_env", "hint"),
    [
        ({"FAKE_PUSH_RC": "1"}, "git tag -d v1.1.0"),
        ({"FAKE_GH_CREATE_RC": "1"}, "gh release create v1.1.0 --title v1.1.0"),
    ],
)
def test_failure_after_first_change_prints_recovery(tmp_path: Path, fake_env: dict[str, str], hint: str) -> None:
    result, _, _ = _run(tmp_path, fake_env=fake_env)
    assert result.returncode != 0
    assert f"to recover: {hint}" in result.stderr


OLD_RELEASES = "v1.1.0 v1.0.4 v1.0.3 v1.0.2 v1.0.1 v1.0.0 v0.9.0"  # newest first, as `gh release list` prints


def _deleted(calls: list[str]) -> list[str]:
    return [c.split()[3] for c in calls if c.startswith("gh release delete-asset")]


def test_prunes_exe_from_all_but_newest_five_releases(tmp_path: Path) -> None:
    fake_env = {"FAKE_RELEASES": OLD_RELEASES, "FAKE_EXE_TAGS": "v1.0.4 v1.0.3 v1.0.2 v1.0.1 v1.0.0"}
    result, calls, _ = _run(tmp_path, fake_env=fake_env)
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == ["v1.0.0"]  # v0.9.0 is past the cut but carries no exe
    assert "gh release delete-asset v1.0.0 Transcribe-Setup.exe --yes" in calls
    create = next(i for i, c in enumerate(calls) if c.startswith("gh release create"))
    delete = next(i for i, c in enumerate(calls) if c.startswith("gh release delete-asset"))
    assert create < delete  # never prune before the new release is out


def test_five_or_fewer_releases_prune_nothing(tmp_path: Path) -> None:
    fake_env = {"FAKE_RELEASES": "v1.1.0 v1.0.4 v1.0.3 v1.0.2 v1.0.1", "FAKE_EXE_TAGS": "v1.0.4 v1.0.3 v1.0.2 v1.0.1"}
    result, calls, _ = _run(tmp_path, fake_env=fake_env)
    assert result.returncode == 0, result.stderr
    assert _deleted(calls) == []


@pytest.mark.parametrize("fake_env", [{"FAKE_GH_DELETE_RC": "1"}, {"FAKE_GH_LIST_RC": "1"}])
def test_prune_failure_only_warns(tmp_path: Path, fake_env: dict[str, str]) -> None:
    fake_env = {"FAKE_RELEASES": OLD_RELEASES, "FAKE_EXE_TAGS": "v1.0.0 v0.9.0", **fake_env}
    result, _, _ = _run(tmp_path, fake_env=fake_env)
    assert result.returncode == 0, result.stderr
    assert "warning" in result.stderr
    assert "to recover" not in result.stderr
