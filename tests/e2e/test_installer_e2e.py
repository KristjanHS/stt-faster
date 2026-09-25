"""Installer slice 4b guard: a headless Linux install + uninstall into a scratch HOME leaves no trace outside
the install and config dirs, and never touches the pre-seeded shared HF / uv caches."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from installer.setup_gui import MODELS, default_install_paths, hf_hub_cache, model_cache_dir

pytestmark = [
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux headless run"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
Snapshot = dict[str, tuple[str, int, int]]


def snapshot(root: Path) -> Snapshot:
    """Every path under ``root`` with (kind, size, mtime_ns); dirs record kind only (a new child bumps their mtime)."""
    entries: Snapshot = {}
    for path in root.rglob("*"):
        stat = path.lstat()
        kind = "link" if path.is_symlink() else "dir" if path.is_dir() else "file"
        entries[str(path.relative_to(root))] = (kind, 0, 0) if kind == "dir" else (kind, stat.st_size, stat.st_mtime_ns)
    return entries


def installer_env(home: Path, tmp: Path) -> dict[str, str]:
    drop = ("HF_", "UV_", "XDG_", "VIRTUAL_ENV", "PYTHONPATH")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop) or k == "HF_TOKEN"}
    return {**env, "HOME": str(home), "TMPDIR": str(tmp)}


def run_setup(env: dict[str, str], *args: str) -> None:
    cmd = [sys.executable, "-m", "installer.setup_gui", "--headless", *args]
    result = subprocess.run(cmd, env=env, cwd=env["HOME"], capture_output=True, text=True, timeout=3600, check=False)  # noqa: S603
    assert result.returncode == 0, f"{args} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"


def seed_shared_caches(home: Path) -> None:
    """A v1.1.0-style HF cache (models hardlinked from the host cache, no 3 GB download) plus uv's shared dirs."""
    host_hub = hf_hub_cache(os.environ, Path.home())
    for spec in MODELS:
        host_model = model_cache_dir(host_hub, spec.repo_id)
        if not host_model.is_dir():
            pytest.skip(f"host HF cache lacks {spec.repo_id}; download it once first")
        seeded = model_cache_dir(home / ".cache/huggingface/hub", spec.repo_id)
        shutil.copytree(host_model, seeded, symlinks=True, copy_function=os.link)
    for marker in (".cache/uv/CACHEDIR.TAG", ".local/share/uv/python/.lock", ".local/bin/.keep", ".config/uv/uv.toml"):
        (home / marker).parent.mkdir(parents=True, exist_ok=True)
        (home / marker).write_text("seeded\n", encoding="utf-8")


def test_install_then_uninstall_leaves_scratch_home_as_found(tmp_path: Path) -> None:
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    tmp.mkdir()
    seed_shared_caches(home)
    env = installer_env(home, tmp)
    paths = default_install_paths(env, sys.platform, home)
    owned = (paths.install_dir, paths.config_file.parent)
    before = snapshot(tmp_path)

    run_setup(env, "--source", str(REPO_ROOT))
    assert paths.gui_exe.is_file()
    assert paths.config_file.read_text(encoding="utf-8") == "device=cpu\n"
    for spec in MODELS:
        assert any(model_cache_dir(paths.hf_home / "hub", spec.repo_id).glob("snapshots/*/*"))
    installed = snapshot(tmp_path)
    changed = {p for p in before if installed.get(p) != before[p]}
    assert not changed, f"install touched pre-existing paths: {sorted(changed)[:20]}"
    new = installed.keys() - before.keys()
    stray = sorted(p for p in new if not any((tmp_path / p).is_relative_to(d) for d in owned))
    assert not stray, f"install wrote outside {owned}: {stray[:20]}"

    run_setup(env, "--uninstall", "--yes")
    after = snapshot(tmp_path)
    left = sorted(after.keys() - before.keys())
    assert after == before, (
        f"uninstall left {left[:20]}, changed {sorted(p for p in before if after.get(p) != before[p])[:20]}"
    )
