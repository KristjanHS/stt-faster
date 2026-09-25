"""Pure logic behind installer/setup_gui.py — paths, archives, commands, config, task ordering."""

from __future__ import annotations

import io
import sys
import tarfile
import threading
import tomllib
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from backend.gui import default_app_paths
from installer.setup_gui import (
    HF_TOOL_PINS,
    MODELS,
    Events,
    InstallError,
    InstallPaths,
    ModelSpec,
    Task,
    classify_source,
    clean_install,
    default_install_paths,
    deps_command,
    deps_env,
    dir_size,
    ensure_device_config,
    expected_model_size,
    extract_named,
    extract_zip_stripped,
    hf_hub_cache,
    latest_release_zip,
    model_cache_dir,
    model_command,
    run_tasks,
    shortcut_script,
    source_ignore,
    swap_in,
    uv_asset,
)


@pytest.fixture
def paths(tmp_path: Path) -> InstallPaths:
    return InstallPaths(install_dir=tmp_path / "inst", config_file=tmp_path / "cfg" / "config", windows=True)


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_install_paths_match_gui_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plat: str) -> None:
    env = {k: str(tmp_path / k) for k in ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME")}
    monkeypatch.setattr(sys, "platform", plat)
    gui = default_app_paths(env)
    ours = default_install_paths(env, plat)
    assert (ours.install_dir, ours.config_file) == (gui.install_dir, gui.config_file)
    assert ours.ffmpeg_bin == gui.ffmpeg_bin


def test_gui_exe_path_per_platform(tmp_path: Path) -> None:
    win = InstallPaths(tmp_path, tmp_path / "c", windows=True)
    nix = InstallPaths(tmp_path, tmp_path / "c", windows=False)
    assert win.gui_exe == tmp_path / ".venv" / "Scripts" / "stt-faster-gui.exe"
    assert nix.gui_exe == tmp_path / ".venv" / "bin" / "stt-faster-gui"


def test_uv_asset() -> None:
    assert uv_asset("win32", "AMD64") == "uv-x86_64-pc-windows-msvc.zip"
    assert uv_asset("linux", "x86_64") == "uv-x86_64-unknown-linux-gnu.tar.gz"
    with pytest.raises(InstallError):
        uv_asset("win32", "i386")


def test_hf_hub_cache_resolution(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert hf_hub_cache({}, home) == home / ".cache" / "huggingface" / "hub"
    assert hf_hub_cache({"HF_HOME": "/h"}, home) == Path("/h/hub")
    assert hf_hub_cache({"HF_HOME": "/h", "HF_HUB_CACHE": "/c"}, home) == Path("/c")
    assert model_cache_dir(Path("/c"), "Org/name") == Path("/c/models--Org--name")


def test_dir_size_skips_symlinks(tmp_path: Path) -> None:
    (tmp_path / "blobs").mkdir()
    blob = tmp_path / "blobs" / "abc"
    blob.write_bytes(b"x" * 10)
    (tmp_path / "snap").mkdir()
    (tmp_path / "snap" / "model.bin").symlink_to(blob)
    assert dir_size(tmp_path) == 10
    assert dir_size(tmp_path / "missing") == 0


def test_expected_model_size_filters_include() -> None:
    tree = [
        {"type": "file", "path": "ct2/model.bin", "size": 100, "lfs": {"size": 3000}},
        {"type": "file", "path": "ct2/config.json", "size": 20},
        {"type": "file", "path": "pytorch_model.bin", "size": 9999},
        {"type": "directory", "path": "ct2"},
    ]
    assert expected_model_size(ModelSpec("a/b", ("ct2/*",)), lambda _url: tree) == 3020
    assert expected_model_size(ModelSpec("a/b"), lambda _url: tree) == 3020 + 9999


def test_expected_model_size_unknown_on_error() -> None:
    def boom(_url: str) -> Any:
        raise OSError("offline")

    assert expected_model_size(ModelSpec("a/b"), boom) is None


def test_latest_release_zip_404_points_at_source_flag() -> None:
    def not_found(url: str) -> Any:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    with pytest.raises(InstallError, match="--source"):
        latest_release_zip(not_found)
    assert latest_release_zip(lambda _u: {"zipball_url": "https://z"}) == "https://z"


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_zip_stripped_drops_top_folder(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "s.zip", {"repo-abc/pyproject.toml": b"p", "repo-abc/backend/x.py": b"x"})
    extract_zip_stripped(archive, tmp_path / "out")
    assert (tmp_path / "out" / "pyproject.toml").read_bytes() == b"p"
    assert (tmp_path / "out" / "backend" / "x.py").is_file()


def test_extract_zip_stripped_rejects_traversal(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "s.zip", {"top/../../evil.txt": b"e", "top/ok": b"o"})
    with pytest.raises(InstallError, match="Unsafe"):
        extract_zip_stripped(archive, tmp_path / "out")


def test_extract_named_zip_flattens_and_requires_all(tmp_path: Path) -> None:
    members = {"ff-9/bin/ffmpeg.exe": b"f", "ff-9/bin/ffprobe.exe": b"p", "ff-9/doc/x": b"d"}
    archive = _zip(tmp_path / "ff.zip", members)
    extract_named(archive, ("ffmpeg.exe", "ffprobe.exe"), tmp_path / "bin")
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == ["ffmpeg.exe", "ffprobe.exe"]
    with pytest.raises(InstallError, match="ffplay.exe"):
        extract_named(archive, ("ffplay.exe",), tmp_path / "bin2")


def test_extract_named_tarball(tmp_path: Path) -> None:
    archive = tmp_path / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"uv"))
    extract_named(archive, ("uv",), tmp_path / "bin")
    assert (tmp_path / "bin" / "uv").read_bytes() == b"uv"


def test_swap_in_replaces_target(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "stale.py").write_text("old")
    (tmp_path / "app.new").mkdir()
    (tmp_path / "app.new" / "pyproject.toml").write_text("new")
    swap_in(tmp_path / "app.new", tmp_path / "app")
    assert [p.name for p in (tmp_path / "app").iterdir()] == ["pyproject.toml"]
    assert not (tmp_path / "app.new").exists() and not (tmp_path / "app.old").exists()


def test_classify_source(tmp_path: Path) -> None:
    assert classify_source("https://github.com/o/r/archive/main.zip") == "url"
    assert classify_source(str(tmp_path)) == "dir"
    assert classify_source(str(_zip(tmp_path / "a.zip", {"a/b": b""}))) == "zip"
    with pytest.raises(InstallError):
        classify_source(str(tmp_path / "nope"))


def test_source_ignore_skips_dot_and_output_dirs(tmp_path: Path) -> None:
    for d in (".venv", ".cache", "logs", "backend"):
        (tmp_path / d).mkdir()
    (tmp_path / ".python-version").write_text("3.12")
    names = [".venv", ".cache", "logs", "backend", ".python-version"]
    assert source_ignore(str(tmp_path), names) == {".venv", ".cache", "logs"}


def test_deps_command_uses_lean_gui_install(paths: InstallPaths) -> None:
    cmd = deps_command(paths)
    assert cmd[:4] == [str(paths.uv_exe), "sync", "--frozen", "--no-dev"]
    assert "--extra" in cmd and cmd[cmd.index("--extra") + 1] == "gui"
    assert cmd.count("--extra") == 1  # no cpu/cu130: diarization stays out of the lean install
    assert deps_env({"A": "1"}, paths) == {"A": "1", "UV_PROJECT_ENVIRONMENT": str(paths.venv_dir)}


def test_model_command_include_and_force(paths: InstallPaths) -> None:
    estonian = next(m for m in MODELS if m.include)
    cmd = model_command(paths, estonian, force=False)
    assert cmd[cmd.index("download") + 1 :] == [estonian.repo_id, "--include", "ct2/*"]
    assert model_command(paths, estonian, force=True)[-1] == "--force-download"
    assert cmd[cmd.index("--python") + 1] == "3.12"
    assert cmd[cmd.index("--from") + 1].startswith("huggingface_hub==")


def test_hf_tool_pins_match_uv_lock() -> None:
    lock = tomllib.loads((Path(__file__).parents[2] / "uv.lock").read_text(encoding="utf-8"))
    locked = {pkg["name"]: pkg["version"] for pkg in lock["package"]}
    for pin in HF_TOOL_PINS:
        name, version = pin.split("==")
        assert locked[name.replace("_", "-")] == version, pin


def test_shortcut_script_quotes_paths() -> None:
    script = shortcut_script(Path("C:/Users/O'Brien/stt/gui.exe"))
    assert "'C:/Users/O''Brien/stt/gui.exe'" in script
    assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script


def test_ensure_device_config_writes_cpu_once(tmp_path: Path) -> None:
    config = tmp_path / "cfg" / "config"
    ensure_device_config(config)
    assert config.read_text() == "device=cpu\n"
    config.write_text("token=abc\ndevice=cuda\n")
    ensure_device_config(config)
    assert config.read_text() == "token=abc\ndevice=cuda\n"


def test_clean_install_keeps_setup_copy_and_logs(paths: InstallPaths) -> None:
    for d in (paths.app_dir, paths.venv_dir, paths.ffmpeg_bin, paths.uv_dir, paths.log_file.parent):
        d.mkdir(parents=True)
    paths.setup_copy.write_bytes(b"exe")
    clean_install(paths)
    assert sorted(p.name for p in paths.install_dir.iterdir()) == ["Transcribe-Setup.exe", "logs"]


def _recorder() -> tuple[Events, list[tuple[str, str]]]:
    log: list[tuple[str, str]] = []
    lock = threading.Lock()

    def state(key: str, value: str, _message: str) -> None:
        with lock:
            log.append((key, value))

    return Events(progress=lambda *_a: None, state=state), log


def test_run_tasks_orders_dependencies() -> None:
    order: list[str] = []
    events, _log = _recorder()
    tasks = [
        Task("a", "A", lambda _r: order.append("a")),
        Task("b", "B", lambda _r: order.append("b"), needs=("a",)),
    ]
    assert run_tasks(tasks, events) is True
    assert order == ["a", "b"]


def test_run_tasks_skips_dependants_of_failure() -> None:
    def fail(_report: object) -> None:
        raise InstallError("net down")

    events, log = _recorder()
    tasks = [
        Task("a", "A", fail),
        Task("c", "C", lambda _r: None),
        Task("b", "B", lambda _r: None, needs=("a",)),
    ]
    assert run_tasks(tasks, events) is False
    assert ("a", "failed") in log and ("b", "skipped") in log and ("c", "done") in log
    assert ("b", "running") not in log
