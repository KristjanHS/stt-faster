"""Pure logic behind installer/setup_gui.py — paths, archives, commands, config, task ordering."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tarfile
import threading
import time
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
    Cancelled,
    Events,
    Installer,
    InstallError,
    InstallPaths,
    ModelSpec,
    Task,
    app_in_use,
    classify_source,
    clean_install,
    default_install_paths,
    deps_command,
    deps_env,
    dir_size,
    download,
    ensure_device_config,
    expected_model_size,
    extract_named,
    extract_zip_stripped,
    hf_hub_cache,
    isolated_env,
    latest_release_zip,
    model_cache_dir,
    model_command,
    register_uninstall,
    remove_shortcuts_script,
    run_process,
    run_tasks,
    seed_model_cache,
    self_delete_command,
    shortcut_script,
    uninstall,
    unregister_uninstall,
    source_ignore,
    swap_in,
    system_tool,
    uv_asset,
)


@pytest.fixture
def paths(tmp_path: Path) -> InstallPaths:
    return InstallPaths(install_dir=tmp_path / "inst", config_file=tmp_path / "cfg" / "config", windows=True)


@pytest.mark.parametrize("with_env", [True, False], ids=["env", "fallbacks"])
@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_install_paths_match_gui_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plat: str, with_env: bool
) -> None:
    keys = ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME")
    env = {k: str(tmp_path / k) for k in keys} if with_env else {}
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


@pytest.mark.parametrize(
    "members",
    [
        {"top/C:/x": b"x", "top/ok": b"o"},  # stripped to a drive-relative `C:/x`
        {"/abs/evil": b"e", "other/ok": b"o"},  # mixed tops, so nothing is stripped
    ],
    ids=["drive", "rooted"],
)
def test_extract_zip_stripped_rejects_drive_and_rooted(tmp_path: Path, members: dict[str, bytes]) -> None:
    with pytest.raises(InstallError, match="Unsafe"):
        extract_zip_stripped(_zip(tmp_path / "s.zip", members), tmp_path / "out")


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


def test_swap_in_restores_target_when_rename_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("old")
    (tmp_path / "app.new").mkdir()
    real_rename = Path.rename

    def flaky_rename(self: Path, target: Any) -> Path:
        if self.name == "app.new":
            raise OSError("locked")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(OSError, match="locked"):
        swap_in(tmp_path / "app.new", tmp_path / "app")
    assert (tmp_path / "app" / "main.py").read_text() == "old"
    assert not (tmp_path / "app.old").exists()


def test_download_file_url(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"abc" * 10)
    dest = tmp_path / "dl" / "out.bin"
    seen: list[tuple[int, int | None]] = []
    download(source.as_uri(), dest, lambda done, total: seen.append((done, total)), threading.Event(), chunk=8)
    assert dest.read_bytes() == b"abc" * 10
    assert seen[-1] == (30, 30) and len(seen) == 4
    assert not dest.with_name("out.bin.part").exists()


def test_download_cancelled_leaves_no_part(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"abc")
    dest = tmp_path / "out.bin"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        download(source.as_uri(), dest, lambda _d, _t: None, cancel)
    assert list(tmp_path.iterdir()) == [source]


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
    assert deps_env({"A": "1"}, paths) == {
        **isolated_env({"A": "1"}, paths),
        "UV_PROJECT_ENVIRONMENT": str(paths.venv_dir),
    }


def test_isolated_env_overrides_inherited_caches(paths: InstallPaths) -> None:
    env = isolated_env({"HF_HUB_CACHE": "/shared/hub", "HF_HOME": "/shared", "UV_CACHE_DIR": "/shared/uv"}, paths)
    for key in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        assert Path(env[key]).is_relative_to(paths.install_dir), key


def test_seed_model_cache_copies_once_and_keeps_legacy(tmp_path: Path) -> None:
    legacy, target = tmp_path / "old" / "models--a--b", tmp_path / "new" / "models--a--b"
    (legacy / "blobs").mkdir(parents=True)
    (legacy / "blobs" / "x").write_bytes(b"model")
    (legacy / "snapshots").mkdir()
    (legacy / "snapshots" / "x").symlink_to("../blobs/x")
    target.parent.mkdir()
    assert seed_model_cache(legacy, target) is True
    assert (target / "snapshots" / "x").read_bytes() == b"model" and (legacy / "blobs" / "x").is_file()
    assert not target.with_name(target.name + ".seed").exists()
    assert seed_model_cache(legacy, target) is False  # never overwrites an existing install copy
    assert seed_model_cache(tmp_path / "missing", tmp_path / "new" / "other") is False


def test_seed_model_cache_failed_copy_leaves_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy, target = tmp_path / "old", tmp_path / "new" / "models--a--b"
    legacy.mkdir()
    target.parent.mkdir()

    def broken(_src: Path, dst: Path, **_kw: Any) -> None:
        dst.mkdir()
        raise OSError("disk full")

    monkeypatch.setattr("installer.setup_gui.shutil.copytree", broken)
    assert seed_model_cache(legacy, target) is False
    assert list(target.parent.iterdir()) == []


class _FakeReg:
    HKEY_CURRENT_USER, REG_SZ, REG_DWORD = "HKCU", 1, 4

    def __init__(self) -> None:
        self.keys: dict[str, dict[str, Any]] = {}

    def CreateKey(self, _root: str, name: str) -> Any:  # noqa: N802 - winreg's API
        self.keys.setdefault(name, {})
        return contextlib.nullcontext(name)

    def SetValueEx(self, key: str, name: str, _reserved: int, _kind: int, value: Any) -> None:  # noqa: N802
        self.keys[key][name] = value

    def DeleteKey(self, _root: str, name: str) -> None:  # noqa: N802
        if name not in self.keys:
            raise FileNotFoundError(name)
        del self.keys[name]


def test_uninstall_registry_entry_roundtrip(paths: InstallPaths) -> None:
    reg = _FakeReg()
    register_uninstall(paths, reg)
    ((key, values),) = reg.keys.items()
    assert key.endswith(r"CurrentVersion\Uninstall\stt-faster")
    assert values["UninstallString"] == f'"{paths.setup_copy}" --uninstall'
    unregister_uninstall(reg)
    unregister_uninstall(reg)  # already gone is not an error
    assert reg.keys == {}


def test_uninstall_removes_install_and_config_only(tmp_path: Path) -> None:
    linux = InstallPaths(tmp_path / "share" / "stt-faster", tmp_path / "cfg" / "stt-faster" / "config", windows=False)
    for d in (linux.venv_dir, linux.hf_home / "hub", linux.config_file.parent):
        d.mkdir(parents=True)
    linux.config_file.write_text("device=cpu\n")
    (tmp_path / "cfg" / "other").mkdir()
    uninstall(linux, {})
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == ["cfg", "cfg/other", "share"]


def test_uninstall_refuses_unexpected_folder_and_open_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    odd = InstallPaths(tmp_path / "home", tmp_path / "cfg" / "stt-faster" / "config", windows=False)
    odd.install_dir.mkdir()
    with pytest.raises(InstallError, match="unexpected folder"):
        uninstall(odd, {})
    assert odd.install_dir.is_dir()
    monkeypatch.setattr("installer.setup_gui.app_in_use", lambda _p: True)
    with pytest.raises(InstallError, match="Close Transcribe"):
        uninstall(odd, {})


def test_remove_shortcuts_script_targets_both_folders() -> None:
    script = remove_shortcuts_script()
    assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script
    assert "'Transcribe.lnk'" in script


def test_self_delete_command_removes_only_its_temp_folder() -> None:
    exe = Path("C:/Temp/stt-faster-uninstall-ab12/Transcribe-Setup.exe")
    cmd = self_delete_command(exe, {"SYSTEMROOT": "C:/Windows"})
    assert f'rmdir /s /q "{exe.parent}"' in cmd and f'if not exist "{exe.parent}" exit' in cmd
    assert cmd.startswith(f'"{Path("C:/Windows", "System32", "cmd.exe")}" /d /c for /l')


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
    config.write_text("token=abc\n")
    ensure_device_config(config)
    assert config.read_text() == "token=abc\ndevice=cpu\n"


def test_clean_install_keeps_setup_copy_and_logs(paths: InstallPaths) -> None:
    for d in (paths.app_dir, paths.venv_dir, paths.ffmpeg_bin, paths.uv_dir, paths.log_file.parent):
        d.mkdir(parents=True)
    paths.setup_copy.write_bytes(b"exe")
    clean_install(paths)
    assert sorted(p.name for p in paths.install_dir.iterdir()) == ["Transcribe-Setup.exe", "logs"]
    clean_install(paths)  # nothing left to remove is not an error


def test_clean_install_surfaces_locked_venv(monkeypatch: pytest.MonkeyPatch, paths: InstallPaths) -> None:
    def locked(path: Path, *, ignore_errors: bool = False) -> None:
        if not ignore_errors:
            raise PermissionError(13, "Access is denied", str(path))

    monkeypatch.setattr("installer.setup_gui.shutil.rmtree", locked)
    with pytest.raises(InstallError, match="Close Transcribe"):
        clean_install(paths)


def test_app_in_use_only_probes_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def locked(*_a: Any) -> None:
        raise PermissionError(13, "in use")

    win = InstallPaths(tmp_path, tmp_path / "c", windows=True)
    (win.venv_dir / "Scripts").mkdir(parents=True)
    (win.venv_dir / "Scripts" / "python.exe").write_bytes(b"")
    assert app_in_use(win) is False
    assert (win.venv_dir / "Scripts" / "python.exe").is_file()  # probe renamed it back
    monkeypatch.setattr("installer.setup_gui.os.rename", locked)
    assert app_in_use(win) is True
    assert app_in_use(InstallPaths(tmp_path, tmp_path / "c", windows=False)) is False


def test_installer_refuses_while_app_in_use(monkeypatch: pytest.MonkeyPatch, paths: InstallPaths) -> None:
    monkeypatch.setattr("installer.setup_gui.app_in_use", lambda _p: True)
    with pytest.raises(InstallError, match="Close Transcribe"):
        Installer(paths=paths, clean=True).run(_recorder()[0])
    assert not paths.install_dir.exists()


def test_system_tool_prefers_system_root() -> None:
    assert system_tool({"SYSTEMROOT": "C:/Windows"}, "System32", "taskkill.exe") == str(
        Path("C:/Windows", "System32", "taskkill.exe")
    )
    assert system_tool({}, "System32", "WindowsPowerShell", "v1.0", "powershell.exe") == "powershell"


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

    def slow_a(_report: object) -> None:
        time.sleep(0.2)  # without the dependency wait, b would append first
        order.append("a")

    tasks = [
        Task("a", "A", slow_a),
        Task("b", "B", lambda _r: order.append("b"), needs=("a",)),
    ]
    assert run_tasks(tasks, events) is True
    assert order == ["a", "b"]


def test_run_tasks_rejects_undeclared_dependency() -> None:
    with pytest.raises(ValueError, match="nope"):
        run_tasks([Task("a", "A", lambda _r: None, needs=("nope",))], _recorder()[0])


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


def test_run_process_cancel_kills_promptly() -> None:
    cancel = threading.Event()
    cancel.set()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], os.environ, lambda *_a: None, cancel)
    assert time.monotonic() - started < 5


def test_run_process_failure_reports_last_line() -> None:
    cmd = [sys.executable, "-c", "import sys; print('boom-line'); sys.exit(3)"]
    with pytest.raises(InstallError) as caught:
        run_process(cmd, os.environ, lambda *_a: None, threading.Event())
    assert str(caught.value) == "boom-line"


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().split(") ", 1)[1].startswith("Z")  # killed, not yet reaped


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_process_cancel_kills_grandchildren(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    spawner = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(30)"
    )
    cancel = threading.Event()

    def report(_fraction: float | None, _text: str) -> None:
        if pid_file.is_file() and pid_file.read_text():
            cancel.set()

    with pytest.raises(Cancelled):
        run_process([sys.executable, "-c", spawner], os.environ, report, cancel)
    grandchild = int(pid_file.read_text())
    try:
        deadline = time.monotonic() + 5
        while not _gone(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(grandchild)
    finally:
        if not _gone(grandchild):
            os.kill(grandchild, 9)
