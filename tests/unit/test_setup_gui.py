"""Pure logic behind installer/setup_gui.py — paths, archives, commands, config, task ordering."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import queue
import sys
import tarfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from backend.gui import default_app_paths
from installer.setup_gui import (
    HF_AUTH_HELP,
    HF_TOKEN_PROBE_URL,
    HF_TOOL_PINS,
    MODELS,
    UNINSTALL_KEY,
    DIARIZATION_MODEL,
    app_env,
    GpuInfo,
    Cancelled,
    CommandFailed,
    Events,
    FFMPEG_BINARIES,
    Installer,
    InstallError,
    InstallPaths,
    PYTHON_VERSION,
    ModelSpec,
    RETRY_NOTE,
    ReadyHandoff,
    SetupWindow,
    Stalled,
    Task,
    app_in_use,
    classify_source,
    clean_install,
    default_install_paths,
    deps_command,
    deps_env,
    dir_size,
    download,
    detect_gpu,
    ensure_device_config,
    expected_model_size,
    fit_tail,
    gpu_capable,
    gpu_summary,
    headless_gpu,
    extract_named,
    extract_zip_stripped,
    hf_hub_cache,
    hf_token_status,
    is_installed,
    isolated_env,
    latest_release_zip,
    launch_gui,
    model_cache_dir,
    model_command,
    open_file,
    parse_args,
    read_config,
    register_uninstall,
    relaunch_from_temp,
    hop_behind_window,
    setup_hop,
    unlink_when_released,
    remove_shortcuts_script,
    download_progress,
    run_process,
    run_tasks,
    save_device,
    schedule_self_delete,
    seed_model_cache,
    self_delete_command,
    set_config_value,
    shortcut_script,
    UninstallWindow,
    uninstall,
    uninstall_main,
    unregister_uninstall,
    source_ignore,
    wait_for_marker,
    swap_in,
    system_tool,
    uv_asset,
)


@pytest.fixture
def paths(tmp_path: Path) -> InstallPaths:
    return InstallPaths(install_dir=tmp_path / "inst", config_file=tmp_path / "cfg" / "config", windows=True)


@pytest.fixture
def win_paths(tmp_path: Path) -> InstallPaths:
    """Windows layout whose folders pass uninstall's ``stt-faster`` name guard."""
    return InstallPaths(tmp_path / "local" / "stt-faster", tmp_path / "roam" / "stt-faster" / "config", windows=True)


@pytest.fixture
def popen_calls() -> list[tuple[Any, dict[str, Any]]]:
    return []


@pytest.fixture
def popen(popen_calls: list[tuple[Any, dict[str, Any]]]) -> Callable[..., None]:
    """Records Popen calls into ``popen_calls`` instead of spawning."""
    return lambda args, **kw: popen_calls.append((args, kw))


@pytest.fixture
def tempdir(tmp_path: Path) -> Callable[[], str]:
    """%TEMP% under tmp_path."""
    (tmp_path / "temp").mkdir()
    return lambda: str(tmp_path / "temp")


@pytest.mark.parametrize("with_env", [True, False], ids=["env", "fallbacks"])
@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_install_paths_match_gui_paths(tmp_path: Path, plat: str, with_env: bool) -> None:
    keys = ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME")
    env = {k: str(tmp_path / k) for k in keys} if with_env else {}
    gui = default_app_paths(env, plat)
    ours = default_install_paths(env, plat)
    assert (ours.install_dir, ours.config_file) == (gui.install_dir, gui.config_file)
    assert (ours.ffmpeg_bin, ours.hf_home) == (gui.ffmpeg_bin, gui.hf_home)


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


def test_swap_in_restores_target_when_rename_fails(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("old")
    (tmp_path / "app.new").mkdir()
    real_rename = Path.rename

    def flaky_rename(self: Path, target: Any) -> Path:
        if self.name == "app.new":
            raise OSError("locked")
        return real_rename(self, target)

    with pytest.raises(OSError, match="locked"):
        swap_in(tmp_path / "app.new", tmp_path / "app", rename=flaky_rename)
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
    assert cmd[cmd.index("--python") + 1] == PYTHON_VERSION
    paths.app_dir.mkdir(parents=True)
    (paths.app_dir / ".python-version").write_text("3.12.11\n", encoding="utf-8")
    pinned = deps_command(paths)
    assert pinned[pinned.index("--python") + 1] == "3.12.11"  # the repo pin, not uv's newest 3.12
    assert deps_env({"A": "1"}, paths) == {
        **isolated_env({"A": "1"}, paths),
        "UV_PROJECT_ENVIRONMENT": str(paths.venv_dir),
    }


@pytest.mark.parametrize("key", ["SSLKEYLOGFILE", "SslKeyLogFile"])
def test_isolated_env_drops_an_inherited_ssl_key_log(paths: InstallPaths, key: str) -> None:
    # An unwritable SSLKEYLOGFILE broke `hf download` on a user's Windows machine (ssl.create_default_context).
    env = isolated_env({key: r"\\?\Volume{ba5fbc33}\virtual_file.log", "A": "1"}, paths)
    assert key not in env and env["A"] == "1"


def test_isolated_env_overrides_inherited_caches(paths: InstallPaths) -> None:
    env = isolated_env({"HF_HUB_CACHE": "/shared/hub", "HF_HOME": "/shared", "UV_CACHE_DIR": "/shared/uv"}, paths)
    for key in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        assert Path(env[key]).is_relative_to(paths.install_dir), key
    assert isolated_env({"UV_PYTHON_INSTALL_BIN": "1"}, paths)["UV_PYTHON_INSTALL_BIN"] == "0"  # no ~/.local/bin shim
    assert env["UV_PYTHON_INSTALL_REGISTRY"] == "0"  # no HKCU PEP 514 entry
    assert env["UV_NO_CONFIG"] == "1"  # the user's uv.toml is never read
    for key in ("UV_TOOL_DIR", "TMPDIR", "TEMP", "TMP"):
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


def test_seed_model_cache_failed_copy_leaves_nothing(tmp_path: Path) -> None:
    legacy, target = tmp_path / "old", tmp_path / "new" / "models--a--b"
    legacy.mkdir()
    target.parent.mkdir()

    def broken(_src: Path, dst: Path, **_kw: Any) -> None:
        assert dst.name.endswith(".seed") and not target.exists()  # copies into a staging dir, never the target
        dst.mkdir()
        raise OSError("disk full")

    assert seed_model_cache(legacy, target, copytree=broken) is False
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
    assert values["UninstallString"] == f'"{paths.setup_exe}" --uninstall'
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


def test_uninstall_reports_each_step(win_paths: InstallPaths) -> None:
    steps: list[str] = []
    uninstall(win_paths, {}, reg=_FakeReg(), run=lambda *_a, **_kw: None, step=steps.append)
    assert steps == [
        "Removing app files and models…",
        "Removing settings…",
        "Removing shortcuts…",
        "Removing the Apps & features entry…",
    ]


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="read-only dir must block unlink")
def test_partial_uninstall_still_unregisters_and_stays_retryable(win_paths: InstallPaths) -> None:
    locked = win_paths.install_dir / "hf" / "locked"
    locked.mkdir(parents=True)
    (locked / "model.bin").write_bytes(b"m")
    win_paths.config_file.parent.mkdir(parents=True)
    reg = _FakeReg()
    register_uninstall(win_paths, reg)
    locked.chmod(0o500)
    try:
        with pytest.raises(InstallError, match="model.bin"):
            uninstall(win_paths, {}, reg=reg, run=lambda *_a, **_kw: None)  # no shortcut-removal powershell
    finally:
        locked.chmod(0o700)
    assert reg.keys == {} and not win_paths.config_file.parent.exists()
    assert is_installed(win_paths)  # the window offers Uninstall again


def test_is_installed_ignores_setup_logs(paths: InstallPaths) -> None:
    assert is_installed(paths) is False
    paths.log_file.parent.mkdir(parents=True)  # the window's own log exists before any install
    paths.tmp_dir.mkdir()  # created before the first task; an aborted first install is not an install
    assert is_installed(paths) is False
    paths.setup_copy.write_bytes(b"exe")
    assert is_installed(paths) is True


def test_uninstall_refuses_unexpected_folder_and_open_app(tmp_path: Path) -> None:
    odd = InstallPaths(tmp_path / "home", tmp_path / "cfg" / "stt-faster" / "config", windows=False)
    odd.install_dir.mkdir()
    with pytest.raises(InstallError, match="unexpected folder"):
        uninstall(odd, {})
    assert odd.install_dir.is_dir()
    with pytest.raises(InstallError, match="Close Transcribe"):
        uninstall(odd, {}, in_use=lambda _p: True, sleep=lambda _s: None)
    busy = iter([True, True, False])  # the launcher that started the hop exits a moment later
    uninstall(
        odd.__class__(tmp_path / "x" / "stt-faster", odd.config_file, windows=False),
        {},
        in_use=lambda _p: next(busy),
        sleep=lambda _s: None,
    )


def test_remove_shortcuts_script_removes_desktop_link_old_link_and_folder() -> None:
    script = remove_shortcuts_script()
    assert "(Join-Path ([Environment]::GetFolderPath('Desktop')) 'Transcribe.lnk')" in script
    assert "$p = ([Environment]::GetFolderPath('Programs')); $f = Join-Path $p 'Transcribe'" in script
    assert "Remove-Item -LiteralPath (Join-Path $p 'Transcribe.lnk')" in script  # ≤1.2.x top-level link
    assert "Remove-Item -LiteralPath $f -Recurse -Force" in script


def test_self_delete_command_removes_only_its_temp_folder() -> None:
    exe = Path("C:/Temp/stt-faster-setup-ab12/Transcribe-Setup.exe")
    cmd = self_delete_command(exe, {"SYSTEMROOT": "C:/Windows"})
    assert f'rmdir /s /q "{exe.parent}"' in cmd and f'if not exist "{exe.parent}" exit' in cmd
    assert cmd.startswith(f'"{Path("C:/Windows", "System32", "cmd.exe")}" /d /c for /l')
    assert " %i in (1,1,10) do " in cmd and "%%" not in cmd  # a command line takes %i; %% is batch-file only


def test_spawned_processes_never_run_inside_what_they_delete(
    win_paths: InstallPaths,
    popen_calls: list[tuple[Any, dict[str, Any]]],
    popen: Callable[..., None],
    tempdir: Callable[[], str],
    fake_dialogs: _FakeDialogs,
) -> None:
    exe = win_paths.setup_copy
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    relaunch_from_temp(exe, ["--uninstall"], popen=popen, tempdir=tempdir)
    temp_exe = Path(popen_calls[0][0][0])
    schedule_self_delete(temp_exe, {}, popen=popen)
    window = SimpleNamespace(
        root=_FakeTk(),
        popen=popen,
        tempdir=tempdir,
        dialogs=lambda: fake_dialogs,
        logger=logging.Logger("test"),
        install_button=_FakeLabel(),
        launch_button=_FakeLabel(),
        summary=_FakeLabel(),
    )
    SetupWindow.uninstall(cast(Any, window))
    assert not window.root.destroyed and window.summary.text == "Starting uninstall…"  # until the child's window
    assert window.install_button.state == window.launch_button.state == "disabled"
    assert "--ready-file" in popen_calls[2][0]
    cwds = [Path(kw["cwd"]) for _args, kw in popen_calls]
    assert cwds == [Path(tempdir()), temp_exe.parent.parent, Path(tempdir())]
    assert not any(cwd.is_relative_to(win_paths.install_dir) or cwd.is_relative_to(temp_exe.parent) for cwd in cwds)


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


def _links(script: str) -> list[dict[str, str]]:
    """The shortcuts a generated script creates: folder, name, target, args, workdir per ``$l.Save()``."""
    links = []
    for chunk in script.split("$l = $s.CreateShortcut((Join-Path ")[1:]:
        folder, rest = chunk.split(" '", 1)
        link = {"folder": folder, "name": rest.split("'", 1)[0]}
        for key, field in (("target", "TargetPath"), ("args", "Arguments"), ("workdir", "WorkingDirectory")):
            if f"$l.{field} = '" in chunk:
                link[key] = chunk.split(f"$l.{field} = '", 1)[1].split("';", 1)[0].replace("''", "'")
        links.append(link)
    return links


def test_shortcut_script_desktop_link_and_start_menu_tools() -> None:
    paths = InstallPaths(Path("C:/Users/O'Brien/AppData/Local/stt-faster"), Path("C:/cfg/config"), windows=True)
    script = shortcut_script(paths, tools=True)
    root = str(paths.install_dir)
    assert [(link["folder"], link["name"]) for link in _links(script)] == [
        ("([Environment]::GetFolderPath('Desktop'))", "Transcribe.lnk"),
        ("$f", "Transcribe.lnk"),
        ("$f", "Repair Transcribe.lnk"),
        ("$f", "Uninstall Transcribe.lnk"),
        ("$f", "Setup log.lnk"),
    ]
    desktop, start, repair, remove, log = _links(script)
    assert desktop["target"] == start["target"] == str(paths.gui_exe)
    assert desktop["workdir"] == start["workdir"] == root  # never cwd inside the venv's Scripts
    assert repair["target"] == remove["target"] == str(paths.setup_exe)  # never the SAC-blocked exe copy
    assert paths.setup_exe.parent == paths.gui_exe.parent  # launched like the app itself
    assert "args" not in repair and remove["args"] == "--uninstall"
    assert log["target"] == str(paths.log_file)
    assert "$f = Join-Path $p 'Transcribe'" in script and "GetFolderPath('Programs')" in script
    # updates from ≤1.2.x: the old top-level Start-menu link goes, the folder is rebuilt from scratch
    assert script.index("Remove-Item -LiteralPath (Join-Path $p 'Transcribe.lnk')") < script.index("New-Item")
    assert script.index("Remove-Item -LiteralPath $f -Recurse") < script.index("New-Item")


def test_shortcut_script_without_setup_launcher_has_no_tool_links(win_paths: InstallPaths) -> None:
    names = [link["name"] for link in _links(shortcut_script(win_paths, tools=False))]
    assert names == ["Transcribe.lnk", "Transcribe.lnk"]


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


def test_clean_install_keeps_setup_copy_logs_and_download_caches(paths: InstallPaths) -> None:
    kept = (paths.log_file.parent, paths.uv_cache, paths.hf_home)
    for d in (paths.app_dir, paths.venv_dir, paths.ffmpeg_bin, paths.uv_dir, paths.python_dir, *kept):
        d.mkdir(parents=True)
    paths.setup_copy.write_bytes(b"exe")
    clean_install(paths)
    assert sorted(p.name for p in paths.install_dir.iterdir()) == ["Transcribe-Setup.exe", "hf", "logs", "uv-cache"]
    clean_install(paths)  # nothing left to remove is not an error


def test_clean_install_surfaces_locked_venv(paths: InstallPaths) -> None:
    def locked(path: Path, *, ignore_errors: bool = False) -> None:
        if not ignore_errors:
            raise PermissionError(13, "Access is denied", str(path))

    with pytest.raises(InstallError, match="Close Transcribe"):
        clean_install(paths, rmtree=locked)


def test_app_in_use_only_probes_on_windows(tmp_path: Path) -> None:
    def locked(*_a: Any) -> None:
        raise PermissionError(13, "in use")

    win = InstallPaths(tmp_path, tmp_path / "c", windows=True)
    (win.venv_dir / "Scripts").mkdir(parents=True)
    (win.venv_dir / "Scripts" / "python.exe").write_bytes(b"")
    assert app_in_use(win) is False
    assert (win.venv_dir / "Scripts" / "python.exe").is_file()  # probe renamed it back
    assert app_in_use(win, rename=locked) is True
    assert app_in_use(InstallPaths(tmp_path, tmp_path / "c", windows=False), rename=locked) is False


def test_installer_refuses_while_app_in_use(paths: InstallPaths) -> None:
    with pytest.raises(InstallError, match="Close Transcribe"):
        Installer(paths=paths, clean=True, in_use=lambda _p: True).run(_recorder()[0])
    assert not paths.install_dir.exists()


def test_fetch_model_uses_install_cache_and_seeds_only_without_clean(tmp_path: Path, paths: InstallPaths) -> None:
    runs: list[tuple[list[str], dict[str, str]]] = []
    seeds: list[tuple[Path, Path]] = []
    spec = ModelSpec("Org/name")
    installer = Installer(
        paths=paths,
        env={"HF_HOME": str(tmp_path / "shared")},
        runner=lambda cmd, env, *_a: runs.append((list(cmd), dict(env))),
        seeder=lambda legacy, target: seeds.append((legacy, target)) or False,
        model_size=lambda *_a: None,
    )
    installer.fetch_model(spec, lambda *_a: None)
    assert seeds == [(tmp_path / "shared" / "hub" / "models--Org--name", paths.hf_home / "hub" / "models--Org--name")]
    assert runs[0][1]["HF_HUB_CACHE"] == str(paths.hf_home / "hub")
    assert runs[0][1]["TQDM_POSITION"] == "-1"  # hf then prints its aggregate byte bar
    installer.clean = True
    installer.fetch_model(spec, lambda *_a: None)
    assert len(seeds) == 1 and runs[1][0][-1] == "--force-download"


def test_finish_registers_uninstall_and_tool_links_only_with_setup_launcher(win_paths: InstallPaths) -> None:
    reg = _FakeReg()
    scripts: list[str] = []
    win_paths.gui_exe.parent.mkdir(parents=True)
    win_paths.gui_exe.write_bytes(b"exe")
    win_paths.setup_copy.write_bytes(b"exe")  # a ≤1.2.3 install's copy
    installer = Installer(
        paths=win_paths, env={}, winreg=lambda: reg, runner=lambda cmd, *_a: scripts.append(list(cmd)[-1])
    )
    installer.finish(lambda *_a: None)
    assert reg.keys == {}  # no launcher to point UninstallString at
    assert "Repair Transcribe.lnk" not in scripts[-1]
    assert win_paths.setup_copy.exists()  # an older source without the launcher: its entry still needs the copy
    win_paths.setup_exe.write_bytes(b"exe")
    installer.finish(lambda *_a: None)
    assert list(reg.keys) == [UNINSTALL_KEY]
    assert not win_paths.setup_copy.exists()  # Smart App Control blocks it; nothing points at it any more
    assert "'Repair Transcribe.lnk'" in scripts[-1] and "'Uninstall Transcribe.lnk'" in scripts[-1]


def test_relaunch_from_temp_takes_the_scripts_python_along(
    tmp_path: Path,
    popen_calls: list[tuple[Any, dict[str, Any]]],
    popen: Callable[..., None],
    tempdir: Callable[[], str],
) -> None:
    script = tmp_path / "app" / "installer" / "setup_gui.py"
    runtime = tmp_path / "python"
    for f in (script, runtime / "pythonw.exe", runtime / "Lib" / "os.py", runtime / "Lib" / "site-packages" / "x.py"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    env = {"PATH": "C:/Windows", "__PYVENV_LAUNCHER__": "C:/venv/Scripts/pythonw.exe"}
    relaunch_from_temp(script, ["--extras"], popen=popen, tempdir=tempdir, runtime=runtime, env=env)
    ((cmd, kw),) = popen_calls
    hop = Path(cmd[2]).parent
    assert hop.name.startswith("stt-faster-setup-") and hop.parent == Path(tempdir())
    assert cmd == [str(hop / "python" / "pythonw.exe"), "-I", str(hop / "setup_gui.py"), "--extras"]
    assert (hop / "python" / "Lib" / "os.py").is_file() and not (hop / "python" / "Lib" / "site-packages").exists()
    assert kw["env"] == {"PATH": "C:/Windows"}  # else the copy would start as the venv it left


@pytest.mark.parametrize("home", ["install", "downloads"])
def test_setup_hop_only_from_the_install_dir(win_paths: InstallPaths, tmp_path: Path, home: str) -> None:
    program = win_paths.app_dir / "installer" / "setup_gui.py" if home == "install" else tmp_path / "Transcribe.exe"
    relaunched: list[tuple[Path, list[str]]] = []
    covered: list[tuple[Path, str]] = []
    hopped = setup_hop(
        win_paths,
        ["--extras"],
        headless=False,
        program=lambda: program,
        relaunch=lambda exe, args: relaunched.append((exe, list(args))),
        marker=lambda: tmp_path / "own.ready",
        tk_root=_FakeTk,
        starting=lambda root, marker, **kw: (root.run_after(), covered.append((marker, kw["text"]))),
    )
    assert hopped is (home == "install")
    if hopped:
        assert relaunched == [(program, ["--extras", "--ready-file", str(tmp_path / "own.ready")])]
        assert covered == [(tmp_path / "own.ready", "Starting setup…")]
    else:
        assert relaunched == covered == []


def test_hop_copy_failure_says_so_and_closes(tmp_path: Path, fake_dialogs: _FakeDialogs) -> None:
    errors: list[str] = []
    fake_dialogs.showerror = lambda _title, message, **_kw: errors.append(message)  # type: ignore[method-assign]
    root = _FakeTk()

    def fail() -> None:
        raise OSError(28, "No space left on device")

    hop_behind_window(
        root, fail, tmp_path / "x.ready", starting=lambda r, *_a, **_kw: r.run_after(), dialogs=lambda: fake_dialogs
    )
    assert root.destroyed and "No space left" in errors[0]


def test_hopped_window_waits_for_the_launcher_to_exit_before_extras(tmp_path: Path) -> None:
    now = [0.0]
    started: list[bool] = []
    window = SimpleNamespace(
        root=_FakeTk(),
        handoff=ReadyHandoff(tmp_path / "x.ready", clock=lambda: now[0]),
        install_button=_FakeLabel(),
        installer=SimpleNamespace(extras=True),
        start=lambda: started.append(True),
    )
    window._await_handoff = lambda: SetupWindow._await_handoff(cast(Any, window))
    window._await_handoff()
    assert started == [] and len(window.root.pending) == 1  # marker up; the launcher has not let go yet
    (tmp_path / "x.ready").unlink()  # the launcher took it and exits
    now[0] = 0.5
    window.root.run_after()
    assert started == []  # its exe unlocks within the grace
    now[0] = 2.0
    window.root.run_after()
    assert started == [True] and window.install_button.state == "normal"


def test_parse_args_takes_the_uninstall_ready_file() -> None:
    assert parse_args(["--uninstall", "--yes", "--ready-file", "C:/Temp/x.ready"]).ready_file == Path("C:/Temp/x.ready")
    assert parse_args(["--uninstall"]).ready_file is None


def test_parse_args_extras_only_alone() -> None:
    assert parse_args(["--extras", "--headless"]).extras is True
    assert parse_args([]).extras is False
    with pytest.raises(SystemExit):
        parse_args(["--extras", "--clean"])


def test_deps_command_keeps_cpu_extra_once_diarization_is_on(paths: InstallPaths) -> None:
    def extras(cmd: list[str]) -> list[str]:
        return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--extra"]

    assert extras(deps_command(paths)) == ["gui"]
    assert extras(deps_command(paths, diarization=True)) == ["gui", "cpu"]  # the --extras run itself
    set_config_value(paths.config_file, "device", "cpu")
    set_config_value(paths.config_file, "extras", "diarization")
    assert extras(deps_command(paths)) == ["gui", "cpu"]  # repair / clean keep speaker detection
    assert read_config(paths.config_file) == {"device": "cpu", "extras": "diarization"}


class _ExtrasRun:
    """An installed ``paths`` plus a recording runner / launcher for ``--extras`` runs."""

    def __init__(self, paths: InstallPaths, *, token: str | None = "hf_abc", model_output: str | None = None) -> None:
        paths.uv_exe.parent.mkdir(parents=True)
        paths.uv_exe.write_bytes(b"uv")
        paths.app_dir.mkdir(parents=True)
        if token is not None:
            paths.hf_token_file.parent.mkdir(parents=True, exist_ok=True)
            paths.hf_token_file.write_text(token, encoding="utf-8")
        self.paths = paths
        self.model_output = model_output
        self.runs: list[tuple[list[str], dict[str, str]]] = []
        self.launches: list[InstallPaths] = []
        self.states: dict[str, tuple[str, str]] = {}
        self.lock = threading.Lock()

    def runner(self, cmd: list[str], env: dict[str, str], *_a: Any) -> None:
        with self.lock:
            self.runs.append((list(cmd), dict(env)))
        if self.model_output is not None and DIARIZATION_MODEL.repo_id in cmd:
            raise CommandFailed(self.model_output.splitlines()[-1], self.model_output)

    def installer(self, **kw: Any) -> Installer:
        return Installer(
            paths=self.paths,
            env={},
            extras=True,
            runner=self.runner,
            seeder=lambda *_a: False,
            model_size=lambda *_a: None,
            launch=self.launches.append,
            **{"token_status": lambda _token: 200, **kw},
        )

    def events(self) -> Events:
        def state(key: str, value: str, message: str) -> None:
            with self.lock:
                self.states[key] = (value, message)

        return Events(progress=lambda *_a: None, state=state)


def test_extras_success_writes_flag_then_relaunches(paths: InstallPaths) -> None:
    run = _ExtrasRun(paths, token="  hf_abc\n")
    assert run.installer().run(run.events()) is True
    (deps_cmd, _), (model_cmd, model_env) = sorted(run.runs, key=lambda r: DIARIZATION_MODEL.repo_id in r[0])
    assert deps_cmd[deps_cmd.index("gui") + 1 : deps_cmd.index("gui") + 3] == ["--extra", "cpu"]
    assert model_cmd[model_cmd.index("download") + 1 :] == [DIARIZATION_MODEL.repo_id]  # full snapshot
    assert model_env["HF_TOKEN"] == "hf_abc"
    assert model_env["HF_HOME"] == str(paths.hf_home)
    assert read_config(paths.config_file).get("extras") == "diarization"
    assert run.launches == [paths]


@pytest.mark.parametrize("token", [None, " \n"], ids=["missing", "blank"])
def test_extras_without_token_fails_before_any_command(paths: InstallPaths, token: str | None) -> None:
    run = _ExtrasRun(paths, token=token)
    with pytest.raises(InstallError, match="No Hugging Face token"):
        run.installer().run(run.events())
    assert run.runs == [] and run.launches == []
    assert "extras" not in read_config(paths.config_file)


@pytest.mark.parametrize(
    "output",
    [
        "Traceback ...\nhuggingface_hub.errors.GatedRepoError: 403 Client Error.\nCannot access gated repo for url x",
        "requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api",
    ],
    ids=["403-licence", "401-token"],
)
def test_extras_auth_failure_explains_fixes_and_stays_off(paths: InstallPaths, output: str) -> None:
    run = _ExtrasRun(paths, model_output=output)
    assert run.installer().run(run.events()) is False
    state, message = run.states[DIARIZATION_MODEL.repo_id]
    assert state == "failed"
    assert "https://hf.co/pyannote/speaker-diarization-community-1" in message and "hf.co/settings/tokens" in message
    assert run.states["extras"][0] == "skipped"
    assert "extras" not in read_config(paths.config_file)
    assert run.launches == []


def test_extras_other_download_failure_keeps_its_own_message(paths: InstallPaths) -> None:
    run = _ExtrasRun(paths, model_output="OSError: [Errno 28] No space left on device")
    assert run.installer().run(run.events()) is False
    assert run.states[DIARIZATION_MODEL.repo_id] == ("failed", "OSError: [Errno 28] No space left on device")


def test_extras_waits_for_the_app_to_close_then_gives_up(paths: InstallPaths) -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    open_for = [3]

    def in_use(_p: InstallPaths) -> bool:
        open_for[0] -= 1
        return open_for[0] >= 0

    run = _ExtrasRun(paths)
    assert run.installer(in_use=in_use, sleep=sleep, clock=lambda: now[0], close_timeout=10).run(run.events())
    assert now[0] == 1.5  # three polls while the app was still closing
    run = _ExtrasRun(InstallPaths(paths.install_dir.with_name("b"), paths.config_file, windows=True))
    stuck = run.installer(in_use=lambda _p: True, sleep=sleep, clock=lambda: now[0], close_timeout=10)
    with pytest.raises(InstallError, match="Close Transcribe first"):
        stuck.run(run.events())
    assert run.runs == [] and run.launches == []


@pytest.mark.parametrize("status", [401, 403])
def test_extras_rejected_token_fails_before_any_command(paths: InstallPaths, status: int) -> None:
    run = _ExtrasRun(paths)
    checked: list[str] = []

    def token_status(token: str) -> int:
        checked.append(token)
        return status

    with pytest.raises(InstallError) as caught:
        run.installer(token_status=token_status).run(run.events())
    assert str(caught.value) == HF_AUTH_HELP
    assert checked == ["hf_abc"] and run.runs == [] and run.launches == []


@pytest.mark.parametrize("status", [200, None], ids=["ok", "offline"])
def test_extras_token_check_only_blocks_on_a_refusal(paths: InstallPaths, status: int | None) -> None:
    run = _ExtrasRun(paths)
    assert run.installer(token_status=lambda _t: status).run(run.events()) is True
    assert len(run.runs) == 2 and run.launches == [paths]


def test_hf_token_status_heads_the_gated_config_with_the_token() -> None:
    seen: list[urllib.request.Request] = []

    def answer(outcome: Any) -> Callable[..., Any]:
        def urlopen(request: urllib.request.Request, *, timeout: float) -> Any:
            seen.append(request)
            if isinstance(outcome, BaseException):
                raise outcome
            return contextlib.nullcontext(SimpleNamespace(status=outcome))

        return urlopen

    refused = urllib.error.HTTPError(HF_TOKEN_PROBE_URL, 401, "Unauthorized", Message(), None)
    assert hf_token_status("hf_abc", urlopen=answer(refused)) == 401
    assert hf_token_status("hf_abc", urlopen=answer(urllib.error.URLError("offline"))) is None
    assert hf_token_status("hf_abc", urlopen=answer(TimeoutError())) is None
    assert hf_token_status("hf_abc", urlopen=answer(200)) == 200
    request = seen[0]
    assert (request.get_method(), request.full_url) == ("HEAD", HF_TOKEN_PROBE_URL)
    assert request.get_header("Authorization") == "Bearer hf_abc"
    assert HF_TOKEN_PROBE_URL.startswith(f"https://huggingface.co/{DIARIZATION_MODEL.repo_id}/")


def test_app_env_drops_tcl_paths_into_setups_unpack_dir(tmp_path: Path) -> None:
    # Transcribe inherited these from the frozen setup and died once setup exited and deleted _MEI.
    bundle = tmp_path / "_MEI123"
    base = {"TCL_LIBRARY": str(bundle / "_tcl_data"), "TK_LIBRARY": str(bundle / "_tk_data"), "A": "1"}
    assert app_env(base, str(bundle)) == {"A": "1"}
    own = {"TCL_LIBRARY": str(tmp_path / "tcl8.6"), "A": "1"}  # a user's own Tcl stays
    assert app_env(own, str(bundle)) == own
    assert app_env(base, None) == base  # not frozen


def test_setup_opens_the_app_outside_its_venv(
    paths: InstallPaths, popen_calls: list[tuple[Any, dict[str, Any]]], popen: Callable[..., None]
) -> None:
    launch_gui(paths, popen=popen, env={"A": "1"})
    ((args, kw),) = popen_calls
    assert kw["env"] == {"A": "1"}
    assert args == [str(paths.gui_exe)]
    assert not Path(kw["cwd"]).is_relative_to(paths.venv_dir)


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""
        self.state = ""

    def config(self, **kw: Any) -> None:
        self.text = kw.get("text", self.text)
        self.state = kw.get("state", self.state)

    def pack(self, **_kw: Any) -> None:
        self.shown = True


def test_pre_task_error_stays_in_the_final_summary(paths: InstallPaths) -> None:
    def run(_events: Events) -> bool:
        raise InstallError("Close Transcribe first, then retry.")

    summary, log_link = _FakeLabel(), _FakeLabel()
    window = SimpleNamespace(
        installer=SimpleNamespace(run=run, extras=False, paths=paths),
        events=queue.Queue(),
        failure="",
        running=True,
        rows={},
        summary=summary,
        log_link=log_link,
        install_button=_FakeLabel(),
        root=_FakeTk(),
    )
    window._finished = lambda ok: SetupWindow._finished(cast(Any, window), ok)
    SetupWindow._work(cast(Any, window), Events(progress=lambda *_a: None, state=lambda *_a: None))
    SetupWindow._poll(cast(Any, window))
    assert "Close Transcribe first, then retry." in summary.text
    assert summary.text.startswith("Setup did not finish.")
    assert log_link.text == str(paths.log_file) and log_link.shown  # a clickable link to the log


@pytest.mark.parametrize("windows", [True, False])
def test_open_file_uses_the_platform_opener(tmp_path: Path, windows: bool) -> None:
    started: list[str] = []
    popened: list[list[str]] = []
    open_file(tmp_path / "setup.log", windows=windows, popen=lambda cmd: popened.append(cmd), startfile=started.append)
    log = str(tmp_path / "setup.log")
    assert (started, popened) == (([log], []) if windows else ([], [["xdg-open", log]]))


class _FakeTk:
    def __init__(self) -> None:
        self.destroyed = False
        self.pending: list[Callable[[], None]] = []

    def withdraw(self) -> None:
        pass

    def destroy(self) -> None:
        self.destroyed = True

    def after(self, _ms: int, callback: Callable[[], None]) -> None:
        self.pending.append(callback)

    def run_after(self) -> None:
        self.pending.pop(0)()

    def mainloop(self) -> None:
        pass


class _FakeDialogs:
    """tkinter.messagebox stand-in: askyesno answers ``yes``."""

    def __init__(self) -> None:
        self.yes = True

    def askyesno(self, *_a: Any, **_kw: Any) -> bool:
        return self.yes

    def showerror(self, *_a: Any, **_kw: Any) -> None:
        pass

    def showinfo(self, *_a: Any, **_kw: Any) -> None:
        pass


@pytest.fixture
def fake_dialogs() -> _FakeDialogs:
    return _FakeDialogs()


def test_uninstall_main_declined_removes_nothing(win_paths: InstallPaths, fake_dialogs: _FakeDialogs) -> None:
    fake_dialogs.yes = False
    win_paths.venv_dir.mkdir(parents=True)
    code = uninstall_main(
        win_paths,
        {},
        headless=False,
        confirmed=False,
        tk_root=_FakeTk,
        dialogs=lambda: fake_dialogs,
        winreg=_FakeReg,
        remove=lambda *_a, **_kw: pytest.fail("removed after a No"),
    )
    assert code == 1
    assert win_paths.venv_dir.is_dir()


def test_uninstall_main_hops_to_temp_copy_when_run_from_install_dir(
    win_paths: InstallPaths,
    popen_calls: list[tuple[Any, dict[str, Any]]],
    popen: Callable[..., None],
    tempdir: Callable[[], str],
) -> None:
    exe = win_paths.setup_copy.resolve()
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    code = uninstall_main(
        win_paths,
        {},
        headless=True,
        confirmed=True,
        program=lambda: exe,
        winreg=_FakeReg,
        relaunch=lambda e, args: relaunch_from_temp(e, args, popen=popen, tempdir=tempdir),
        remove=lambda *_a, **_kw: pytest.fail("the temp copy removes"),
    )
    assert code == 0
    ((args, _kw),) = popen_calls
    assert not Path(args[0]).is_relative_to(win_paths.install_dir) and {"--uninstall", "--yes"} <= set(args[1:])


def test_uninstall_main_self_deletes_its_hop_copy_either_way(tmp_path: Path, win_paths: InstallPaths) -> None:
    exe = tmp_path / "temp" / "stt-faster-setup-x" / "Transcribe-Setup.exe"
    deletes: list[Path] = []

    def locked(*_a: Any, **_kw: Any) -> None:
        raise InstallError("locked")

    def run(remove: Callable[..., None]) -> int:
        return uninstall_main(
            win_paths,
            {},
            headless=True,
            confirmed=True,
            program=lambda: exe,
            winreg=_FakeReg,
            remove=remove,
            self_delete=lambda e, _env: deletes.append(e),
        )

    assert run(locked) == 1
    assert deletes == [exe]  # a retry starts from the Start menu again
    assert run(lambda *_a, **_kw: None) == 0
    assert deletes == [exe, exe]


@pytest.mark.parametrize("given", [None, "setup.ready"])
def test_uninstall_main_windowed_hop_forwards_or_covers_the_ready_file(
    win_paths: InstallPaths, tmp_path: Path, given: str | None
) -> None:
    ready_file = None if given is None else tmp_path / given
    relaunched: list[list[str]] = []
    covered: list[Path] = []
    code = uninstall_main(
        win_paths,
        {},
        headless=False,
        confirmed=True,
        ready_file=ready_file,
        tk_root=_FakeTk,
        winreg=_FakeReg,
        program=lambda: win_paths.setup_copy.resolve(),
        relaunch=lambda _exe, args: relaunched.append(list(args)),
        remove=lambda *_a, **_kw: pytest.fail("the temp copy removes"),
        marker=lambda: tmp_path / "own.ready",
        starting=lambda root, marker, **_kw: (root.run_after(), covered.append(marker)),
    )
    expected = ready_file or tmp_path / "own.ready"
    assert code == 0
    assert relaunched == [["--uninstall", "--yes", "--ready-file", str(expected)]]
    assert covered == ([] if given else [expected])  # the setup window already covers the hop


@pytest.mark.parametrize("ok", [True, False])
def test_uninstall_main_windowed_removes_in_its_window(win_paths: InstallPaths, tmp_path: Path, ok: bool) -> None:
    exe = tmp_path / "Downloads" / "Transcribe-Setup.exe"
    ready = tmp_path / "setup.ready"
    steps: list[str] = []
    deletes: list[Path] = []

    def remove(_paths: InstallPaths, _env: Any, *, reg: Any, step: Callable[[str], None]) -> None:
        assert isinstance(reg, _FakeReg)
        step("Removing settings…")

    def window(_root: Any, run: Callable[[Callable[[str], None]], None], *, ready_file: Path | None) -> Any:
        assert ready_file == ready
        run(steps.append)
        return SimpleNamespace(ok=ok)

    code = uninstall_main(
        win_paths,
        {},
        headless=False,
        confirmed=True,
        ready_file=ready,
        tk_root=_FakeTk,
        winreg=_FakeReg,
        program=lambda: exe,
        remove=remove,
        self_delete=lambda e, _env: deletes.append(e),
        window=window,
    )
    assert code == (0 if ok else 1)
    assert steps == ["Removing settings…"]
    assert deletes == [exe]  # a no-op outside a hop folder


def test_wait_for_marker_closes_on_the_marker_or_the_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "x.ready"
    now = [0.0]
    root = _FakeTk()
    wait_for_marker(root, marker, timeout=20, clock=lambda: now[0])
    assert not root.destroyed
    marker.touch()
    root.run_after()
    assert root.destroyed and not marker.exists()

    root = _FakeTk()
    wait_for_marker(root, marker, timeout=20, clock=lambda: now[0])
    now[0] = 19.9
    root.run_after()
    assert not root.destroyed
    now[0] = 20.0
    root.run_after()
    assert root.destroyed

    stuck = tmp_path / "stuck.ready"
    stuck.mkdir()  # taken, but its unlink raises: still close
    root = _FakeTk()
    wait_for_marker(root, stuck, timeout=20, clock=lambda: now[0])
    assert root.destroyed


def test_ready_handoff_waits_for_the_waiter_to_let_go(tmp_path: Path) -> None:
    marker = tmp_path / "setup.ready"
    now = [0.0]
    handoff = ReadyHandoff(marker, clock=lambda: now[0])
    assert not handoff.done() and marker.exists()  # signalled: the waiter may close
    marker.unlink()  # the waiter took it at t=0.5 and is exiting
    now[0] = 0.5
    assert not handoff.done()
    now[0] = 1.4
    assert not handoff.done()
    now[0] = 1.5
    assert handoff.done()

    handoff = ReadyHandoff(marker, clock=lambda: now[0])  # nobody takes it (the waiter already timed out)
    now[0] = 10.0
    assert not handoff.done()
    now[0] = 14.9
    assert not handoff.done() and marker.exists()
    now[0] = 15.0
    assert handoff.done() and not marker.exists()

    handoff = ReadyHandoff(tmp_path / "gone" / "x.ready", clock=lambda: now[0])  # can't signal: only the cap
    assert not handoff.done()
    now[0] = 19.9
    assert not handoff.done()
    now[0] = 20.0
    assert handoff.done()

    stuck = tmp_path / "stuck.ready"
    stuck.mkdir()  # unlink raises: done() must still return, or _poll's chain dies with it
    handoff = ReadyHandoff(stuck, clock=lambda: now[0])
    assert not handoff.done()
    now[0] = 25.0
    assert handoff.done()


@pytest.mark.parametrize("error", [None, InstallError("locked"), OSError("boom")])
def test_uninstall_window_shows_steps_then_the_result(tmp_path: Path, error: Exception | None) -> None:
    def remove(step: Callable[[str], None]) -> None:
        step("Removing settings…")
        if error is not None:
            raise error

    status, close = _FakeLabel(), _FakeLabel()
    spawned: list[Callable[[], None]] = []
    let_go = [False]
    window = SimpleNamespace(
        root=_FakeTk(),
        remove=remove,
        handoff=SimpleNamespace(done=lambda: let_go[0]),
        spawn=spawned.append,
        started=False,
        events=queue.Queue(),
        done=False,
        ok=False,
        status=status,
        bar=SimpleNamespace(stop=lambda: None, pack_forget=lambda: None),
        close_button=close,
    )
    window._poll = lambda: UninstallWindow._poll(cast(Any, window))
    window._work = lambda: UninstallWindow._work(cast(Any, window))
    UninstallWindow._poll(cast(Any, window))
    assert spawned == []  # the setup window may still hold the install dir
    let_go[0] = True
    window.events.put(("step", "Removing app files and models…"))
    window.root.run_after()
    assert spawned == [window._work] and status.text == "Removing app files and models…"
    UninstallWindow._on_close(cast(Any, window))
    assert not window.root.destroyed  # X is ignored while removing

    UninstallWindow._work(cast(Any, window))
    window.root.run_after()
    assert window.done and window.ok is (error is None)
    assert status.text == ("Transcribe was removed." if error is None else f"Uninstall did not finish:\n{error}")
    assert close.shown
    UninstallWindow._on_close(cast(Any, window))
    assert window.root.destroyed


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


# captured from `hf download Systran/faster-whisper-tiny` (huggingface_hub 1.2.1, TQDM_POSITION=-1), ANSI stripped
HF_BAR_LINES = [
    ("Downloading (incomplete total...): 0.00B [00:00, ?B/s]", None, "0.00B"),
    ("Fetching 6 files:  17%|█▋        | 1/6 [00:00<00:01,  3.58it/s]", ..., ""),
    (
        "Downloading (incomplete total...):  15%|█▍        | 11.2M/75.5M [00:03<00:18, 3.50MB/s]",
        11.2 / 75.5,
        "11.2M/75.5M",
    ),
    ("Downloading (incomplete total...):   0%|          | 3.47k/75.5M [00:00<2:55:39, 7.17kB/s]", 3470 / 75.5e6, ""),
    ("Downloading (incomplete total...): 78.2MB [00:09, 9.58MB/s]", None, "78.2MB"),
    ("Download complete: : 78.2MB [00:09, 8.67MB/s]", None, "78.2MB"),
    ("Downloading numpy (15.2MiB)", ..., ""),  # uv's own lines never read as the hf bar
]


@pytest.mark.parametrize(("line", "fraction", "text"), HF_BAR_LINES)
def test_download_progress_reads_hf_byte_bar(line: str, fraction: float | None, text: str) -> None:
    parsed = download_progress(line)
    if fraction is ...:
        assert parsed is None
        return
    assert parsed is not None
    assert parsed[0] == (None if fraction is None else pytest.approx(fraction))
    if text:
        assert parsed[1] == text


def test_run_process_holds_byte_bar_and_ticks_elapsed() -> None:
    bar = "Downloading (incomplete total...):  15%| | 11.2M/75.5M [00:03<00:18, 3.50MB/s]"
    script = f"import sys, time; sys.stdout.write('\\x1b[A{bar}\\x1b[A\\n'); sys.stdout.flush(); time.sleep(2.5)"
    reports: list[tuple[float | None, str]] = []
    run_process(
        [sys.executable, "-c", script], os.environ, lambda f, t: reports.append((f, t)), threading.Event(), lambda: 0.9
    )
    held = [(f, t) for f, t in reports if t.startswith("11.2M/75.5M · 0:0")]
    assert held and all(f == pytest.approx(11.2 / 75.5) for f, _t in held)  # the bar, not poll()'s dir size
    assert len({t for _f, t in held}) >= 2  # the elapsed ticker moves while hf prints nothing


@pytest.mark.parametrize("moving", [False, True], ids=["stuck-bar", "moving-bar"])
def test_run_process_stops_a_stalled_download(moving: bool, tmp_path: Path) -> None:
    # hf re-printing an unchanged bar is still a stall; a bar that moves is not
    script = (
        "import os, sys, time\n"
        f"open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid()))\n"
        "for i in range(10):\n"
        f"    n = i if {moving} else 0\n"
        "    print(f'Downloading (incomplete total...):  1%| | {n + 1}.00M/75.5M [00:0{i}, {i}.50MB/s]', flush=True)\n"
        "    time.sleep(0.25)\n"
    )
    started = time.monotonic()
    run = lambda: run_process(  # noqa: E731
        [sys.executable, "-c", script], os.environ, lambda *_a: None, threading.Event(), None, 1.0
    )
    if moving:
        run()
        return
    with pytest.raises(Stalled, match="no progress for 0:01"):
        run()
    assert time.monotonic() - started < 2.4  # stopped before the script's own 2.5 s end
    assert _gone(int((tmp_path / "pid").read_text()))  # killed, not left to finish on its own


def test_stalled_model_retries_once_without_xet_after_clearing_the_partial(paths: InstallPaths) -> None:
    spec = MODELS[0]
    part = model_cache_dir(paths.hf_home / "hub", spec.repo_id) / "blobs" / "abc.incomplete"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\0" * 8)
    runs: list[tuple[str | None, bool]] = []
    shown: list[str] = []

    def runner(_cmd: list[str], env: dict[str, str], report: Callable[[float | None, str], None], *_a: Any) -> None:
        runs.append((env.get("HF_HUB_DISABLE_XET"), part.exists()))
        if len(runs) == 1:
            raise Stalled("Download stalled")
        report(None, "")
        report(0.1, "8.00M/75.5M · 0:03")

    installer = Installer(paths=paths, env={}, runner=runner, seeder=lambda *_a: False, model_size=lambda *_a: None)
    installer.fetch_model(spec, lambda _f, text: shown.append(text))
    assert runs == [(None, True), ("1", False)]  # the HTTP retry never resumes xet's holed partial
    assert shown[-2:] == [RETRY_NOTE, f"{RETRY_NOTE} · 8.00M/75.5M · 0:03"]  # why the bar restarted stays visible


def test_unlink_when_released_waits_out_a_dying_holder(tmp_path: Path) -> None:
    held = [2]
    sleeps: list[float] = []

    def unlink(_p: Path) -> None:
        if held[0]:
            held[0] -= 1
            raise PermissionError("[WinError 32] used by another process")

    unlink_when_released(tmp_path / "a.incomplete", unlink=unlink, sleep=sleeps.append)
    assert held == [0] and sleeps == [0.25, 0.25]
    held[0] = 99
    with pytest.raises(PermissionError):
        unlink_when_released(tmp_path / "a.incomplete", attempts=3, unlink=unlink, sleep=sleeps.append)


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


def _smi(stdout: str, returncode: int = 0) -> Callable[..., Any]:
    def run(cmd: list[str], **_kw: Any) -> SimpleNamespace:
        run.cmd = cmd  # type: ignore[attr-defined]
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    return run


def test_detect_gpu_reads_first_gpu_from_system32_nvidia_smi() -> None:
    run = _smi("NVIDIA GeForce RTX 3060, Laptop, 12288, 560.94\nTesla T4, 15360, 560.94\n")
    info = detect_gpu({"SystemRoot": "C:\\Windows"}, run=run)
    assert info == GpuInfo("NVIDIA GeForce RTX 3060, Laptop", 12288, (560, 94))
    assert Path(run.cmd[0]).parts[-2:] == ("System32", "nvidia-smi.exe")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [("", 0), ("RTX 3060, [N/A], 560.94\n", 0), ("RTX 3060, 12288, 560.94\n", 9)],
    ids=["no-output", "unparsable", "exit-code"],
)
def test_detect_gpu_any_doubt_is_none(stdout: str, returncode: int) -> None:
    assert detect_gpu({}, run=_smi(stdout, returncode)) is None


def test_detect_gpu_missing_tool_is_none() -> None:
    def run(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    assert detect_gpu({}, run=run) is None


@pytest.mark.parametrize(
    ("info", "capable"),
    [
        (GpuInfo("RTX 3050", 4096, (528, 33)), True),
        (GpuInfo("RTX 3050", 4096, (528, 24)), False),  # below the CUDA 12 driver floor
        (GpuInfo("GTX 1050", 3072, (560, 94)), False),  # too little VRAM
        (None, False),
    ],
)
def test_gpu_capable_needs_driver_and_vram(info: GpuInfo | None, capable: bool) -> None:
    assert gpu_capable(info) is capable
    assert ("CPU" in gpu_summary(info)) is not capable


def test_save_device_pick_overwrites_none_keeps(tmp_path: Path) -> None:
    config = tmp_path / "config"
    save_device(config, None)  # an old install without a device line
    assert read_config(config)["device"] == "cpu"
    save_device(config, True)  # new / clean install, or a same-window retry after unticking
    assert read_config(config)["device"] == "cuda"
    save_device(config, None)  # repair keeps it
    assert read_config(config)["device"] == "cuda"
    save_device(config, False)
    assert read_config(config)["device"] == "cpu"


def test_install_run_saves_gpu_pick_before_deps_sync(paths: InstallPaths, tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\n")
    for exe in (paths.uv_exe, *(paths.ffmpeg_bin / name for name in FFMPEG_BINARIES)):
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"x")
    set_config_value(paths.config_file, "device", "cpu")  # a failed earlier attempt already saved one
    runs: list[list[str]] = []

    def runner(cmd: list[str], *_a: Any) -> None:
        runs.append(list(cmd))
        if "sync" in cmd:  # the deps sync creates the venv
            paths.gui_exe.parent.mkdir(parents=True, exist_ok=True)
            paths.gui_exe.write_bytes(b"exe")

    installer = Installer(
        paths=paths,
        source=str(source),
        env={},
        runner=runner,
        seeder=lambda *_a: False,
        model_size=lambda *_a: None,
        in_use=lambda _p: False,
        gpu=True,
    )
    assert installer.run(Events(progress=lambda *_a: None, state=lambda *_a: None)) is True
    (sync,) = [cmd for cmd in runs if "sync" in cmd]
    assert sync[sync.index("gui") + 1 :][:2] == ["--extra", "gpu-win"]
    assert read_config(paths.config_file)["device"] == "cuda"


def test_deps_command_adds_gpu_extra_for_cuda_device(paths: InstallPaths) -> None:
    def extras(cmd: list[str]) -> list[str]:
        return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--extra"]

    set_config_value(paths.config_file, "device", "cuda")
    assert extras(deps_command(paths, diarization=True)) == ["gui", "cpu", "gpu-win"]  # --extras keeps GPU mode
    set_config_value(paths.config_file, "device", "cpu")
    assert extras(deps_command(paths)) == ["gui"]


def test_headless_gpu_detects_on_windows_and_moves_only_legacy_cpu(paths: InstallPaths, tmp_path: Path) -> None:
    calls: list[object] = []

    def detect(env: object) -> GpuInfo:
        calls.append(env)
        return GpuInfo("RTX 3060", 12288, (560, 94))

    def headless(p: InstallPaths, **kw: Any) -> bool | None:
        return headless_gpu(Installer(paths=p, env={}, detect=detect, clean=kw.pop("clean", False)), **kw)

    assert headless(paths, cpu=False) is True
    assert headless(paths, cpu=True) is False
    assert headless(InstallPaths(tmp_path / "l", tmp_path / "l" / "config", windows=False), cpu=False) is None
    paths.uv_exe.parent.mkdir(parents=True)
    paths.uv_exe.write_bytes(b"uv")  # installed
    set_config_value(paths.config_file, "device", "cpu")
    _write_app_version(paths, "1.1.0")
    assert headless(paths, cpu=False) is True  # repair: CPU-only v1.1.0 wrote cpu unasked -> the GPU
    _write_app_version(paths, "1.2.0")
    assert headless(paths, cpu=False) is False  # a GPU-mode release's cpu was a pick (or the app's fallback)
    assert headless(paths, cpu=False, clean=True) is True
    assert headless(paths, cpu=True) is False  # --cpu always wins
    assert len(calls) == 4
    assert parse_args(["--cpu"]).cpu is True


def _write_app_version(paths: InstallPaths, version: str) -> None:
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    (paths.app_dir / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')


@pytest.mark.parametrize(
    ("mode", "installed", "saved", "version", "capable", "ticked"),
    [
        ("repair", True, "cpu", "1.1.0", True, True),  # CPU-only v1.1.0 wrote cpu unasked: GPU hosts start ticked
        ("repair", True, "cpu", "1.1.0", False, False),
        ("repair", True, "cpu", "1.2.0", True, False),  # a GPU-mode release's cpu is a pick or the app's fallback
        ("repair", True, "cpu", "", True, False),  # unreadable version: keep what is saved
        ("repair", True, "cuda", "1.2.1", False, True),
        ("clean", True, "cpu", "1.2.1", True, True),  # re-picks from detection
        ("repair", False, None, "", True, True),
    ],
)
def test_gpu_checkbox_starts_from_saved_device_on_repair(
    paths: InstallPaths, mode: str, installed: bool, saved: str, version: str, capable: bool, ticked: bool
) -> None:
    if installed:
        paths.gui_exe.parent.mkdir(parents=True)
        paths.gui_exe.write_bytes(b"exe")
    if saved:
        set_config_value(paths.config_file, "device", saved)
    if version:
        _write_app_version(paths, version)
    configured: dict[str, bool] = {}
    window = SimpleNamespace(
        mode=SimpleNamespace(get=lambda: mode),
        installer=SimpleNamespace(paths=paths),
        gpu_info=GpuInfo("RTX 3060", 12288 if capable else 2048, (560, 94)),
        use_gpu=SimpleNamespace(set=lambda v: configured.update(ticked=v)),
    )
    SetupWindow._sync_gpu_check(cast(Any, window))
    assert configured == {"ticked": ticked}


def test_start_saves_the_editable_checkbox_on_repair(paths: InstallPaths) -> None:
    installer = Installer(paths=paths, env={})
    window = SimpleNamespace(
        mode=SimpleNamespace(get=lambda: "repair"),
        installer=installer,
        use_gpu=SimpleNamespace(get=lambda: True),
        gpu_check=object(),
        install_button=SimpleNamespace(config=lambda **_kw: None),
        summary=SimpleNamespace(config=lambda **_kw: None),
        launch_button=_FakeLabel(),
        log_link=SimpleNamespace(pack_forget=lambda: None),
        rows={},
        tails={},
        events=queue.Queue(),
        root=SimpleNamespace(after=lambda *_a: None),
        _work=lambda _events: None,
        _poll=lambda: None,
    )
    window.launch_button.config(state="normal")  # an existing install opens without a repair first
    SetupWindow.start(cast(Any, window))
    assert window.launch_button.state == "disabled"  # never open the app while the run replaces its venv
    assert installer.gpu is True  # repair saves the (editable) checkbox
    window.gpu_check = None
    SetupWindow.start(cast(Any, window))
    assert installer.gpu is None  # no checkbox shown: keep the saved device


@pytest.mark.parametrize(
    ("pixels", "shown"),
    [
        (0, "0123456789"[-40:]),  # not laid out yet
        (10, "0123456789"),  # fits as is
        (5, "…6789"),  # keeps the newest end
        (1_000, "0123456789"),  # a wider window shows it all
    ],
)
def test_fit_tail_keeps_the_end_that_fits(pixels: int, shown: str) -> None:
    assert fit_tail("0123456789", pixels, len) == shown
