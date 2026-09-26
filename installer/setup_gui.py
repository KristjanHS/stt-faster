"""Transcribe-Setup: per-user bootstrap installer for the stt-faster GUI (stdlib only).

Fetches uv, the app source, its dependencies, the models and ffmpeg into ``%LOCALAPPDATA%\\stt-faster``
with no admin rights. PyInstaller freezes this file into ``Transcribe-Setup.exe``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import platform
import queue
import re
import shutil
import signal
import subprocess  # nosec B404 - runs uv / powershell with fixed argument lists
import sys
import tarfile
import tempfile
import threading
import time
import tkinter as tk
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import Any

LOGGER = logging.getLogger("setup_gui")

APP_NAME = "stt-faster"
GITHUB_REPO = "KristjanHS/stt-faster"
SHORTCUT_NAME = "Transcribe"
SETUP_EXE_NAME = "Transcribe-Setup.exe"
UV_VERSION = "0.12.19"
# The app's uv.lock versions: an unpinned typer-slim no longer pulls in click, which the `hf` CLI imports.
HF_TOOL_PINS = ("huggingface_hub==1.2.1", "typer-slim==0.20.0", "click==8.3.1", "hf-xet==1.2.0")
PYTHON_VERSION = "3.12"
FFMPEG_VERSION = "9.0.2"
FFMPEG_URL = (
    f"https://github.com/GyanD/codexffmpeg/releases/download/{FFMPEG_VERSION}/"
    f"ffmpeg-{FFMPEG_VERSION}-essentials_build.zip"
)
FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe")  # inspect_audio needs ffprobe even when no preprocessing runs
USER_AGENT = f"{APP_NAME}-setup"
SOURCE_IGNORE_DIRS = frozenset({"__pycache__", "node_modules", "logs", "reports", "build", "dist"})
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash from a --windowed exe
MIN_GPU_DRIVER = (528, 33)  # CUDA 12.0 on Windows; newer cuBLAS 12.x runs via minor-version compatibility
MIN_GPU_VRAM_MIB = 4000  # ~4 GB; a misjudged GPU falls back to CPU (model load or the GUI's retry)
GPU_EXTRA_SIZE = "1.2 GB"  # gpu-win wheels: cuBLAS + cuDNN 9.1
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}"  # HKCU: Apps & features, no admin
UNINSTALL_TEMP_PREFIX = f"{APP_NAME}-uninstall-"
SHORTCUT_DIRS_PS = "@([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))"


class InstallError(RuntimeError):
    """A step failed with a message fit to show the user."""


class Cancelled(InstallError):
    pass


class CommandFailed(InstallError):
    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output  # the command's last output lines


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    include: tuple[str, ...] = ()


# Estonian (et-large) and English (turbo) GUI profiles, see backend/model_config.py.
MODELS = (
    ModelSpec("TalTechNLP/whisper-large-v3-turbo-et-verbatim", ("ct2/*",)),
    ModelSpec("Systran/faster-distil-whisper-large-v3"),
)
DIARIZATION_MODEL = ModelSpec("pyannote/speaker-diarization-community-1")  # gated: needs the user's token
HF_AUTH_ERROR = re.compile(r"\b40[13]\b|gated repo|unauthori[sz]ed|forbidden|invalid credentials", re.IGNORECASE)
HF_AUTH_HELP = (
    "Hugging Face refused the speaker model (401/403). To fix: "
    "1) sign in and accept the licence at https://hf.co/pyannote/speaker-diarization-community-1; "
    "2) create a read token at https://hf.co/settings/tokens; "
    "3) paste that token in Transcribe and try again."
)


@dataclass(frozen=True)
class InstallPaths:
    install_dir: Path
    config_file: Path
    windows: bool

    @property
    def uv_dir(self) -> Path:
        return self.install_dir / "uv"

    @property
    def uv_exe(self) -> Path:
        return self.uv_dir / ("uv.exe" if self.windows else "uv")

    @property
    def app_dir(self) -> Path:
        return self.install_dir / "app"

    @property
    def venv_dir(self) -> Path:
        return self.install_dir / ".venv"

    @property
    def ffmpeg_bin(self) -> Path:
        return self.install_dir / "ffmpeg" / "bin"

    @property
    def hf_home(self) -> Path:
        return self.install_dir / "hf"

    @property
    def uv_cache(self) -> Path:
        return self.install_dir / "uv-cache"

    @property
    def uv_tools(self) -> Path:
        return self.install_dir / "uv-tools"

    @property
    def tmp_dir(self) -> Path:
        return self.install_dir / "tmp"

    @property
    def python_dir(self) -> Path:
        return self.install_dir / "python"

    @property
    def gui_exe(self) -> Path:
        if self.windows:
            return self.venv_dir / "Scripts" / "stt-faster-gui.exe"
        return self.venv_dir / "bin" / "stt-faster-gui"

    @property
    def setup_copy(self) -> Path:
        return self.install_dir / SETUP_EXE_NAME

    @property
    def log_file(self) -> Path:
        return self.install_dir / "logs" / "setup.log"

    @property
    def hf_token_file(self) -> Path:
        return self.config_file.parent / "hf_token"  # written by the app before it starts --extras


def default_install_paths(
    env: Mapping[str, str] | None = None, plat: str | None = None, home: Path | None = None
) -> InstallPaths:
    """Mirror of ``backend.gui.default_app_paths`` (this file cannot import the app)."""
    env = os.environ if env is None else env
    plat = sys.platform if plat is None else plat
    home = Path.home() if home is None else home

    def env_dir(key: str, default: Path) -> Path:
        value = env.get(key, "")
        return Path(value) if value else default

    windows = plat == "win32"
    if windows:
        install = env_dir("LOCALAPPDATA", home / "AppData" / "Local") / APP_NAME
        config_dir = env_dir("APPDATA", home / "AppData" / "Roaming") / APP_NAME
    else:
        install = env_dir("XDG_DATA_HOME", home / ".local" / "share") / APP_NAME
        config_dir = env_dir("XDG_CONFIG_HOME", home / ".config") / APP_NAME
    return InstallPaths(install_dir=install, config_file=config_dir / "config", windows=windows)


def uv_asset(plat: str, machine: str) -> str:
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine.lower())
    if arch is None:
        raise InstallError(f"Unsupported processor: {machine}")
    if plat == "win32":
        return f"uv-{arch}-pc-windows-msvc.zip"
    if plat.startswith("linux"):
        return f"uv-{arch}-unknown-linux-gnu.tar.gz"
    if plat == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    raise InstallError(f"Unsupported platform: {plat}")


def uv_url(asset: str, version: str = UV_VERSION) -> str:
    return f"https://github.com/astral-sh/uv/releases/download/{version}/{asset}"


def hf_hub_cache(env: Mapping[str, str], home: Path) -> Path:
    """Same resolution order as huggingface_hub, so the app finds what the installer downloaded."""
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    xdg = env.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else home / ".cache") / "huggingface" / "hub"


def model_cache_dir(hub_cache: Path, repo_id: str) -> Path:
    return hub_cache / f"models--{repo_id.replace('/', '--')}"


def dir_size(path: Path) -> int:
    """Bytes of regular files under ``path``; symlinks are skipped so HF snapshot links don't double-count."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file = Path(root) / name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except OSError:
                    continue
    return total


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def expected_model_size(spec: ModelSpec, fetch_json: Callable[[str], Any]) -> int | None:
    """Total bytes ``hf download`` will fetch, from the HF tree API; None when unknown."""
    url = f"https://huggingface.co/api/models/{spec.repo_id}/tree/main?recursive=true"
    try:
        entries = fetch_json(url)
    except (OSError, ValueError) as error:
        LOGGER.info("Model size lookup failed for %s: %s", spec.repo_id, error)
        return None
    sizes = [
        int(entry.get("lfs", {}).get("size") or entry.get("size") or 0)
        for entry in entries
        if entry.get("type") == "file" and _matches(entry.get("path", ""), spec.include)
    ]
    return sum(sizes) or None


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310 - fixed https URLs
        return json.loads(response.read().decode("utf-8"))


HF_TOKEN_PROBE_URL = f"https://huggingface.co/{DIARIZATION_MODEL.repo_id}/resolve/main/config.yaml"


def hf_token_status(token: str, *, urlopen: Callable[..., Any] = urllib.request.urlopen) -> int | None:
    """HTTP status of an authenticated HEAD on the gated model's config; None when no answer came back."""
    headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
    request = urllib.request.Request(HF_TOKEN_PROBE_URL, headers=headers, method="HEAD")
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310  # nosec B310 - fixed https URL
            return int(response.status)
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:  # URLError, timeouts: offline here says nothing about the token
        return None


def latest_release_zip(fetch: Callable[[str], Any] = fetch_json, repo: str = GITHUB_REPO) -> str:
    try:
        release = fetch(f"https://api.github.com/repos/{repo}/releases/latest")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            message = f"No release of {repo} is published yet. Rerun with --source <url|zip|folder>."
            raise InstallError(message) from None
        raise InstallError(f"Could not look up the latest release: {error}") from None
    except OSError as error:
        raise InstallError(f"Could not reach GitHub: {error}") from None
    return str(release["zipball_url"])


def download(
    url: str,
    dest: Path,
    on_progress: Callable[[int, int | None], None],
    cancel: threading.Event,
    chunk: int = 1 << 20,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:  # noqa: S310  # nosec B310
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            done = 0
            while block := response.read(chunk):
                if cancel.is_set():
                    raise Cancelled("Cancelled")
                out.write(block)
                done += len(block)
                on_progress(done, total)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise InstallError(f"Download failed ({url}): {error}") from None
    except Cancelled:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)


def extract_zip_stripped(archive: Path, dest: Path) -> None:
    """Extract a GitHub source zip, dropping its single ``<repo>-<ref>/`` top-level folder."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.strip("/")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        root = dest.resolve()
        for info in zf.infolist():
            parts = Path(info.filename).parts[1:] if strip else Path(info.filename).parts
            if not parts or info.is_dir():
                continue
            # Drive (`C:`) and rooted (`/`, `\`) parts escape dest on Windows even when not is_absolute().
            unsafe = any(p == ".." or ":" in p or p.startswith(("/", "\\")) for p in parts)
            target = dest.joinpath(*parts)
            if unsafe or not target.resolve().is_relative_to(root):
                raise InstallError(f"Unsafe path in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def extract_named(archive: Path, names: Sequence[str], dest_dir: Path) -> None:
    """Pull the named files (matched by basename, anywhere in the archive) flat into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                base = info.filename.rsplit("/", 1)[-1]
                if base in names and not info.is_dir():
                    with zf.open(info) as src, (dest_dir / base).open("wb") as out:
                        shutil.copyfileobj(src, out)
                    found.add(base)
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                base = member.name.rsplit("/", 1)[-1]
                source = tf.extractfile(member) if member.isfile() and base in names else None
                if source is not None:
                    with source, (dest_dir / base).open("wb") as out:
                        shutil.copyfileobj(source, out)
                    (dest_dir / base).chmod(0o755)
                    found.add(base)
    missing = sorted(set(names) - found)
    if missing:
        raise InstallError(f"{archive.name} is missing {', '.join(missing)}")


def swap_in(staged: Path, target: Path, *, rename: Callable[[Path, Path], Any] = Path.rename) -> None:
    """Replace ``target`` with ``staged`` so a failed fetch never leaves a half-written app dir."""
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        rename(target, old)
    try:
        rename(staged, target)
    except OSError:
        if old.exists():  # put the working install back
            rename(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)


def source_ignore(directory: str, names: list[str]) -> set[str]:
    """copytree filter for a folder source: skip dot-dirs (.git, .venv, caches) and build/log output."""
    return {n for n in names if (n.startswith(".") or n in SOURCE_IGNORE_DIRS) and (Path(directory) / n).is_dir()}


def classify_source(source: str) -> str:
    if source.startswith(("https://", "http://")):
        return "url"
    path = Path(source)
    if path.is_dir():
        return "dir"
    if path.is_file() and zipfile.is_zipfile(path):
        return "zip"
    raise InstallError(f"--source must be a URL, a .zip file or a folder: {source}")


def deps_command(paths: InstallPaths, *, diarization: bool = False) -> list[str]:
    pin = paths.app_dir / ".python-version"  # UV_NO_CONFIG also skips .python-version discovery
    python = pin.read_text(encoding="utf-8").strip() if pin.is_file() else PYTHON_VERSION
    # a full sync drops unlisted extras: once speaker detection / GPU mode is on, every repair keeps it
    config = read_config(paths.config_file)
    diarization = diarization or config.get("extras") == "diarization"
    return [
        str(paths.uv_exe),
        "sync",
        "--frozen",
        "--no-dev",
        "--extra",
        "gui",
        *(("--extra", "cpu") if diarization else ()),
        *(("--extra", "gpu-win") if config.get("device") == "cuda" else ()),
        "--python",
        python,
        "--python-preference",
        "only-managed",  # never pick a stray system/Store Python
        "--project",
        str(paths.app_dir),
    ]


def isolated_env(base: Mapping[str, str], paths: InstallPaths) -> dict[str, str]:
    """uv's cache + Pythons and every HF cache inside the install dir; inherited values never win."""
    hf = paths.hf_home
    return {
        **base,
        "UV_CACHE_DIR": str(paths.uv_cache),
        "UV_PYTHON_INSTALL_DIR": str(paths.python_dir),
        "UV_PYTHON_INSTALL_BIN": "0",  # no python.exe shim in ~/.local/bin (`uv help python install`)
        "UV_PYTHON_INSTALL_REGISTRY": "0",  # no PEP 514 entry in HKCU
        "UV_NO_CONFIG": "1",  # a user's %APPDATA%\uv\uv.toml never steers (or breaks) the install
        "UV_TOOL_DIR": str(paths.uv_tools),  # `uv tool run` otherwise creates %APPDATA%\uv\tools
        **dict.fromkeys(("TMPDIR", "TEMP", "TMP"), str(paths.tmp_dir)),  # uv's lock files outlive the run
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
        "HF_XET_CACHE": str(hf / "xet"),
    }


def deps_env(base: Mapping[str, str], paths: InstallPaths) -> dict[str, str]:
    # The venv lives beside app/ so a repair can replace the source without touching it.
    return {**isolated_env(base, paths), "UV_PROJECT_ENVIRONMENT": str(paths.venv_dir)}


def seed_model_cache(legacy: Path, target: Path, *, copytree: Callable[..., Any] = shutil.copytree) -> bool:
    """Copy (never move) a model an older install left in the shared HF cache; ``hf download`` then verifies it."""
    if target.exists() or not legacy.is_dir() or legacy.resolve() == target.resolve():
        return False
    staged = target.with_name(target.name + ".seed")
    shutil.rmtree(staged, ignore_errors=True)
    try:  # staged + rename: an interrupted copy must never pass for a complete snapshot
        copytree(legacy, staged, symlinks=True)
        staged.rename(target)
    except OSError as error:
        LOGGER.info("Could not reuse %s (%s); downloading instead", legacy, error)
        shutil.rmtree(staged, ignore_errors=True)
        return False
    return True


def model_command(paths: InstallPaths, spec: ModelSpec, *, force: bool) -> list[str]:
    hub, *extras = HF_TOOL_PINS
    cmd = [str(paths.uv_exe), "tool", "run", "--python-preference", "only-managed", "--python", PYTHON_VERSION]
    cmd += ["--from", hub, *(arg for pin in extras for arg in ("--with", pin)), "hf", "download", spec.repo_id]
    for pattern in spec.include:
        cmd += ["--include", pattern]
    if force:
        cmd.append("--force-download")
    return cmd


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def shortcut_script(target: Path, name: str = SHORTCUT_NAME) -> str:
    """PowerShell creating per-user Desktop + Start-menu shortcuts (WScript.Shell; no admin)."""
    return (
        "$s = New-Object -ComObject WScript.Shell; "
        f"foreach ($d in {SHORTCUT_DIRS_PS}) {{ "
        f"$l = $s.CreateShortcut((Join-Path $d {_ps_quote(name + '.lnk')})); "
        f"$l.TargetPath = {_ps_quote(str(target))}; "
        f"$l.WorkingDirectory = {_ps_quote(str(target.parent))}; "
        "$l.Save() }"
    )


def remove_shortcuts_script(name: str = SHORTCUT_NAME) -> str:
    return (
        f"foreach ($d in {SHORTCUT_DIRS_PS}) {{ "
        f"Remove-Item -LiteralPath (Join-Path $d {_ps_quote(name + '.lnk')}) -Force -ErrorAction SilentlyContinue }}"
    )


def app_version(app_dir: Path) -> str:
    try:
        return str(tomllib.loads((app_dir / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, ValueError):
        return ""


def register_uninstall(paths: InstallPaths, reg: Any) -> None:
    """Apps & features entry under HKCU (``reg`` = the ``winreg`` module; injected for tests)."""
    values = {
        "DisplayName": SHORTCUT_NAME,
        "DisplayVersion": app_version(paths.app_dir),
        "Publisher": APP_NAME,
        "InstallLocation": str(paths.install_dir),
        "DisplayIcon": str(paths.setup_copy),
        "UninstallString": f'"{paths.setup_copy}" --uninstall',
    }
    with reg.CreateKey(reg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
        for name, value in values.items():
            reg.SetValueEx(key, name, 0, reg.REG_SZ, value)
        for name in ("NoModify", "NoRepair"):
            reg.SetValueEx(key, name, 0, reg.REG_DWORD, 1)


def unregister_uninstall(reg: Any) -> None:
    try:
        reg.DeleteKey(reg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except FileNotFoundError:
        pass


def read_config(config_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if config_file.is_file():
        for line in config_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip():
                values[key.strip()] = value.strip()
    return values


def set_config_value(config_file: Path, key: str, value: str) -> None:
    values = {**read_config(config_file), key: value}
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def ensure_device_config(config_file: Path, device: str = "cpu") -> None:
    """Write ``device=`` unless one is already set (a repair keeps the user's / the GUI's choice)."""
    if not read_config(config_file).get("device"):
        set_config_value(config_file, "device", device)


def save_device(config_file: Path, gpu: bool | None) -> None:
    """``gpu`` is the user's pick (new / clean install); None keeps the saved device (repair)."""
    if gpu is None:
        ensure_device_config(config_file)
    else:
        set_config_value(config_file, "device", "cuda" if gpu else "cpu")


def initial_gpu_choice(info: GpuInfo | None, saved: str | None, *, fresh: bool) -> bool:
    return gpu_capable(info) if fresh else saved == "cuda"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mib: int
    driver: tuple[int, ...]


def detect_gpu(env: Mapping[str, str], *, run: Callable[..., Any] = subprocess.run) -> GpuInfo | None:
    """First GPU from ``nvidia-smi`` (ships with every NVIDIA driver); None on any doubt."""
    cmd = [
        system_tool(env, "System32", "nvidia-smi.exe"),
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run(cmd, capture_output=True, text=True, timeout=15, check=False, creationflags=NO_WINDOW)
        name, vram, driver = (part.strip() for part in result.stdout.splitlines()[0].rsplit(",", 2))
        info = GpuInfo(name, int(float(vram)), tuple(int(n) for n in driver.split(".")))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return info if result.returncode == 0 else None


def gpu_capable(info: GpuInfo | None) -> bool:
    return info is not None and info.vram_mib >= MIN_GPU_VRAM_MIB and info.driver >= MIN_GPU_DRIVER


def gpu_summary(info: GpuInfo | None) -> str:
    if info is None:
        return "No NVIDIA graphics card found: transcription runs on the CPU."
    found = f"Found {info.name} ({info.vram_mib / 1024:.0f} GB)"
    if gpu_capable(info):
        return f"{found}."
    need = f"{MIN_GPU_VRAM_MIB / 1000:.0f} GB and driver {'.'.join(map(str, MIN_GPU_DRIVER))}+"
    return f"{found}, driver {'.'.join(map(str, info.driver))}: GPU mode needs {need}, so the CPU is used."


def read_hf_token(token_file: Path) -> str:
    try:
        token = token_file.read_text(encoding="utf-8-sig").strip()
    except OSError:
        token = ""  # nosec B105 - empty sentinel, not a credential
    if not token:
        raise InstallError(
            f"No Hugging Face token saved ({token_file}). Paste one in Transcribe's speaker panel, then retry."
        )
    return token


def is_installed(paths: InstallPaths) -> bool:
    """Anything but setup logs in the install dir, so a partial uninstall's leftovers stay retryable."""
    try:
        return any(child.name not in {"logs", paths.tmp_dir.name} for child in paths.install_dir.iterdir())
    except OSError:
        return False


def app_in_use(paths: InstallPaths, *, rename: Callable[[Path, Path], None] = os.rename) -> bool:
    """Windows refuses to rename a folder holding a running exe: a rename-and-back probe fails while the app is open."""
    scripts = paths.venv_dir / "Scripts"
    if not paths.windows or not scripts.is_dir():
        return False
    probe = scripts.with_name("Scripts.inuse-probe")
    try:
        rename(scripts, probe)
    except OSError:
        return True
    rename(probe, scripts)
    return False


def system_tool(env: Mapping[str, str], *parts: str) -> str:
    """``%SystemRoot%\\<parts>`` so a same-named exe on PATH is never run; the bare name without SystemRoot."""
    root = next((v for k, v in env.items() if k.upper() == "SYSTEMROOT" and v), "")  # a dict copy loses case-folding
    return str(Path(root, *parts)) if root else parts[-1].removesuffix(".exe")


def clean_install(paths: InstallPaths, *, rmtree: Callable[..., Any] = shutil.rmtree) -> None:
    """Remove what the installer put in the install dir; keep the setup copy, logs and the uv / HF download caches.

    Kept models are not trusted: a clean install re-fetches them with ``--force-download``.
    """
    try:  # the venv first and strictly: a half-deleted venv under a running app is the worst outcome
        rmtree(paths.venv_dir)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise InstallError(f"Could not remove {paths.venv_dir} ({error}). Close Transcribe, then retry.") from None
    for child in (paths.uv_dir, paths.app_dir, paths.python_dir, paths.ffmpeg_bin.parent, paths.install_dir / "work"):
        rmtree(child, ignore_errors=True)


def _rmtree_strict(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise InstallError(f"Could not remove {path} ({error}). Close Transcribe, then retry.") from None


def _rmtree_best_effort(path: Path, left: list[str]) -> None:
    """Remove what can be removed under ``path``, appending ``"<path> (<error>)"`` for each failure to ``left``."""

    def onexc(_func: Callable[..., Any], failed: str, error: BaseException) -> None:
        if not isinstance(error, FileNotFoundError):
            left.append(f"{failed} ({error})")

    shutil.rmtree(path, onexc=onexc)


def uninstall(
    paths: InstallPaths,
    env: Mapping[str, str],
    reg: Any = None,
    *,
    run: Callable[..., Any] = subprocess.run,
    in_use: Callable[[InstallPaths], bool] = app_in_use,
) -> None:
    """Remove the install dir, the config dir, both shortcuts and the Apps & features entry.

    Past the venv, removal is best effort: shortcuts, entry and config go even when a locked file keeps part of
    the install dir, and the InstallError then lists what was left (a rerun of setup offers Uninstall again).
    """
    if in_use(paths):
        raise InstallError("Close Transcribe first, then retry.")
    for target in (paths.install_dir, paths.config_file.parent):
        if target.name != APP_NAME:  # both are <base>\stt-faster by construction; never rmtree anything else
            raise InstallError(f"Refusing to remove unexpected folder {target}")
    _rmtree_strict(paths.venv_dir)  # first, like clean_install: never leave a half-deleted venv
    left: list[str] = []
    for target in (paths.install_dir, paths.config_file.parent):
        _rmtree_best_effort(target, left)
    if paths.windows:
        powershell = system_tool(env, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        cmd = [powershell, "-NoProfile", "-NonInteractive", "-Command", remove_shortcuts_script()]
        run(cmd, capture_output=True, check=False, creationflags=NO_WINDOW)
        if reg is not None:
            unregister_uninstall(reg)
    if left:
        more = f"\n… and {len(left) - 5} more" if len(left) > 5 else ""
        raise InstallError("Could not remove:\n" + "\n".join(left[:5]) + more + "\nClose Transcribe, then retry.")


def running_exe() -> Path | None:
    """The frozen Transcribe-Setup.exe running now; None when run as a script."""
    return Path(sys.executable).resolve() if getattr(sys, "frozen", False) else None


def relaunch_from_temp(
    exe: Path,
    args: Sequence[str],
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    tempdir: Callable[[], str] = tempfile.gettempdir,
) -> None:
    """Windows can't delete a running exe: continue the uninstall from a %TEMP% copy of it."""
    copy = Path(tempfile.mkdtemp(prefix=UNINSTALL_TEMP_PREFIX, dir=tempdir())) / exe.name
    shutil.copy2(exe, copy)
    # cwd outside the install dir: Explorer starts us there, and Windows won't remove a process's cwd.
    popen([str(copy), *args], cwd=tempdir(), creationflags=NO_WINDOW)


def self_delete_command(exe: Path, env: Mapping[str, str]) -> str:
    """Hidden cmd that retries deleting ``exe`` (and its temp folder) for ~20 s, once this process has exited."""
    cmd = system_tool(env, "System32", "cmd.exe")
    ping = system_tool(env, "System32", "PING.EXE")
    folder = exe.parent
    return (
        f'"{cmd}" /d /c for /l %i in (1,1,10) do '
        f'(("{ping}" -n 3 127.0.0.1 >nul) & rmdir /s /q "{folder}" 2>nul & if not exist "{folder}" exit)'
    )


def schedule_self_delete(exe: Path, env: Mapping[str, str], *, popen: Callable[..., Any] = subprocess.Popen) -> None:
    if exe.parent.name.startswith(UNINSTALL_TEMP_PREFIX):  # only ever our own relaunch copy
        cmd = self_delete_command(exe, env)
        popen(cmd, cwd=exe.parent.parent, creationflags=NO_WINDOW)  # never cwd in what it deletes


def self_command() -> list[str]:
    exe = running_exe()
    return [str(exe)] if exe else [sys.executable, str(Path(__file__).resolve())]


UNINSTALL_PROMPT = (
    "Remove Transcribe, its downloaded models and its settings?\n\nYour audio files and transcripts are not touched."
)


def _winreg() -> Any:
    import winreg  # noqa: PLC0415 - Windows-only module

    return winreg


def _messagebox() -> Any:
    from tkinter import messagebox  # noqa: PLC0415 - windowed mode only

    return messagebox


def uninstall_main(
    paths: InstallPaths,
    env: Mapping[str, str],
    *,
    headless: bool,
    confirmed: bool,
    tk_root: Callable[[], Any] = tk.Tk,
    dialogs: Callable[[], Any] = _messagebox,
    winreg: Callable[[], Any] = _winreg,
    current_exe: Callable[[], Path | None] = running_exe,
    relaunch: Callable[[Path, Sequence[str]], None] = relaunch_from_temp,
    remove: Callable[..., None] = uninstall,
    self_delete: Callable[[Path, Mapping[str, str]], None] = schedule_self_delete,
) -> int:
    """``--uninstall``: confirm, hop to a %TEMP% copy when running from the install dir, remove, self-delete."""
    messagebox = dialogs()
    root = None
    if not headless:
        root = tk_root()
        root.withdraw()
    if root is not None and not confirmed:
        if not messagebox.askyesno("Uninstall Transcribe", UNINSTALL_PROMPT, parent=root):
            return 1
    exe = current_exe()
    if exe is not None and paths.windows and exe.is_relative_to(paths.install_dir.resolve()):
        relaunch(exe, ["--uninstall", "--yes", *(["--headless"] if headless else [])])
        return 0
    try:
        remove(paths, env, reg=winreg() if paths.windows else None)
    except InstallError as error:
        LOGGER.error("Uninstall failed: %s", error)
        if root is not None:
            messagebox.showerror("Uninstall Transcribe", f"Uninstall did not finish:\n{error}", parent=root)
        return 1
    LOGGER.info("Removed %s and %s", paths.install_dir, paths.config_file.parent)
    if root is not None:
        messagebox.showinfo("Uninstall Transcribe", "Transcribe was removed.", parent=root)
    if exe is not None and paths.windows:
        self_delete(exe, env)
    return 0


# --- task runner ---------------------------------------------------------------------------------

Report = Callable[[float | None, str], None]  # (fraction 0..1 or None = indeterminate, status text)


@dataclass
class Task:
    key: str
    label: str
    run: Callable[[Report], None]
    needs: tuple[str, ...] = ()


@dataclass
class Events:
    progress: Callable[[str, float | None, str], None]
    state: Callable[[str, str, str], None]  # (key, running|done|failed|skipped, message)


def run_tasks(tasks: Sequence[Task], events: Events) -> bool:
    """Run tasks concurrently, each after its ``needs``; a failed dependency skips its dependants."""
    declared: set[str] = set()
    for task in tasks:
        if unknown := [dep for dep in task.needs if dep not in declared]:
            raise ValueError(f"Task {task.key} needs {unknown}, which are not declared before it")
        declared.add(task.key)
    futures: dict[str, Future[None]] = {}

    def execute(task: Task) -> None:
        for dep in task.needs:
            try:
                futures[dep].result()
            except Exception:  # noqa: BLE001 - the dependency already reported its own failure
                events.state(task.key, "skipped", f"skipped ({dep} failed)")
                raise InstallError(f"{dep} failed") from None
        events.state(task.key, "running", "")
        try:
            task.run(lambda fraction, text: events.progress(task.key, fraction, text))
        except Exception as error:
            LOGGER.exception("Task %s failed", task.key)
            events.state(task.key, "failed", str(error))
            raise
        events.state(task.key, "done", "")

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        for task in tasks:  # dependencies are declared earlier in the list, so their futures exist
            futures[task.key] = pool.submit(execute, task)
    return all(f.exception() is None for f in futures.values())


def kill_tree(proc: subprocess.Popen[str], env: Mapping[str, str]) -> None:
    """Kill ``proc`` and its children (uv spawns python / hf), not just the direct child."""
    if sys.platform == "win32":
        taskkill = [system_tool(env, "System32", "taskkill.exe"), "/T", "/F", "/PID", str(proc.pid)]
        subprocess.run(taskkill, capture_output=True, check=False, creationflags=NO_WINDOW)  # noqa: S603  # nosec B603
        proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # start_new_session made the child a group leader
        except ProcessLookupError:
            pass
    proc.wait()


def run_process(
    cmd: Sequence[str],
    env: Mapping[str, str],
    report: Report,
    cancel: threading.Event,
    poll: Callable[[], float | None] | None = None,
) -> None:
    """Run ``cmd``, reporting its last output line; ``poll`` supplies a progress fraction when it can."""
    LOGGER.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(  # noqa: S603  # nosec B603 - fixed argument list
        list(cmd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
        start_new_session=sys.platform != "win32",  # own process group, so cancel can kill the tree
    )
    lines: queue.Queue[str] = queue.Queue()
    tail: list[str] = []

    def reader() -> None:
        for line in proc.stdout or ():  # universal newlines also split tqdm's \r updates
            if line.strip():
                lines.put(line.strip())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    last = ""
    while proc.poll() is None:
        if cancel.is_set():
            kill_tree(proc, env)
            raise Cancelled("Cancelled")
        while not lines.empty():
            last = lines.get()
            LOGGER.debug("%s", last)
            tail = [*tail[-19:], last]
        report(poll() if poll else None, last[-100:])
        time.sleep(0.3)
    thread.join(timeout=5)
    while not lines.empty():
        tail = [*tail[-19:], lines.get()]
    if proc.returncode != 0:
        LOGGER.error("Command failed (%s):\n%s", proc.returncode, "\n".join(tail))
        raise CommandFailed(tail[-1] if tail else f"exit code {proc.returncode}", "\n".join(tail))


def launch_gui(paths: InstallPaths, *, popen: Callable[..., Any] = subprocess.Popen) -> None:
    # Never cwd inside the venv: a process parked in Scripts makes app_in_use report "open".
    popen([str(paths.gui_exe)], cwd=str(paths.install_dir), creationflags=NO_WINDOW)


@dataclass
class Installer:
    paths: InstallPaths
    source: str | None = None
    clean: bool = False
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    cancel: threading.Event = field(default_factory=threading.Event)
    runner: Callable[..., None] = run_process
    seeder: Callable[[Path, Path], bool] = seed_model_cache
    model_size: Callable[[ModelSpec, Callable[[str], Any]], int | None] = expected_model_size
    in_use: Callable[[InstallPaths], bool] = app_in_use
    winreg: Callable[[], Any] = _winreg
    extras: bool = False
    launch: Callable[[InstallPaths], None] = launch_gui
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    close_timeout: float = 60.0  # the app quits right after starting --extras
    hf_token: str = field(default="", repr=False)  # read from hf_token_file when an --extras run starts
    token_status: Callable[[str], int | None] = hf_token_status
    gpu: bool | None = None  # the user's pick for a new / clean install; None keeps the saved device
    detect: Callable[[Mapping[str, str]], GpuInfo | None] = detect_gpu

    def _download_step(self, url: str, dest: Path, report: Report) -> None:
        def on_progress(done: int, total: int | None) -> None:
            text = f"{done / 1e6:.0f} / {total / 1e6:.0f} MB" if total else f"{done / 1e6:.0f} MB"
            report(done / total if total else None, text)

        download(url, dest, on_progress, self.cancel)

    def fetch_uv(self, report: Report) -> None:
        if self.paths.uv_exe.is_file() and not self.clean:
            report(1.0, "already present")
            return
        asset = uv_asset(sys.platform, platform.machine())
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / asset
            self._download_step(uv_url(asset), archive, report)
            names = ("uv.exe",) if self.paths.windows else ("uv",)
            extract_named(archive, names, self.paths.uv_dir)

    def fetch_source(self, report: Report) -> None:
        source = self.source or latest_release_zip()
        kind = classify_source(source)
        staged = self.paths.install_dir / "app.new"
        shutil.rmtree(staged, ignore_errors=True)
        if kind == "dir":
            report(None, f"copying {source}")
            shutil.copytree(source, staged, ignore=source_ignore)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(source) if kind == "zip" else Path(tmp) / "source.zip"
                if kind == "url":
                    self._download_step(source, archive, report)
                extract_zip_stripped(archive, staged)
        if not (staged / "pyproject.toml").is_file():
            shutil.rmtree(staged, ignore_errors=True)
            raise InstallError(f"{source} does not contain the app (no pyproject.toml)")
        swap_in(staged, self.paths.app_dir)

    def install_deps(self, report: Report) -> None:
        cmd = deps_command(self.paths, diarization=self.extras)
        self.runner(cmd, deps_env(self.env, self.paths), report, self.cancel)

    def fetch_model(self, spec: ModelSpec, report: Report, *, token: str = "") -> None:
        cache = model_cache_dir(self.paths.hf_home / "hub", spec.repo_id)
        if not self.clean:  # a v1.1.0 install downloaded into the user's shared HF cache
            legacy = model_cache_dir(hf_hub_cache(self.env, Path.home()), spec.repo_id)
            report(None, "reusing earlier download")
            if self.seeder(legacy, cache):
                LOGGER.info("Copied %s from %s", spec.repo_id, legacy)
        total = self.model_size(spec, fetch_json)

        def poll() -> float | None:
            return min(dir_size(cache) / total, 1.0) if total else None

        env = isolated_env(self.env, self.paths)
        if token:
            env["HF_TOKEN"] = token
        self.runner(model_command(self.paths, spec, force=self.clean), env, report, self.cancel, poll)

    def fetch_ffmpeg(self, report: Report) -> None:
        if all((self.paths.ffmpeg_bin / name).is_file() for name in FFMPEG_BINARIES) and not self.clean:
            report(1.0, "already present")
            return
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ffmpeg.zip"
            self._download_step(FFMPEG_URL, archive, report)
            extract_named(archive, FFMPEG_BINARIES, self.paths.ffmpeg_bin)

    def finish(self, report: Report) -> None:
        if not self.paths.gui_exe.is_file():
            raise InstallError(f"Install finished but {self.paths.gui_exe.name} is missing")
        if getattr(sys, "frozen", False):  # keep a copy so the GUI can re-run setup (repair / extras)
            exe = Path(sys.executable)
            if exe.resolve() != self.paths.setup_copy.resolve():
                shutil.copy2(exe, self.paths.setup_copy)
        if self.paths.windows:
            report(None, "creating shortcuts")
            powershell = system_tool(self.env, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
            cmd = [powershell, "-NoProfile", "-NonInteractive", "-Command", shortcut_script(self.paths.gui_exe)]
            self.runner(cmd, self.env, report, self.cancel)
            if self.paths.setup_copy.is_file():  # the entry's UninstallString runs that copy
                register_uninstall(self.paths, self.winreg())

    def fetch_diarization(self, report: Report) -> None:
        try:
            self.fetch_model(DIARIZATION_MODEL, report, token=self.hf_token)
        except CommandFailed as error:
            if HF_AUTH_ERROR.search(error.output):
                raise InstallError(HF_AUTH_HELP) from None
            raise

    def enable_extras(self, report: Report) -> None:
        set_config_value(self.paths.config_file, "extras", "diarization")
        report(None, "opening Transcribe")
        self.launch(self.paths)

    def wait_closed(self) -> None:
        deadline = self.clock() + self.close_timeout
        while self.in_use(self.paths):
            if self.clock() >= deadline or self.cancel.is_set():
                raise InstallError("Close Transcribe first, then retry.")
            self.sleep(0.5)

    def tasks(self) -> list[Task]:
        if self.extras:  # the base install is in place; one deps sync + the gated model, then the flag
            model = DIARIZATION_MODEL.repo_id
            return [
                Task("deps", "Python + speaker libraries", self.install_deps),
                Task(model, f"Model: {model.split('/', 1)[1]}", self.fetch_diarization),
                Task("extras", "Speaker detection", self.enable_extras, needs=("deps", model)),
            ]
        tasks = [
            Task("uv", "Installer tools (uv)", self.fetch_uv),
            Task("app", "Transcribe app", self.fetch_source),
        ]
        if self.paths.windows:  # elsewhere the system ffmpeg is used, as with the CLI
            tasks.append(Task("ffmpeg", "Audio decoder (ffmpeg)", self.fetch_ffmpeg))
        for spec in MODELS:
            label = f"Model: {spec.repo_id.split('/', 1)[1]}"
            tasks.append(Task(spec.repo_id, label, lambda r, s=spec: self.fetch_model(s, r), needs=("uv",)))
        tasks.append(Task("deps", "Python + libraries", self.install_deps, needs=("uv", "app")))
        tasks.append(Task("finish", "Shortcuts", self.finish, needs=tuple(t.key for t in tasks)))
        return tasks

    def run(self, events: Events) -> bool:
        if self.extras:
            if not self.paths.uv_exe.is_file() or not self.paths.app_dir.is_dir():
                raise InstallError("Transcribe is not installed yet. Run Transcribe-Setup first.")
            self.hf_token = read_hf_token(self.paths.hf_token_file)
            status = self.token_status(self.hf_token)
            LOGGER.info("Hugging Face token check: %s", status if status is not None else "no answer")
            if status in (401, 403):  # fail before the long deps sync; other answers leave it to fetch_diarization
                raise InstallError(HF_AUTH_HELP)
            self.wait_closed()
        elif self.in_use(self.paths):
            raise InstallError("Close Transcribe first, then retry.")
        self.paths.tmp_dir.mkdir(parents=True, exist_ok=True)
        if self.clean:
            clean_install(self.paths)
        if not self.extras:  # before deps: device=cuda adds the gpu-win extra to the sync
            save_device(self.paths.config_file, self.gpu)
        return run_tasks(self.tasks(), events)


# --- window --------------------------------------------------------------------------------------


class SetupWindow:
    def __init__(
        self,
        root: tk.Tk,
        installer: Installer,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        tempdir: Callable[[], str] = tempfile.gettempdir,
        dialogs: Callable[[], Any] = _messagebox,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root = root
        self.installer = installer
        self.popen = popen
        self.tempdir = tempdir
        self.dialogs = dialogs
        self.logger = logging.getLogger() if logger is None else logger  # the root logger holds logs\setup.log
        self.events: queue.Queue[tuple[str, str, float | None, str]] = queue.Queue()
        self.rows: dict[str, tuple[ttk.Progressbar, ttk.Label]] = {}
        self.running = False
        self.failure = ""  # first failed task's message, shown in the summary

        root.title("Transcribe — Setup")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.minsize(520, 0)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"Installs to {installer.paths.install_dir}\nNo administrator rights needed.").pack(
            anchor="w"
        )

        self.mode = tk.StringVar(value="repair")
        if is_installed(installer.paths) and not installer.extras:
            mode_row = ttk.Frame(frame)
            mode_row.pack(anchor="w", pady=(8, 0))
            ttk.Label(mode_row, text="Already installed:").pack(side="left", padx=(0, 8))
            ttk.Radiobutton(mode_row, text="Repair / update", value="repair", variable=self.mode).pack(side="left")
            ttk.Radiobutton(mode_row, text="Clean reinstall", value="clean", variable=self.mode).pack(
                side="left", padx=8
            )
            ttk.Radiobutton(mode_row, text="Uninstall", value="uninstall", variable=self.mode).pack(side="left")

        self.use_gpu = tk.BooleanVar(value=False)
        self.gpu_check: ttk.Checkbutton | None = None
        self.gpu_info: GpuInfo | None = None
        self.gpu_pick = False  # True: this run saves use_gpu; False: a repair keeps the saved device
        if installer.paths.windows and not installer.extras:
            self.gpu_info = installer.detect(installer.env)
            ttk.Label(frame, text=gpu_summary(self.gpu_info), wraplength=480).pack(anchor="w", pady=(8, 0))
            if gpu_capable(self.gpu_info) or read_config(installer.paths.config_file).get("device") == "cuda":
                text = f"Use the graphics card (faster; {GPU_EXTRA_SIZE} extra download)"
                self.gpu_check = ttk.Checkbutton(frame, text=text, variable=self.use_gpu)
                self.gpu_check.pack(anchor="w")
            self._sync_gpu_check()

        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=12)
        grid.columnconfigure(1, weight=1)
        for row, task in enumerate(installer.tasks()):
            ttk.Label(grid, text=task.label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            bar = ttk.Progressbar(grid, mode="determinate", maximum=1.0, length=220)
            bar.grid(row=row, column=1, sticky="ew", pady=2)
            status = ttk.Label(grid, text="waiting", foreground="gray", width=24)
            status.grid(row=row, column=2, sticky="w", padx=(8, 0))
            self.rows[task.key] = (bar, status)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.install_button = ttk.Button(buttons, text="Install", command=self.start)
        self.install_button.pack(side="left")
        self.launch_button = ttk.Button(buttons, text="Open Transcribe", command=self.launch, state="disabled")
        self.launch_button.pack(side="right")
        self.summary = ttk.Label(frame, text="", foreground="gray", wraplength=480)
        self.summary.pack(anchor="w", pady=(8, 0))
        self.mode.trace_add("write", self._on_mode)
        if installer.extras:  # the app asked for this run and has already quit
            root.after(100, self.start)

    def _on_mode(self, *_: object) -> None:
        self.install_button.config(text="Uninstall" if self.mode.get() == "uninstall" else "Install")
        self._sync_gpu_check()

    def _sync_gpu_check(self) -> None:
        """New / clean installs pick the device; a repair shows the saved one, read-only."""
        self.gpu_pick = self.mode.get() == "clean" or not is_installed(self.installer.paths)
        saved = read_config(self.installer.paths.config_file).get("device")
        self.use_gpu.set(initial_gpu_choice(self.gpu_info, saved, fresh=self.gpu_pick))
        if self.gpu_check is not None:
            self.gpu_check.config(state="normal" if self.gpu_pick else "disabled")

    def uninstall(self) -> None:
        if not self.dialogs().askyesno("Uninstall Transcribe", UNINSTALL_PROMPT, parent=self.root):
            return
        for handler in self.logger.handlers[:]:  # release logs\setup.log so the install dir can go
            handler.close()
            self.logger.removeHandler(handler)
        cmd = [*self_command(), "--uninstall", "--yes"]
        self.popen(cmd, cwd=self.tempdir(), creationflags=NO_WINDOW)  # not the install dir
        self.root.destroy()

    def start(self) -> None:
        if self.mode.get() == "uninstall":
            self.uninstall()
            return
        self.running = True
        self.failure = ""
        self.installer.clean = self.mode.get() == "clean"
        self.installer.gpu = self.use_gpu.get() if self.gpu_pick else None  # kept on a same-window retry
        self.installer.cancel.clear()
        self.install_button.config(state="disabled")
        self.summary.config(text="Installing…")
        for bar, status in self.rows.values():
            bar.stop()
            bar.config(mode="determinate", value=0)
            status.config(text="waiting")
        events = Events(
            progress=lambda key, fraction, text: self.events.put(("progress", key, fraction, text)),
            state=lambda key, state, message: self.events.put(("state", key, None, f"{state}\t{message}")),
        )
        threading.Thread(target=self._work, args=(events,), daemon=True).start()
        self.root.after(100, self._poll)

    def _work(self, events: Events) -> None:
        try:
            ok = self.installer.run(events)
        except Exception as error:  # noqa: BLE001 - surfaced in the window, details in the log
            LOGGER.exception("Setup failed")
            self.events.put(("summary", "", None, f"Failed: {error}"))
            ok = False
        self.events.put(("finished", "", None, "ok" if ok else "failed"))

    def _poll(self) -> None:
        while not self.events.empty():
            kind, key, fraction, text = self.events.get()
            if kind == "progress":
                bar, status = self.rows[key]
                if fraction is None:
                    if str(bar.cget("mode")) != "indeterminate":
                        bar.config(mode="indeterminate")
                        bar.start(15)
                else:
                    bar.stop()
                    bar.config(mode="determinate", value=fraction)
                status.config(text=text[-40:])
            elif kind == "state":
                bar, status = self.rows[key]
                state, _, message = text.partition("\t")
                if state == "failed" and not self.failure:
                    self.failure = message
                if state != "running":
                    bar.stop()
                    bar.config(mode="determinate", value=1.0 if state == "done" else 0)
                    status.config(text={"done": "✓ done", "skipped": "skipped"}.get(state, f"✗ {message}"[:40]))
            elif kind == "summary":
                self.failure = self.failure or text  # a pre-task error; _finished would otherwise drop it
                self.summary.config(text=text)
            else:
                self._finished(text == "ok")
                return
        self.root.after(100, self._poll)

    def _finished(self, ok: bool) -> None:
        self.running = False
        if ok and self.installer.extras:  # the last extras task already reopened Transcribe
            self.root.destroy()
            return
        if ok:
            self.summary.config(text="Done. Transcribe is on your Desktop and in the Start menu.")
            self.launch_button.config(state="normal")
            self.install_button.config(text="Reinstall", state="normal")
        else:
            log = self.installer.paths.log_file
            reason = f"\n{self.failure}\n" if self.failure else " "
            self.summary.config(text=f"Setup did not finish.{reason}Details: {log}")
            self.install_button.config(text="Retry", state="normal")

    def launch(self) -> None:
        launch_gui(self.installer.paths, popen=self.popen)
        self.root.destroy()

    def _on_close(self) -> None:
        self.installer.cancel.set()  # kills running uv / hf processes and stops downloads
        self.root.destroy()


def headless_gpu(installer: Installer, *, cpu: bool) -> bool | None:
    if installer.extras or not installer.paths.windows:  # elsewhere the CLI's own device probe decides
        return None
    if cpu:
        return False
    if is_installed(installer.paths) and not installer.clean:
        return None  # a repair keeps the saved device
    gpu = installer.detect(installer.env)
    LOGGER.info("%s", gpu_summary(gpu))
    return gpu_capable(gpu)


def _setup_logging(log_file: Path | None, *, console: bool) -> None:
    handlers: list[logging.Handler] = [logging.NullHandler()]
    if log_file is not None:  # None while uninstalling: an open log would lock the install dir
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if console:  # a --windowed exe has no stderr
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the Transcribe app for the current user.")
    parser.add_argument("--source", help="app source: URL of a zip, a local .zip or a folder (default: latest release)")
    parser.add_argument("--clean", action="store_true", help="clean reinstall (headless; the window asks)")
    parser.add_argument("--headless", action="store_true", help="no window; log progress to the console")
    parser.add_argument("--uninstall", action="store_true", help="remove the app, its models, settings and shortcuts")
    parser.add_argument("--extras", action="store_true", help="add speaker detection to an existing install")
    parser.add_argument("--cpu", action="store_true", help="headless: skip GPU mode even if an NVIDIA GPU is found")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # already confirmed (relaunch)
    args = parser.parse_args(argv)
    if args.extras and (args.source or args.clean or args.uninstall):
        parser.error("--extras cannot be combined with --source, --clean or --uninstall")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = default_install_paths()
    if args.uninstall:
        _setup_logging(None, console=args.headless)
        return uninstall_main(paths, os.environ, headless=args.headless, confirmed=args.yes)
    _setup_logging(paths.log_file, console=args.headless)
    installer = Installer(paths=paths, source=args.source, clean=args.clean, extras=args.extras)
    if args.headless:
        installer.gpu = headless_gpu(installer, cpu=args.cpu)
        last: dict[str, str] = {}

        def progress(key: str, fraction: float | None, text: str) -> None:
            line = f"{text} ({fraction:.0%})" if fraction is not None else text
            if last.get(key) != line:
                last[key] = line
                LOGGER.info("%s: %s", key, line)

        def state(key: str, value: str, message: str) -> None:
            LOGGER.info("%s: %s %s", key, value, message)

        return 0 if installer.run(Events(progress=progress, state=state)) else 1

    root = tk.Tk()
    SetupWindow(root, installer)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
