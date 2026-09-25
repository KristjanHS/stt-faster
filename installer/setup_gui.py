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
import shutil
import subprocess  # nosec B404 - runs uv / powershell with fixed argument lists
import sys
import tarfile
import tempfile
import threading
import time
import tkinter as tk
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


class InstallError(RuntimeError):
    """A step failed with a message fit to show the user."""


class Cancelled(InstallError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    include: tuple[str, ...] = ()


# Estonian (et-large) and English (turbo) GUI profiles, see backend/model_config.py.
MODELS = (
    ModelSpec("TalTechNLP/whisper-large-v3-turbo-et-verbatim", ("ct2/*",)),
    ModelSpec("Systran/faster-distil-whisper-large-v3"),
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


def default_install_paths(env: Mapping[str, str] | None = None, plat: str | None = None) -> InstallPaths:
    """Mirror of ``backend.gui.default_app_paths`` (this file cannot import the app)."""
    env = os.environ if env is None else env
    plat = sys.platform if plat is None else plat
    home = Path.home()

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
        for info in zf.infolist():
            parts = Path(info.filename).parts[1:] if strip else Path(info.filename).parts
            if not parts or info.is_dir():
                continue
            if ".." in parts or Path(info.filename).is_absolute():
                raise InstallError(f"Unsafe path in archive: {info.filename}")
            target = dest.joinpath(*parts)
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


def swap_in(staged: Path, target: Path) -> None:
    """Replace ``target`` with ``staged`` so a failed fetch never leaves a half-written app dir."""
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    staged.rename(target)
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


def deps_command(paths: InstallPaths) -> list[str]:
    return [
        str(paths.uv_exe),
        "sync",
        "--frozen",
        "--no-dev",
        "--extra",
        "gui",
        "--python-preference",
        "only-managed",  # never pick a stray system/Store Python
        "--project",
        str(paths.app_dir),
    ]


def deps_env(base: Mapping[str, str], paths: InstallPaths) -> dict[str, str]:
    # The venv lives beside app/ so a repair can replace the source without touching it.
    return {**base, "UV_PROJECT_ENVIRONMENT": str(paths.venv_dir)}


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
        "foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { "
        f"$l = $s.CreateShortcut((Join-Path $d {_ps_quote(name + '.lnk')})); "
        f"$l.TargetPath = {_ps_quote(str(target))}; "
        f"$l.WorkingDirectory = {_ps_quote(str(target.parent))}; "
        "$l.Save() }"
    )


def ensure_device_config(config_file: Path, device: str = "cpu") -> None:
    """Write ``device=`` unless one is already set (a repair keeps the user's / the GUI's choice)."""
    values: dict[str, str] = {}
    if config_file.is_file():
        for line in config_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip():
                values[key.strip()] = value.strip()
    if values.get("device"):
        return
    values["device"] = device
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def is_installed(paths: InstallPaths) -> bool:
    return paths.app_dir.is_dir() or paths.venv_dir.is_dir()


def clean_install(paths: InstallPaths) -> None:
    """Remove everything the installer put in the install dir, except the running setup copy and logs."""
    for child in (paths.uv_dir, paths.app_dir, paths.venv_dir, paths.ffmpeg_bin.parent, paths.install_dir / "work"):
        shutil.rmtree(child, ignore_errors=True)


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
            proc.kill()
            proc.wait()
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
        raise InstallError(tail[-1] if tail else f"exit code {proc.returncode}")


@dataclass
class Installer:
    paths: InstallPaths
    source: str | None = None
    clean: bool = False
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    cancel: threading.Event = field(default_factory=threading.Event)

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
        run_process(deps_command(self.paths), deps_env(self.env, self.paths), report, self.cancel)

    def fetch_model(self, spec: ModelSpec, report: Report) -> None:
        cache = model_cache_dir(hf_hub_cache(self.env, Path.home()), spec.repo_id)
        total = expected_model_size(spec, fetch_json)

        def poll() -> float | None:
            return min(dir_size(cache) / total, 1.0) if total else None

        run_process(model_command(self.paths, spec, force=self.clean), self.env, report, self.cancel, poll)

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
        ensure_device_config(self.paths.config_file)
        if getattr(sys, "frozen", False):  # keep a copy so the GUI can re-run setup (repair / extras)
            exe = Path(sys.executable)
            if exe.resolve() != self.paths.setup_copy.resolve():
                shutil.copy2(exe, self.paths.setup_copy)
        if self.paths.windows:
            report(None, "creating shortcuts")
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", shortcut_script(self.paths.gui_exe)]
            run_process(cmd, self.env, report, self.cancel)

    def tasks(self) -> list[Task]:
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
        self.paths.install_dir.mkdir(parents=True, exist_ok=True)
        if self.clean:
            clean_install(self.paths)
        return run_tasks(self.tasks(), events)


# --- window --------------------------------------------------------------------------------------


class SetupWindow:
    def __init__(self, root: tk.Tk, installer: Installer) -> None:
        self.root = root
        self.installer = installer
        self.events: queue.Queue[tuple[str, str, float | None, str]] = queue.Queue()
        self.rows: dict[str, tuple[ttk.Progressbar, ttk.Label]] = {}
        self.running = False

        root.title("Transcribe — Setup")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.minsize(520, 0)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"Installs to {installer.paths.install_dir}\nNo administrator rights needed.").pack(
            anchor="w"
        )

        self.mode = tk.StringVar(value="repair")
        if is_installed(installer.paths):
            mode_row = ttk.Frame(frame)
            mode_row.pack(anchor="w", pady=(8, 0))
            ttk.Label(mode_row, text="Already installed:").pack(side="left", padx=(0, 8))
            ttk.Radiobutton(mode_row, text="Repair / update", value="repair", variable=self.mode).pack(side="left")
            ttk.Radiobutton(mode_row, text="Clean reinstall", value="clean", variable=self.mode).pack(
                side="left", padx=8
            )

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

    def start(self) -> None:
        self.running = True
        self.installer.clean = self.mode.get() == "clean"
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
                if state != "running":
                    bar.stop()
                    bar.config(mode="determinate", value=1.0 if state == "done" else 0)
                    status.config(text={"done": "✓ done", "skipped": "skipped"}.get(state, f"✗ {message}"[:40]))
            elif kind == "summary":
                self.summary.config(text=text)
            else:
                self._finished(text == "ok")
                return
        self.root.after(100, self._poll)

    def _finished(self, ok: bool) -> None:
        self.running = False
        if ok:
            self.summary.config(text="Done. Transcribe is on your Desktop and in the Start menu.")
            self.launch_button.config(state="normal")
            self.install_button.config(text="Reinstall", state="normal")
        else:
            log = self.installer.paths.log_file
            self.summary.config(text=f"Setup did not finish. Details: {log}")
            self.install_button.config(text="Retry", state="normal")

    def launch(self) -> None:
        subprocess.Popen([str(self.installer.paths.gui_exe)], creationflags=NO_WINDOW)  # noqa: S603  # nosec B603
        self.root.destroy()

    def _on_close(self) -> None:
        self.installer.cancel.set()  # kills running uv / hf processes and stops downloads
        self.root.destroy()


def _setup_logging(log_file: Path, *, console: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if console:  # a --windowed exe has no stderr
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the Transcribe app for the current user.")
    parser.add_argument("--source", help="app source: URL of a zip, a local .zip or a folder (default: latest release)")
    parser.add_argument("--clean", action="store_true", help="clean reinstall (headless; the window asks)")
    parser.add_argument("--headless", action="store_true", help="no window; log progress to the console")
    args = parser.parse_args(argv)

    paths = default_install_paths()
    _setup_logging(paths.log_file, console=args.headless)
    installer = Installer(paths=paths, source=args.source, clean=args.clean)
    if args.headless:
        last: dict[str, str] = {}

        def progress(key: str, fraction: float | None, text: str) -> None:
            line = f"{text} ({fraction:.0%})" if fraction is not None else text
            if last.get(key) != line:
                last[key] = line
                LOGGER.debug("%s: %s", key, line)

        def state(key: str, value: str, message: str) -> None:
            LOGGER.info("%s: %s %s", key, value, message)

        return 0 if installer.run(Events(progress=progress, state=state)) else 1

    root = tk.Tk()
    SetupWindow(root, installer)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
