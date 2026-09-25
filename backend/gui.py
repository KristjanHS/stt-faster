"""Simple tkinter front-end: pick audio files, choose a language, transcribe via the CLI.

The GUI never transcribes in-process. It copies the chosen files into a private work
dir, runs ``stt-faster transcribe process`` on it as a subprocess, and copies each
resulting ``.txt`` next to the original audio file (never overwriting).
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess  # nosec B404 - runs our own CLI with a fixed argument list
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from backend.config import setup_logging

LOGGER = logging.getLogger(__name__)

APP_NAME = "stt-faster"
GUI_VARIANT = 61


@dataclass(frozen=True)
class GuiProfile:
    preset: str
    language: str
    variant: int = GUI_VARIANT


# Mirrors scripts/windows/transcribe_{estonian,english}_Teams.bat.
GUI_PROFILES: dict[str, GuiProfile] = {
    "Estonian": GuiProfile(preset="et-large", language="et"),
    "English": GuiProfile(preset="turbo", language="en"),
}


@dataclass(frozen=True)
class AppPaths:
    install_dir: Path
    config_file: Path

    @property
    def work_root(self) -> Path:
        return self.install_dir / "work"

    @property
    def ffmpeg_bin(self) -> Path:
        return self.install_dir / "ffmpeg" / "bin"


def default_app_paths(env: Mapping[str, str] | None = None) -> AppPaths:
    """%LOCALAPPDATA% / %APPDATA% on Windows, XDG dirs elsewhere."""
    env = os.environ if env is None else env
    home = Path.home()
    if sys.platform == "win32":
        install = Path(env.get("LOCALAPPDATA", home / "AppData" / "Local")) / APP_NAME
        config_dir = Path(env.get("APPDATA", home / "AppData" / "Roaming")) / APP_NAME
    else:
        install = Path(env.get("XDG_DATA_HOME", home / ".local" / "share")) / APP_NAME
        config_dir = Path(env.get("XDG_CONFIG_HOME", home / ".config")) / APP_NAME
    return AppPaths(install_dir=install, config_file=config_dir / "config")


def read_config(config_file: Path) -> dict[str, str]:
    if not config_file.is_file():
        return {}
    values: dict[str, str] = {}
    for line in config_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip():
            values[key.strip()] = value.strip()
    return values


def write_config_value(config_file: Path, key: str, value: str) -> None:
    values = read_config(config_file)
    values[key] = value
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def read_device(config_file: Path) -> str | None:
    """Configured device (``cpu``/``cuda``), or None to let the CLI auto-detect."""
    return read_config(config_file).get("device") or None


@cache
def supported_extensions() -> frozenset[str]:
    # Lazy: backend.components pulls in the transcription stack (seconds of import time).
    from backend.components import SUPPORTED_AUDIO_EXTENSIONS  # noqa: PLC0415

    return frozenset(SUPPORTED_AUDIO_EXTENSIONS)


def stage_files(files: Iterable[Path], work_dir: Path) -> dict[str, Path]:
    """Copy ``files`` into ``work_dir`` under unique stems; return staged stem → original.

    Extensions are lower-cased (the CLI globs ``*.mp3`` case-sensitively) and clashing
    stems (``a.mp3`` + ``a.wav``) get a ``__N`` suffix so their outputs don't collide.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for original in files:
        stem = original.stem
        n = 2
        while stem in staged:
            stem = f"{original.stem}__{n}"
            n += 1
        shutil.copy2(original, work_dir / f"{stem}{original.suffix.lower()}")
        staged[stem] = original
    return staged


def build_command(work_dir: Path, profile: GuiProfile, *, timestamps: bool) -> list[str]:
    # `--variant`, never `-v` (bound to both --variant and --verbose); CLI default is --diarize.
    return [
        sys.executable,
        "-m",
        "backend.cli.main",
        "transcribe",
        "process",
        str(work_dir),
        "--preset",
        profile.preset,
        "--variant",
        str(profile.variant),
        "--language",
        profile.language,
        "--output-format",
        "txt",
        "--no-diarize",
        "--timestamps" if timestamps else "--no-timestamps",
    ]


def build_env(base: Mapping[str, str], *, device: str | None, ffmpeg_bin: Path | None) -> dict[str, str]:
    env = dict(base)
    env["PYTHONIOENCODING"] = "utf-8"
    if device:
        env["STT_DEVICE"] = device
    if ffmpeg_bin is not None and ffmpeg_bin.is_dir():
        env["PATH"] = os.pathsep.join([str(ffmpeg_bin), env.get("PATH", "")])
    return env


def find_outputs(work_dir: Path, staged: Mapping[str, Path]) -> dict[Path, Path | None]:
    """Map each original file to its produced ``.txt`` in ``work_dir`` (None if missing)."""
    produced: dict[str, Path] = {}
    for txt in work_dir.rglob("*.txt"):
        produced.setdefault(txt.stem, txt)
    return {original: produced.get(stem) for stem, original in staged.items()}


def unique_destination(original: Path) -> Path:
    """``<stem>.txt`` next to ``original``; ``<stem> (N).txt`` if that already exists."""
    candidate = original.with_suffix(".txt")
    n = 2
    while candidate.exists():
        candidate = original.with_name(f"{original.stem} ({n}).txt")
        n += 1
    return candidate


def deliver_outputs(outputs: Mapping[Path, Path | None]) -> list[Path]:
    delivered: list[Path] = []
    for original, txt in outputs.items():
        if txt is not None:
            dest = unique_destination(original)
            shutil.copyfile(txt, dest)
            delivered.append(dest)
    return delivered


Runner = Callable[[list[str], dict[str, str], Callable[[str], None]], int]


def run_subprocess(cmd: list[str], env: dict[str, str], on_line: Callable[[str], None]) -> int:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with subprocess.Popen(  # nosec B603 - fixed command list, no shell
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    ) as proc:
        for line in proc.stdout or ():
            on_line(line.rstrip())
        return proc.wait()


@dataclass
class JobResult:
    delivered: list[Path] = field(default_factory=list[Path])
    missing: list[Path] = field(default_factory=list[Path])
    fell_back_to_cpu: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing


def run_job(
    files: list[Path],
    profile: GuiProfile,
    *,
    timestamps: bool,
    paths: AppPaths,
    runner: Runner = run_subprocess,
    on_line: Callable[[str], None] = lambda _line: None,
    base_env: Mapping[str, str] | None = None,
) -> JobResult:
    """Transcribe ``files``; a non-CPU run that fails is retried once on CPU (persisted)."""
    device = read_device(paths.config_file)
    base_env = os.environ if base_env is None else base_env
    result = JobResult()
    attempts = [device] if device == "cpu" else [device, "cpu"]
    for attempt_device in attempts:
        work_dir = paths.work_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{attempt_device or 'auto'}"
        try:
            staged = stage_files(files, work_dir)
            env = build_env(base_env, device=attempt_device, ffmpeg_bin=paths.ffmpeg_bin)
            exit_code = runner(build_command(work_dir, profile, timestamps=timestamps), env, on_line)
            outputs = find_outputs(work_dir, staged)
            if exit_code == 0 and all(txt is not None for txt in outputs.values()):
                result.delivered = deliver_outputs(outputs)
                result.missing = []
                return result
            result.missing = [original for original, txt in outputs.items() if txt is None] or list(files)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        if attempt_device != "cpu":
            LOGGER.warning("Transcription failed on device=%s; retrying once on CPU", attempt_device or "auto")
            on_line("Retrying on CPU…")
            write_config_value(paths.config_file, "device", "cpu")
            result.fell_back_to_cpu = True
    return result


def open_folder(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606  # nosec B606  # type: ignore[attr-defined]
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(folder)])  # noqa: S603  # nosec B603


def _make_root() -> tuple[tk.Tk, bool]:
    if sys.platform.startswith("linux"):
        # tkdnd links system libX11 while uv's Python embeds its own Xlib in libtk → xcb abort (uncatchable).
        return tk.Tk(), False
    try:
        from tkinterdnd2 import TkinterDnD  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]

        return TkinterDnD.Tk(), True
    except (ImportError, RuntimeError, tk.TclError) as error:
        LOGGER.info("Drag-and-drop unavailable (%s); using click-to-browse", error)
        return tk.Tk(), False


class TranscribeApp:
    def __init__(self, root: tk.Tk, *, dnd: bool, paths: AppPaths) -> None:
        self.root = root
        self.paths = paths
        self.files: list[Path] = []
        self.last_output_dir: Path | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        root.title("Transcribe")
        root.minsize(460, 300)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)

        hint = "Drop audio files here\nor click to browse" if dnd else "Click to choose files"
        self.zone = tk.Label(frame, text=hint, relief="groove", borderwidth=2, height=6, cursor="hand2")
        self.zone.pack(fill="x")
        self.zone.bind("<Button-1>", lambda _e: self.browse())
        if dnd:
            self.zone.drop_target_register("DND_Files")  # type: ignore[attr-defined]
            self.zone.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]

        self.files_label = ttk.Label(frame, text="No files selected")
        self.files_label.pack(anchor="w", pady=(6, 10))

        self.language = tk.StringVar(value="Estonian")
        lang_row = ttk.Frame(frame)
        lang_row.pack(anchor="w")
        for name in GUI_PROFILES:
            ttk.Radiobutton(lang_row, text=name, value=name, variable=self.language).pack(side="left", padx=(0, 12))

        self.timestamps = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Include timestamps", variable=self.timestamps).pack(anchor="w", pady=8)

        action_row = ttk.Frame(frame)
        action_row.pack(fill="x", pady=(4, 0))
        self.start_button = ttk.Button(action_row, text="START", command=self.start)
        self.start_button.pack(side="left")
        self.progress = ttk.Progressbar(action_row, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.open_button = ttk.Button(action_row, text="Open result", command=self.open_result, state="disabled")
        self.open_button.pack(side="left")

        self.status = ttk.Label(frame, text="", foreground="gray")
        self.status.pack(anchor="w", pady=(8, 0))

    def _on_drop(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self.set_files(Path(p) for p in self.root.tk.splitlist(event.data))  # type: ignore[attr-defined]

    def browse(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(supported_extensions()))
        chosen = filedialog.askopenfilenames(filetypes=[("Audio / video", patterns), ("All files", "*.*")])
        self.set_files(Path(p) for p in chosen)

    def set_files(self, candidates: Iterable[Path]) -> None:
        allowed = supported_extensions()
        files = [p for p in candidates if p.is_file() and p.suffix.lower() in allowed]
        if files:
            self.files = files
        count = len(self.files)
        self.files_label.config(text=self.files[0].name if count == 1 else f"{count} files selected")

    def start(self) -> None:
        if not self.files:
            messagebox.showinfo("Transcribe", "Choose at least one audio file first.")
            return
        self.start_button.config(state="disabled")
        self.open_button.config(state="disabled")
        self.progress.start(12)
        profile = GUI_PROFILES[self.language.get()]
        worker = threading.Thread(
            target=self._work, args=(list(self.files), profile, self.timestamps.get()), daemon=True
        )
        worker.start()
        self.root.after(100, self._poll)

    def _work(self, files: list[Path], profile: GuiProfile, timestamps: bool) -> None:
        try:
            result = run_job(
                files,
                profile,
                timestamps=timestamps,
                paths=self.paths,
                on_line=lambda line: self.events.put(("line", line)),
            )
            self.events.put(("done", result))
        except Exception as error:  # noqa: BLE001 — surface anything to the user, never crash the window
            LOGGER.exception("Transcription job crashed")
            self.events.put(("error", error))

    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                self.root.after(100, self._poll)
                return
            if kind == "line" and str(payload).strip().lstrip("│╭╰─┃━"):  # skip Rich table borders
                self.status.config(text=str(payload)[-90:])
            elif kind in ("done", "error"):
                self._finish(payload)
                return

    def _finish(self, payload: object) -> None:
        self.progress.stop()
        self.start_button.config(state="normal")
        if isinstance(payload, JobResult) and payload.delivered:
            self.last_output_dir = payload.delivered[0].parent
            self.open_button.config(state="normal")
        if isinstance(payload, JobResult) and payload.ok:
            self.status.config(text=f"Done — {len(payload.delivered)} transcript(s) saved next to the audio.")
        elif isinstance(payload, JobResult):
            names = ", ".join(p.name for p in payload.missing)
            self.status.config(text=f"Failed: {names}")
            messagebox.showerror("Transcribe", f"Could not transcribe: {names}")
        else:
            self.status.config(text="Failed.")
            messagebox.showerror("Transcribe", f"Transcription failed: {payload}")

    def open_result(self) -> None:
        if self.last_output_dir is not None:
            open_folder(self.last_output_dir)


def main() -> None:
    setup_logging()
    root, dnd = _make_root()
    TranscribeApp(root, dnd=dnd, paths=default_app_paths())
    root.mainloop()


if __name__ == "__main__":
    main()
