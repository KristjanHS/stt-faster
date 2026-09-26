"""Simple tkinter front-end: pick audio files, choose a language, transcribe via the CLI.

The GUI never transcribes in-process. It copies the chosen files into a private work
dir, runs ``stt-faster transcribe process`` on it as a subprocess, and copies each
resulting ``.txt`` next to the original audio file (never overwriting).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import queue
import re
import shutil
import subprocess  # nosec B404 - runs our own CLI with a fixed argument list
import sys
import tempfile
import threading
import time
import tkinter as tk
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from backend.config import setup_logging
from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.progress import (
    PROGRESS_ENV,
    EtaEstimator,
    JobEtaEstimator,
    ProgressEvent,
    describe_progress,
    format_eta,
    format_progress,
    parse_progress,
)

LOGGER = logging.getLogger(__name__)

APP_NAME = "stt-faster"
GUI_VARIANT = 61
SETUP_EXE_NAME = "Transcribe-Setup.exe"  # mirrors installer.setup_gui.SETUP_EXE_NAME
PYANNOTE_MODULE = "pyannote.audio"
SPEAKERS_MIN, SPEAKERS_MAX, SPEAKERS_DEFAULT = 2, 10, 2  # CLI rejects --num-speakers < 2
EXTRAS_HINT = (
    "Speaker identification needs a free Hugging Face token:\n"
    "1. Create an account at huggingface.co\n"
    "2. Accept the pyannote/speaker-diarization-community-1 licence\n"
    "3. Create a read token at hf.co/settings/tokens and paste it below"
)
EXTRAS_NEED_INSTALL_HINT = "Extras need the installed app (Transcribe-Setup.exe not found)."
# components.FileProcessor logs "Failed to process <file>: <ErrorType>: <message>" per failed file.
_DIARIZATION_FAILURE = re.compile(
    rf"\b(?:{DiarizationConfigError.__name__}|{DiarizationRuntimeError.__name__}): (?P<reason>.+)"
)


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

    @property
    def hf_home(self) -> Path:
        return self.install_dir / "hf"  # the installer's models; mirrors installer.setup_gui.InstallPaths

    @property
    def token_file(self) -> Path:
        return self.config_file.parent / "hf_token"

    @property
    def setup_exe(self) -> Path:
        return self.install_dir / SETUP_EXE_NAME

    @property
    def gui_log(self) -> Path:
        return self.install_dir / "logs" / "gui.log"  # logs\ survives a partial uninstall, like setup.log


def attach_missing_streams(log_file: Path, target: Any = sys) -> None:
    """stt-faster-gui.exe runs windowless with no stdout/stderr; send both — and any crash traceback — to log_file."""
    if target.stdout is not None and target.stderr is not None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  # lives as long as the app
    target.stdout = target.stdout or stream
    target.stderr = target.stderr or stream


def default_app_paths(env: Mapping[str, str] | None = None, plat: str | None = None) -> AppPaths:
    """%LOCALAPPDATA% / %APPDATA% on Windows, XDG dirs elsewhere."""
    env = os.environ if env is None else env
    plat = sys.platform if plat is None else plat
    home = Path.home()

    def env_dir(key: str, default: Path) -> Path:
        value = env.get(key, "")  # an empty value counts as unset (else Path("") → cwd-relative)
        return Path(value) if value else default

    if plat == "win32":
        install = env_dir("LOCALAPPDATA", home / "AppData" / "Local") / APP_NAME
        config_dir = env_dir("APPDATA", home / "AppData" / "Roaming") / APP_NAME
    else:
        install = env_dir("XDG_DATA_HOME", home / ".local" / "share") / APP_NAME
        config_dir = env_dir("XDG_CONFIG_HOME", home / ".config") / APP_NAME
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
    """Configured device (``cpu``/``cuda``, normalised), or None to let the CLI auto-detect."""
    return read_config(config_file).get("device", "").strip().lower() or None


def read_hf_token(token_file: Path) -> str:
    try:  # utf-8-sig + OSError → "" mirror installer.setup_gui.read_hf_token (Notepad adds a BOM)
        return token_file.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def diarization_available(
    token_file: Path, *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> bool:
    """Show "Identify speakers" iff pyannote imports and a non-empty token is saved."""
    try:
        installed = find_spec(PYANNOTE_MODULE) is not None
    except (ImportError, ValueError):  # a dotted name raises when its parent package is missing
        installed = False
    return installed and bool(read_hf_token(token_file))


def extras_install_hint(paths: AppPaths) -> str | None:
    """None when Save & install can run; else why the button is disabled (dev checkout / WSL)."""
    return None if paths.setup_exe.is_file() else EXTRAS_NEED_INSTALL_HINT


def launch_detached(cmd: list[str], cwd: str) -> None:
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(cmd, cwd=cwd, close_fds=True, creationflags=flags)  # noqa: S603  # nosec B603 - our own setup exe


def save_and_install(
    paths: AppPaths,
    token: str,
    *,
    launch: Callable[[list[str], str], None] = launch_detached,
    tempdir: Callable[[], str] = tempfile.gettempdir,
) -> None:
    """Store the HF token, then hand over to ``Transcribe-Setup.exe --extras`` (caller quits)."""
    paths.token_file.parent.mkdir(parents=True, exist_ok=True)
    paths.token_file.touch(mode=0o600)
    paths.token_file.write_text(token.strip(), encoding="utf-8")
    # The shortcut starts us in venv\Scripts; setup inheriting that cwd would read as "app still open".
    launch([str(paths.setup_exe), "--extras"], tempdir())


def diarization_failure(line: str) -> str | None:
    """The reason when ``line`` reports a diarization failure, else None."""
    match = _DIARIZATION_FAILURE.search(line)
    return match.group("reason").strip() if match else None


@cache
def supported_extensions() -> frozenset[str]:
    # Lazy: backend.components pulls in the transcription stack (seconds of import time).
    from backend.components import SUPPORTED_AUDIO_EXTENSIONS  # noqa: PLC0415

    return frozenset(SUPPORTED_AUDIO_EXTENSIONS)


def stage_files(files: Iterable[Path], work_dir: Path) -> dict[str, Path]:
    """Copy ``files`` into ``work_dir`` under unique stems; return staged stem → original.

    Extensions are lower-cased (the CLI globs ``*.mp3`` case-sensitively) and clashing
    stems (``a.mp3`` + ``a.wav``, or ``Meeting.mp3`` + ``meeting.MP3`` — NTFS is
    case-insensitive) get a ``__N`` suffix so their outputs don't collide.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    taken: set[str] = set()
    for original in files:
        stem = original.stem
        n = 2
        while stem.casefold() in taken:
            stem = f"{original.stem}__{n}"
            n += 1
        taken.add(stem.casefold())
        shutil.copy2(original, work_dir / f"{stem}{original.suffix.lower()}")
        staged[stem] = original
    return staged


def build_command(
    work_dir: Path,
    profile: GuiProfile,
    *,
    timestamps: bool,
    diarize: bool = False,
    num_speakers: int = SPEAKERS_DEFAULT,
) -> list[str]:
    # `--variant`, never `-v` (bound to both --variant and --verbose); CLI default is --diarize.
    # One speaker needs no labels, and the CLI rejects --num-speakers < 2.
    speakers = ["--diarize", "--num-speakers", str(num_speakers)] if diarize and num_speakers >= 2 else ["--no-diarize"]
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
        *speakers,
        "--timestamps" if timestamps else "--no-timestamps",
    ]


def build_env(
    base: Mapping[str, str],
    *,
    device: str | None,
    ffmpeg_bin: Path | None,
    hf_home: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, str]:
    env = dict(base)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env[PROGRESS_ENV] = "1"
    if device:
        env["STT_DEVICE"] = device
    if ffmpeg_bin is not None and ffmpeg_bin.is_dir():
        env["PATH"] = os.pathsep.join([str(ffmpeg_bin), env.get("PATH", "")])
    if hf_home is not None and hf_home.is_dir():  # absent in a dev checkout: keep the user's own HF cache
        env.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home / "hub"), HF_XET_CACHE=str(hf_home / "xet"))
    if hf_token:
        env["HF_TOKEN"] = hf_token  # secret: never log env
    return env


RETRY_PREFIX = "Retrying"  # run_job's retry lines; a retry is a new CLI run that restarts at file 1


def renumber_progress(on_line: Callable[[str], None], *, skipped: int, files: int) -> Callable[[str], None]:
    """Relay ``on_line`` with a retry's ``@@progress`` files counted as the last of the job's ``files``.

    The CLI's scan order is not the GUI's list order, so a retry can't name its files' original
    positions; it continues the count instead (``File 4/5`` for the first of 2 retried files).
    """
    if skipped <= 0:
        return on_line

    def relay(line: str) -> None:
        event = parse_progress(line)
        if event is None:
            on_line(line)
        else:  # durations index the retry's subset, not the job
            on_line(format_progress(replace(event, file=event.file + skipped, files=files, durations=None)))

    return relay


def find_outputs(work_dir: Path, staged: Mapping[str, Path]) -> dict[Path, Path | None]:
    """Map each original file to its produced ``.txt`` in ``work_dir`` (None if missing).

    Outputs land anywhere below ``work_dir`` (the CLI moves them into ``processed/``). A
    staged stem matches its exact ``.txt`` stem first, then case-insensitively (a Windows
    filesystem round-trip may change case; staged stems are unique under casefold).
    """
    exact: dict[str, Path] = {}
    folded: dict[str, Path] = {}
    for txt in sorted(work_dir.rglob("*.txt")):
        exact.setdefault(txt.stem, txt)
        folded.setdefault(txt.stem.casefold(), txt)
    return {original: exact.get(stem) or folded.get(stem.casefold()) for stem, original in staged.items()}


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


def run_subprocess(
    cmd: list[str],
    env: dict[str, str],
    on_line: Callable[[str], None],
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
) -> int:
    """Run ``cmd`` streaming merged output lines; ``on_start`` receives the live process."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with subprocess.Popen(  # nosec B603 - fixed command list, no shell
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    ) as proc:
        if on_start is not None:
            on_start(proc)
        for line in proc.stdout or ():
            on_line(line.rstrip())
        return proc.wait()


@dataclass
class JobResult:
    delivered: list[Path] = field(default_factory=list[Path])
    missing: list[Path] = field(default_factory=list[Path])
    fell_back_to_cpu: bool = False
    speakers_skipped: str | None = None  # diarization failure reason, when files were re-run without it

    @property
    def ok(self) -> bool:
        return not self.missing

    @property
    def banner(self) -> str:
        return f"Speakers skipped: {self.speakers_skipped}" if self.speakers_skipped is not None else ""


@dataclass
class _Attempt:
    delivered: list[Path]
    missing: list[Path]
    diarization_error: str | None = None


def _run_attempt(
    files: list[Path],
    device: str | None,
    *,
    diarize: bool,
    num_speakers: int,
    hf_token: str | None,
    profile: GuiProfile,
    timestamps: bool,
    paths: AppPaths,
    runner: Runner,
    on_line: Callable[[str], None],
    base_env: Mapping[str, str],
) -> _Attempt:
    """One CLI run over ``files``: deliver every ``.txt`` produced, note any diarization failure."""
    paths.work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(dir=paths.work_root, prefix=f"job-{device or 'auto'}-"))
    reasons: list[str] = []

    def watch(line: str) -> None:
        if (reason := diarization_failure(line)) is not None:
            reasons.append(reason)
        on_line(line)

    try:
        staged = stage_files(files, work_dir)
        env = build_env(base_env, device=device, ffmpeg_bin=paths.ffmpeg_bin, hf_home=paths.hf_home, hf_token=hf_token)
        cmd = build_command(work_dir, profile, timestamps=timestamps, diarize=diarize, num_speakers=num_speakers)
        exit_code = runner(cmd, env, watch)
        if exit_code != 0:
            LOGGER.warning("Transcriber exited with code %s on device=%s", exit_code, device or "auto")
        outputs = find_outputs(work_dir, staged)
        delivered = deliver_outputs(outputs)
        missing = [original for original, txt in outputs.items() if txt is None]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return _Attempt(delivered, missing, reasons[0] if reasons else None)


def run_job(
    files: list[Path],
    profile: GuiProfile,
    *,
    timestamps: bool,
    paths: AppPaths,
    diarize: bool = False,
    num_speakers: int = SPEAKERS_DEFAULT,
    runner: Runner = run_subprocess,
    on_line: Callable[[str], None] = lambda _line: None,
    base_env: Mapping[str, str] | None = None,
) -> JobResult:
    """Transcribe ``files``; a file succeeded iff its ``.txt`` was produced (exit code only logged).

    Each attempt delivers its outputs immediately. A non-CPU attempt that produces nothing at
    all is re-run once on CPU, and ``device=cpu`` is persisted only if that run produced
    something. A partial failure is reported as-is — never retried, never persisted.
    Independently, files that failed on diarization are re-run once with ``--no-diarize``.
    """
    device = read_device(paths.config_file)
    env = os.environ if base_env is None else base_env
    hf_token = read_hf_token(paths.token_file) if diarize else None
    result = JobResult()
    speakers_on = diarize  # flips off for good after the one --no-diarize retry

    def attempt(batch: list[Path], attempt_device: str | None) -> list[Path]:
        nonlocal speakers_on

        def once(files_now: list[Path], diarize_now: bool) -> _Attempt:
            return _run_attempt(
                files_now,
                attempt_device,
                diarize=diarize_now,
                num_speakers=num_speakers,
                hf_token=hf_token,
                profile=profile,
                timestamps=timestamps,
                paths=paths,
                runner=runner,
                on_line=renumber_progress(on_line, skipped=len(files) - len(files_now), files=len(files)),
                base_env=env,
            )

        run = once(batch, speakers_on)
        delivered = run.delivered
        if speakers_on and run.missing and run.diarization_error is not None:
            LOGGER.warning("Diarization failed; retrying %d file(s) without speakers", len(run.missing))
            on_line(f"{RETRY_PREFIX} without speaker identification…")
            speakers_on = False
            result.speakers_skipped = run.diarization_error
            run = once(run.missing, False)
            delivered = delivered + run.delivered
        result.delivered.extend(delivered)
        result.missing = run.missing
        return delivered

    first = attempt(files, device)
    if device != "cpu" and result.missing and not first:
        LOGGER.warning("No transcripts on device=%s; retrying once on CPU", device or "auto")
        on_line(f"{RETRY_PREFIX} on CPU…")
        if attempt(result.missing, "cpu"):
            write_config_value(paths.config_file, "device", "cpu")
            result.fell_back_to_cpu = True
    return result


def sweep_stale_work_dirs(work_root: Path, *, max_age_s: float = 24 * 3600, now: float | None = None) -> None:
    """Remove work dirs left behind by crashed/killed runs (older than ``max_age_s``)."""
    if not work_root.is_dir():
        return
    now = time.time() if now is None else now
    for child in work_root.iterdir():
        try:
            stale = child.is_dir() and now - child.stat().st_mtime > max_age_s
        except OSError:
            continue
        if stale:
            shutil.rmtree(child, ignore_errors=True)


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
        self.running = False
        self.closing = False
        self.proc: subprocess.Popen[str] | None = None

        sweep_stale_work_dirs(paths.work_root)
        root.title("Transcribe")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.minsize(460, 300)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)

        self.zone = tk.Label(
            frame, text="Click to choose files", relief="groove", borderwidth=2, height=6, cursor="hand2"
        )
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

        self.identify = tk.BooleanVar(value=False)
        self.speakers = tk.IntVar(value=SPEAKERS_DEFAULT)
        self.speakers_shown = diarization_available(paths.token_file)
        if self.speakers_shown:
            speaker_row = ttk.Frame(frame)
            speaker_row.pack(anchor="w", pady=(0, 8))
            ttk.Checkbutton(speaker_row, text="Identify speakers", variable=self.identify).pack(side="left")
            ttk.Label(speaker_row, text="Number of speakers:").pack(side="left", padx=(12, 4))
            ttk.Spinbox(
                speaker_row, from_=SPEAKERS_MIN, to=SPEAKERS_MAX, width=4, textvariable=self.speakers, state="readonly"
            ).pack(side="left")

        self._build_extras(frame)

        action_row = ttk.Frame(frame)
        action_row.pack(fill="x", pady=(4, 0))
        self.start_button = ttk.Button(action_row, text="START", command=self.start)
        self.start_button.pack(side="left")
        self.progress = ttk.Progressbar(action_row, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.open_button = ttk.Button(action_row, text="Open result", command=self.open_result, state="disabled")
        self.open_button.pack(side="left")

        self.detail = ttk.Label(frame, text="")
        self.detail.pack(anchor="w", pady=(8, 0))
        self.eta = EtaEstimator()
        self.job_eta = JobEtaEstimator()
        self.retrying = False
        self.clock: Callable[[], float] = time.monotonic
        self.status = ttk.Label(frame, text="", foreground="gray")
        self.status.pack(anchor="w", pady=(4, 0))
        self.banner = ttk.Label(frame, text="", foreground="#b35c00", wraplength=420)
        self.banner.pack(anchor="w", pady=(4, 0))

    def _build_extras(self, frame: ttk.Frame) -> None:
        self.extras_toggle = ttk.Button(frame, text="▸ Extras", command=self._toggle_extras)
        self.extras_toggle.pack(anchor="w", pady=(0, 8))
        self.extras_frame = ttk.Frame(frame)
        ttk.Label(self.extras_frame, text=EXTRAS_HINT, justify="left").pack(anchor="w")
        self.token = tk.StringVar()
        ttk.Entry(self.extras_frame, textvariable=self.token, show="•", width=48).pack(anchor="w", pady=4)
        install_hint = extras_install_hint(self.paths)
        ttk.Button(
            self.extras_frame,
            text="Save & install",
            command=self.save_and_install,
            state="disabled" if install_hint else "normal",
        ).pack(anchor="w")
        if install_hint:
            ttk.Label(self.extras_frame, text=install_hint, foreground="gray").pack(anchor="w", pady=(4, 0))

    def _toggle_extras(self) -> None:
        if self.extras_frame.winfo_ismapped():
            self.extras_frame.pack_forget()
            self.extras_toggle.config(text="▸ Extras")
        else:
            self.extras_frame.pack(anchor="w", fill="x", pady=(0, 8), after=self.extras_toggle)
            self.extras_toggle.config(text="▾ Extras")

    def save_and_install(self) -> None:
        token = self.token.get().strip()
        if not token:
            messagebox.showinfo("Transcribe", "Paste your Hugging Face token first.", parent=self.root)
            return
        if self.running:
            messagebox.showinfo("Transcribe", "Wait for the transcription to finish first.", parent=self.root)
            return
        try:
            save_and_install(self.paths, token)
        except OSError as error:
            LOGGER.exception("Could not start the Extras install")
            messagebox.showerror("Transcribe", f"Could not start the Extras install: {error}", parent=self.root)
            return
        self.root.destroy()  # setup replaces files in this venv; Windows locks them while we run

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
        self.banner.config(text="")
        self.detail.config(text="Starting…")
        self.eta = EtaEstimator()
        self.job_eta = JobEtaEstimator()
        self.retrying = False
        self.progress.config(mode="indeterminate", value=0)
        self.progress.start(12)
        self.running = True
        profile = GUI_PROFILES[self.language.get()]
        diarize = self.speakers_shown and self.identify.get()
        worker = threading.Thread(
            target=self._work,
            args=(list(self.files), profile, self.timestamps.get(), diarize, self.speakers.get()),
            daemon=True,
        )
        worker.start()
        self.root.after(100, self._poll)

    def _work(self, files: list[Path], profile: GuiProfile, timestamps: bool, diarize: bool, speakers: int) -> None:
        try:
            result = run_job(
                files,
                profile,
                timestamps=timestamps,
                paths=self.paths,
                diarize=diarize,
                num_speakers=speakers,
                runner=lambda cmd, env, on_line: run_subprocess(cmd, env, on_line, on_start=self._track_process),
                on_line=lambda line: self.events.put(("line", line)),
            )
            self.events.put(("done", result))
        except Exception as error:  # noqa: BLE001 — surface anything to the user, never crash the window
            LOGGER.exception("Transcription job crashed")
            self.events.put(("error", error))

    def _track_process(self, proc: subprocess.Popen[str]) -> None:
        # Worker thread. Set proc before reading closing (and _on_close the reverse), so a
        # retry that starts while the window closes is always terminated by one side.
        self.proc = proc
        if self.closing:
            proc.terminate()

    def _on_close(self) -> None:
        if self.running and not messagebox.askokcancel(
            "Transcribe", "A transcription is still running. Stop it and close?", parent=self.root
        ):
            return
        self.closing = True
        proc = self.proc
        if self.running and proc is not None and proc.poll() is None:
            proc.terminate()
        self.root.destroy()

    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                self.root.after(100, self._poll)
                return
            if kind == "line" and (event := parse_progress(str(payload))) is not None:
                self._show_progress(event)
            elif kind == "line" and str(payload).startswith(RETRY_PREFIX):
                self.status.config(text=str(payload))
                self.detail.config(text=str(payload))
                self.progress.config(mode="indeterminate", value=0)
                self.progress.start(12)
                self.eta = EtaEstimator()
                self.job_eta = JobEtaEstimator()
                self.retrying = True
            elif kind == "line" and str(payload).strip().lstrip("│╭╰─┃━"):  # skip Rich table borders
                self.status.config(text=str(payload)[-90:])
            elif kind in ("done", "error"):
                self._finish(payload)
                return

    def _show_progress(self, event: ProgressEvent) -> None:
        """The bar tracks the current stage alone; a stage without quantities pulses."""
        fraction = event.stage_fraction
        determinate = str(self.progress.cget("mode")) == "determinate"
        if fraction is None and determinate:
            self.progress.config(mode="indeterminate", value=0)
            self.progress.start(12)
        elif fraction is not None:
            if not determinate:
                self.progress.stop()
                self.progress.config(mode="determinate")
            self.progress.config(value=fraction * 100)  # default maximum=100; 1.0 makes the pulse jump end to end
        text = describe_progress(event) + (" (retry)" if self.retrying else "")
        now = self.clock()
        stage_left = self.eta.seconds_left(event, now)
        if (job_left := self.job_eta.seconds_left(event, now)) is not None:
            text += f" · {format_eta(job_left)} (all)"
        elif stage_left is not None:
            text += f" · {format_eta(stage_left)}"
        self.detail.config(text=text)

    def _finish(self, payload: object) -> None:
        self.running = False
        self.proc = None
        self.progress.stop()
        succeeded = isinstance(payload, JobResult) and payload.ok
        self.progress.config(mode="determinate", value=100 if succeeded else 0)
        self.detail.config(text="")
        self.start_button.config(state="normal")
        if isinstance(payload, JobResult) and payload.delivered:
            self.last_output_dir = payload.delivered[0].parent
            self.open_button.config(state="normal")
        if isinstance(payload, JobResult):
            self.banner.config(text=payload.banner)
            cpu_note = " (switched to CPU)" if payload.fell_back_to_cpu else ""
            saved = f"{len(payload.delivered)} transcript(s) saved next to the audio"
            if payload.ok:
                self.status.config(text=f"Done — {saved}.{cpu_note}")
            else:
                names = ", ".join(p.name for p in payload.missing)
                self.status.config(text=f"{saved}; failed: {names}{cpu_note}")
                messagebox.showerror("Transcribe", f"Could not transcribe: {names}")
        else:
            self.status.config(text="Failed.")
            messagebox.showerror("Transcribe", f"Transcription failed: {payload}")

    def open_result(self) -> None:
        if self.last_output_dir is not None:
            open_folder(self.last_output_dir)


def main() -> None:
    paths = default_app_paths()
    attach_missing_streams(paths.gui_log)
    setup_logging()
    root, dnd = _make_root()
    TranscribeApp(root, dnd=dnd, paths=paths)
    root.mainloop()


if __name__ == "__main__":
    main()
