"""Pure helpers behind backend.gui — staging, command, env, result delivery, CPU retry."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.processor import TranscriptionProcessor
from backend.progress import ProgressEvent
from backend.run_config import RunConfig
from backend.run_log import JsonlRunLog
from backend.services.factory import ServiceFactory
from backend.services.interfaces import TranscriptionRequest, TranscriptionResult
from installer import setup_gui
from backend import gui
from types import SimpleNamespace

from backend.gui import (
    EXTRAS_NEED_INSTALL_HINT,
    GUI_PROFILES,
    PYANNOTE_MODULE,
    AppPaths,
    attach_missing_streams,
    build_command,
    build_env,
    default_app_paths,
    describe_progress,
    diarization_available,
    diarization_failure,
    extras_install_hint,
    find_outputs,
    read_device,
    read_hf_token,
    run_job,
    save_and_install,
    stage_files,
    sweep_stale_work_dirs,
    unique_destination,
    write_config_value,
)


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths(install_dir=tmp_path / "install", config_file=tmp_path / "cfg" / "config")


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    audio = src / "meeting.M4A"
    audio.write_bytes(b"fake")
    return audio


class FakeRunner:
    """Stands in for the CLI subprocess: writes `<stem>.txt` per staged file unless told to fail.

    ``processed=True`` mimics the real CLI, which moves results into ``<work>/processed/``;
    ``fail_stems`` (staged stems) produce no output, as when the CLI fails individual files;
    ``diarize_error`` fails every file of a ``--diarize`` run, logged as components.FileProcessor does.
    """

    def __init__(
        self,
        fail_devices: set[str | None] | None = None,
        exit_code_on_fail: int = 1,
        *,
        fail_stems: set[str] | None = None,
        exit_code: int = 0,
        processed: bool = False,
        diarize_error: Exception | None = None,
    ) -> None:
        self.fail_devices = fail_devices or set()
        self.exit_code_on_fail = exit_code_on_fail
        self.fail_stems = fail_stems or set()
        self.exit_code = exit_code
        self.processed = processed
        self.diarize_error = diarize_error
        self.envs: list[dict[str, str]] = []
        self.devices: list[str | None] = []
        self.commands: list[list[str]] = []
        self.staged_names: list[list[str]] = []

    def __call__(self, cmd: list[str], env: dict[str, str], on_line: Callable[[str], None]) -> int:
        device = env.get("STT_DEVICE")
        self.devices.append(device)
        self.envs.append(env)
        self.commands.append(cmd)
        work_dir = Path(cmd[cmd.index("process") + 1])
        inputs = sorted(p for p in work_dir.iterdir() if p.is_file())
        self.staged_names.append([p.name for p in inputs])
        if device in self.fail_devices:
            return self.exit_code_on_fail
        if self.diarize_error is not None and "--diarize" in cmd:
            error = self.diarize_error
            for staged in inputs:
                on_line(f"Failed to process {staged}: {type(error).__name__}: {error}")
            return self.exit_code
        out_dir = work_dir / "processed" if self.processed else work_dir
        out_dir.mkdir(exist_ok=True)
        for staged in inputs:
            if staged.stem not in self.fail_stems:
                (out_dir / f"{staged.stem}.txt").write_text(f"text of {staged.name}", encoding="utf-8")
        return self.exit_code


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


def test_stage_files_lowercases_ext_and_dedupes_stems(tmp_path: Path) -> None:
    a_mp3, a_wav = tmp_path / "a.MP3", tmp_path / "sub" / "a.wav"
    a_wav.parent.mkdir()
    a_mp3.write_bytes(b"1")
    a_wav.write_bytes(b"2")

    staged = stage_files([a_mp3, a_wav], tmp_path / "work")

    assert staged == {"a": a_mp3, "a__2": a_wav}
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["a.mp3", "a__2.wav"]


def test_case_only_stem_collision_maps_each_output_to_its_own_original(paths: AppPaths, tmp_path: Path) -> None:
    # NTFS is case-insensitive: Meeting.mp3 and meeting.mp3 would be one staged file there.
    upper, lower = _write(tmp_path / "D1" / "Meeting.mp3"), _write(tmp_path / "D2" / "meeting.MP3")
    assert stage_files([upper, lower], tmp_path / "work") == {"Meeting": upper, "meeting__2": lower}

    runner = FakeRunner(processed=True)
    result = run_job([upper, lower], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.staged_names == [["Meeting.mp3", "meeting__2.mp3"]]
    assert result.ok
    assert upper.with_suffix(".txt").read_text(encoding="utf-8") == "text of Meeting.mp3"
    assert lower.with_suffix(".txt").read_text(encoding="utf-8") == "text of meeting__2.mp3"


def test_find_outputs_matches_stem_case_insensitively(tmp_path: Path) -> None:
    original = tmp_path / "Meeting.mp3"
    produced = _write(tmp_path / "work" / "processed" / "meeting.txt")
    assert find_outputs(tmp_path / "work", {"Meeting": original}) == {original: produced}


def test_build_command_mirrors_teams_bat(tmp_path: Path) -> None:
    cmd = build_command(tmp_path, GUI_PROFILES["English"], timestamps=False)
    tail = cmd[cmd.index("process") :]
    assert tail == [
        "process",
        str(tmp_path),
        "--preset",
        "turbo",
        "--variant",
        "61",
        "--language",
        "en",
        "--output-format",
        "txt",
        "--no-diarize",
        "--no-timestamps",
    ]
    assert "-v" not in cmd


def test_build_env_sets_device_and_prepends_ffmpeg(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    env = build_env({"PATH": "/usr/bin"}, device="cpu", ffmpeg_bin=tmp_path / "bin")
    assert env["STT_DEVICE"] == "cpu"
    assert env["PATH"].startswith(str(tmp_path / "bin"))


def test_build_env_leaves_device_unset_without_config() -> None:
    assert "STT_DEVICE" not in build_env({}, device=None, ffmpeg_bin=None)


def test_build_env_points_hf_caches_at_installed_models_over_inherited(tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    inherited = {"HF_HUB_CACHE": "/shared/hub", "HF_HOME": "/shared"}
    assert build_env(inherited, device=None, ffmpeg_bin=None, hf_home=hf) == {**inherited, **_py_env()}  # no dir yet
    hf.mkdir()
    env = build_env(inherited, device=None, ffmpeg_bin=None, hf_home=hf)
    assert (env["HF_HOME"], env["HF_HUB_CACHE"], env["HF_XET_CACHE"]) == (str(hf), str(hf / "hub"), str(hf / "xet"))


def _py_env() -> dict[str, str]:
    return {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "STT_PROGRESS": "1"}


@pytest.mark.skipif(sys.platform == "win32", reason="exercises the XDG branch")
def test_default_app_paths_treats_empty_xdg_as_unset() -> None:
    paths = default_app_paths({"XDG_DATA_HOME": "", "XDG_CONFIG_HOME": ""})
    assert paths.install_dir == Path.home() / ".local" / "share" / "stt-faster"
    assert paths.config_file == Path.home() / ".config" / "stt-faster" / "config"


def test_sweep_removes_only_work_dirs_older_than_a_day(tmp_path: Path) -> None:
    now = 1_000_000_000.0
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir()
    fresh.mkdir()
    os.utime(old, (now - 25 * 3600, now - 25 * 3600))
    os.utime(fresh, (now - 23 * 3600, now - 23 * 3600))

    sweep_stale_work_dirs(tmp_path, now=now)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["fresh"]


def test_unique_destination_never_overwrites(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.m4a"
    assert unique_destination(audio) == tmp_path / "meeting.txt"
    (tmp_path / "meeting.txt").write_text("edited")
    (tmp_path / "meeting (2).txt").write_text("older")
    assert unique_destination(audio) == tmp_path / "meeting (3).txt"


def test_config_round_trip_keeps_other_keys(paths: AppPaths) -> None:
    assert read_device(paths.config_file) is None
    write_config_value(paths.config_file, "token", "abc")
    write_config_value(paths.config_file, "device", "cuda")
    assert read_device(paths.config_file) == "cuda"
    assert "token=abc" in paths.config_file.read_text(encoding="utf-8")


def test_run_job_delivers_txt_next_to_original(paths: AppPaths, audio: Path) -> None:
    runner = FakeRunner()
    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert result.ok and not result.fell_back_to_cpu
    assert result.delivered == [audio.with_suffix(".txt")]
    assert audio.with_suffix(".txt").read_text(encoding="utf-8") == "text of meeting.m4a"
    assert audio.exists()
    assert runner.devices == [None]
    assert not any(paths.work_root.iterdir())


@pytest.mark.parametrize("exit_code", [1, 0])
@pytest.mark.parametrize(("configured", "first_device"), [(None, None), (" CUDA ", "cuda")])
def test_run_job_retries_whole_job_failure_on_cpu_and_persists(
    paths: AppPaths, audio: Path, exit_code: int, configured: str | None, first_device: str | None
) -> None:
    # exit 0 without a .txt also counts as failure: the CLI returns 0 even when files fail.
    if configured is not None:
        write_config_value(paths.config_file, "device", configured)
    runner = FakeRunner(fail_devices={first_device}, exit_code_on_fail=exit_code, processed=True)

    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == [first_device, "cpu"]
    assert result.ok and result.fell_back_to_cpu
    assert result.delivered == [audio.with_suffix(".txt")]
    assert read_device(paths.config_file) == "cpu"


def test_run_job_cpu_retry_that_also_fails_does_not_persist(paths: AppPaths, audio: Path) -> None:
    runner = FakeRunner(fail_devices={None, "cpu"})

    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == [None, "cpu"]
    assert result.missing == [audio] and not result.fell_back_to_cpu
    assert not paths.config_file.exists()


def test_run_job_partial_gpu_failure_delivers_good_files_without_retry(paths: AppPaths, tmp_path: Path) -> None:
    write_config_value(paths.config_file, "device", "cuda")
    good, bad = _write(tmp_path / "src" / "good.mp3"), _write(tmp_path / "src" / "bad.mp3")
    runner = FakeRunner(fail_stems={"bad"}, exit_code=1, processed=True)

    result = run_job([good, bad], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == ["cuda"]
    assert result.delivered == [good.with_suffix(".txt")]
    assert result.missing == [bad] and not result.fell_back_to_cpu
    assert good.with_suffix(".txt").read_text(encoding="utf-8") == "text of good.mp3"
    assert read_device(paths.config_file) == "cuda"
    assert not any(paths.work_root.iterdir())


def test_run_job_nonzero_exit_with_all_outputs_is_success(
    paths: AppPaths, audio: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = FakeRunner(exit_code=2, processed=True)

    with caplog.at_level(logging.WARNING, logger="backend.gui"):
        result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == [None]
    assert result.ok and result.delivered == [audio.with_suffix(".txt")]
    assert not paths.config_file.exists()
    assert "exited with code 2" in caplog.text


def test_run_job_on_cpu_does_not_retry(paths: AppPaths, audio: Path) -> None:
    write_config_value(paths.config_file, "device", "CPU")  # normalised: still no retry
    runner = FakeRunner(fail_devices={"cpu"})

    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == ["cpu"]
    assert not result.ok and result.missing == [audio]
    assert not audio.with_suffix(".txt").exists()


def _speaker_flags(cmd: list[str]) -> list[str]:
    return [arg for arg in cmd if arg in ("--diarize", "--no-diarize")]


@pytest.mark.parametrize(
    ("diarize", "speakers", "expected"),
    [(True, 3, ["--diarize", "--num-speakers", "3"]), (True, 1, ["--no-diarize"]), (False, 3, ["--no-diarize"])],
)
def test_build_command_speaker_args(tmp_path: Path, diarize: bool, speakers: int, expected: list[str]) -> None:
    # 1 speaker → --no-diarize: the CLI rejects --num-speakers < 2.
    cmd = build_command(tmp_path, GUI_PROFILES["English"], timestamps=True, diarize=diarize, num_speakers=speakers)
    assert cmd[cmd.index("txt") + 1 : cmd.index("--timestamps")] == expected


def test_build_env_sets_hf_token_only_when_given() -> None:
    assert build_env({}, device=None, ffmpeg_bin=None, hf_token="hf_abc")["HF_TOKEN"] == "hf_abc"
    assert "HF_TOKEN" not in build_env({}, device=None, ffmpeg_bin=None, hf_token=None)


@pytest.mark.parametrize(
    ("installed", "token", "shown"),
    [(True, "hf_abc", True), (True, " \n", False), (False, "hf_abc", False), (False, None, False)],
)
def test_identify_speakers_shown_iff_pyannote_and_token(
    paths: AppPaths, installed: bool, token: str | None, shown: bool
) -> None:
    if token is not None:
        paths.token_file.parent.mkdir(parents=True, exist_ok=True)
        paths.token_file.write_text(token, encoding="utf-8")
    probed: list[str] = []

    def find_spec(name: str) -> object | None:
        probed.append(name)
        return object() if installed else None

    assert diarization_available(paths.token_file, find_spec=find_spec) is shown
    assert probed == [PYANNOTE_MODULE]


def test_identify_speakers_hidden_when_pyannote_parent_is_missing(paths: AppPaths) -> None:
    paths.token_file.parent.mkdir(parents=True, exist_ok=True)
    paths.token_file.write_text("hf_abc", encoding="utf-8")

    def find_spec(name: str) -> object | None:
        raise ModuleNotFoundError(name)

    assert diarization_available(paths.token_file, find_spec=find_spec) is False


def test_save_and_install_writes_token_then_launches_setup_extras(paths: AppPaths) -> None:
    launched: list[tuple[list[str], str]] = []

    def launch(cmd: list[str], cwd: str) -> None:
        assert paths.token_file.read_text(encoding="utf-8") == "hf_abc"  # saved before setup starts
        launched.append((cmd, cwd))

    save_and_install(paths, "  hf_abc\n", launch=launch)

    [(cmd, cwd)] = launched
    assert cmd == [str(paths.install_dir / "Transcribe-Setup.exe"), "--extras"]
    # The shortcut parks the app in venv\Scripts; setup inheriting that cwd sees the app as "still open".
    assert not Path(cwd).resolve().is_relative_to(paths.install_dir / ".venv")
    assert paths.token_file == paths.config_file.parent / "hf_token"


def test_read_hf_token_strips_bom_and_treats_unreadable_as_none(paths: AppPaths) -> None:
    paths.token_file.parent.mkdir(parents=True)
    paths.token_file.write_text("\ufeffhf_abc\r\n", encoding="utf-8")  # Notepad's UTF-8 BOM
    assert read_hf_token(paths.token_file) == "hf_abc"
    paths.token_file.unlink()
    paths.token_file.mkdir()  # a directory: read_text raises OSError
    assert read_hf_token(paths.token_file) == ""


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_app_and_installer_agree_on_shared_paths(tmp_path: Path, plat: str) -> None:
    env = {k: str(tmp_path / k) for k in ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME")}
    app = gui.default_app_paths(env, plat)
    setup = setup_gui.default_install_paths(env, plat)
    assert gui.SETUP_EXE_NAME == setup_gui.SETUP_EXE_NAME
    assert app.config_file == setup.config_file
    assert app.token_file == setup.hf_token_file
    assert app.setup_exe == setup.setup_copy


def test_extras_install_disabled_without_setup_exe(paths: AppPaths) -> None:
    assert extras_install_hint(paths) == EXTRAS_NEED_INSTALL_HINT
    _write(paths.setup_exe)
    assert extras_install_hint(paths) is None


def _save_token(paths: AppPaths, token: str = "hf_abc") -> None:
    paths.token_file.parent.mkdir(parents=True, exist_ok=True)
    paths.token_file.write_text(token, encoding="utf-8")


@pytest.mark.parametrize("error_cls", [DiarizationConfigError, DiarizationRuntimeError])
def test_run_job_diarization_failure_reruns_once_without_speakers(
    paths: AppPaths, audio: Path, error_cls: type[Exception]
) -> None:
    _save_token(paths)
    runner = FakeRunner(diarize_error=error_cls("HF_TOKEN was rejected (401)"), processed=True)

    result = run_job(
        [audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, diarize=True, runner=runner, base_env={}
    )

    assert [_speaker_flags(cmd) for cmd in runner.commands] == [["--diarize"], ["--no-diarize"]]
    assert runner.devices == [None, None]
    assert runner.envs[0]["HF_TOKEN"] == "hf_abc"
    assert result.ok and result.delivered == [audio.with_suffix(".txt")]
    assert not result.fell_back_to_cpu and not paths.config_file.exists()
    assert result.banner == "Speakers skipped: HF_TOKEN was rejected (401)"


def test_run_job_non_diarization_failure_is_not_retried_without_speakers(paths: AppPaths, audio: Path) -> None:
    write_config_value(paths.config_file, "device", "cpu")
    runner = FakeRunner(fail_stems={"meeting"}, processed=True)

    result = run_job(
        [audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, diarize=True, runner=runner, base_env={}
    )

    assert [_speaker_flags(cmd) for cmd in runner.commands] == [["--diarize"]]
    assert result.missing == [audio] and result.speakers_skipped is None and result.banner == ""


def test_run_job_cpu_and_diarization_retries_compose_once_each(paths: AppPaths, audio: Path) -> None:
    runner = FakeRunner(fail_devices={None}, diarize_error=DiarizationConfigError("licence"), processed=True)

    result = run_job(
        [audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, diarize=True, runner=runner, base_env={}
    )

    assert runner.devices == [None, "cpu", "cpu"]
    assert [_speaker_flags(cmd) for cmd in runner.commands] == [["--diarize"], ["--diarize"], ["--no-diarize"]]
    assert result.ok and result.fell_back_to_cpu and read_device(paths.config_file) == "cpu"
    assert result.banner == "Speakers skipped: licence"


def test_run_job_diarization_retry_that_fails_on_gpu_falls_back_to_cpu_without_speakers(
    paths: AppPaths, audio: Path
) -> None:
    write_config_value(paths.config_file, "device", "cuda")
    runner = FakeRunner(diarize_error=DiarizationConfigError("licence"), processed=True)

    def gpu_breaks_after_first(cmd: list[str], env: dict[str, str], on_line: Callable[[str], None]) -> int:
        if len(runner.commands) == 1:  # the diarized GPU run got as far as pyannote
            runner.fail_devices.add("cuda")
        return runner(cmd, env, on_line)

    result = run_job(
        [audio],
        GUI_PROFILES["Estonian"],
        timestamps=True,
        paths=paths,
        diarize=True,
        runner=gpu_breaks_after_first,
        base_env={},
    )

    assert runner.devices == ["cuda", "cuda", "cpu"]
    assert [_speaker_flags(cmd) for cmd in runner.commands] == [["--diarize"], ["--no-diarize"], ["--no-diarize"]]
    assert result.ok and result.fell_back_to_cpu


class _DiarizationFailingService:
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise DiarizationConfigError("HuggingFace returned 403")


def test_diarization_failure_parses_the_real_per_file_failure_log(
    tmp_path: Path, audio: Path, caplog: pytest.LogCaptureFixture
) -> None:
    processor = TranscriptionProcessor(
        transcription_service=_DiarizationFailingService(),
        run_log=JsonlRunLog(tmp_path / "runs.jsonl"),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        run_config=RunConfig.from_env_and_variant(audio.parent, None),
        disable_file_moving=True,
    )

    with caplog.at_level(logging.ERROR, logger="backend.components"):
        assert processor.process_file(str(audio)).status == "failed"

    reasons = [diarization_failure(record.getMessage()) for record in caplog.records]
    assert "HuggingFace returned 403" in reasons


def test_windowless_app_writes_its_streams_to_the_gui_log(tmp_path: Path) -> None:
    # stt-faster-gui.exe gives the app no stdout/stderr; setup_logging's sys.stderr.isatty() crashed on None.
    log = tmp_path / "logs" / "gui.log"
    windowless = SimpleNamespace(stdout=None, stderr=None)
    attach_missing_streams(log, windowless)
    assert windowless.stderr.isatty() is False
    windowless.stderr.write("Traceback: boom\n")
    windowless.stdout.write("hello\n")
    assert log.read_text(encoding="utf-8") == "Traceback: boom\nhello\n"
    windowless.stderr.close()
    console = SimpleNamespace(stdout=object(), stderr=object())
    before = vars(console).copy()
    attach_missing_streams(tmp_path / "unused.log", console)
    assert vars(console) == before and not (tmp_path / "unused.log").exists()


def test_describe_progress_names_file_stage_and_substep() -> None:
    event = ProgressEvent(file=2, files=4, stage="transcribe", done=30.0, total=60.0)
    assert describe_progress(event) == "File 2/4 · Transcribing"
    event = ProgressEvent(file=1, files=3, stage="diarize", detail="embeddings")
    assert describe_progress(event) == "File 1/3 · Identifying speakers · embeddings"


class _FakeBar:
    def __init__(self) -> None:
        self.options: dict[str, object] = {"mode": "indeterminate", "maximum": 100, "value": 0}
        self.running = False

    def cget(self, key: str) -> object:
        return self.options[key]

    def config(self, **options: object) -> None:
        self.options.update(options)

    def start(self, _interval: int) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


def test_bar_tracks_each_stage_and_pulses_on_a_bare_one() -> None:
    bar, detail = _FakeBar(), _FakeBar()
    app = SimpleNamespace(progress=bar, detail=detail)
    show = gui.TranscribeApp._show_progress  # pyright: ignore[reportPrivateUsage]

    show(app, ProgressEvent(1, 2, "transcribe", 30.0, 60.0))  # type: ignore[arg-type]
    assert (bar.options["mode"], bar.options["value"], bar.running) == ("determinate", 50.0, False)
    show(app, ProgressEvent(1, 2, "diarize"))  # type: ignore[arg-type]
    assert (bar.options["mode"], bar.running) == ("indeterminate", True)
    assert bar.options["maximum"] == 100  # at 1.0 ttk's pulse jumps end to end each tick
    show(app, ProgressEvent(1, 2, "diarize", 1.0, 4.0, "embeddings"))  # type: ignore[arg-type]
    assert (bar.options["mode"], bar.options["value"], bar.running) == ("determinate", 25.0, False)
    assert detail.options["text"] == "File 1/2 · Identifying speakers · embeddings"
