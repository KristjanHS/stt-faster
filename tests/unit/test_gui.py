"""Pure helpers behind backend.gui — staging, command, env, result delivery, CPU retry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from backend.gui import (
    GUI_PROFILES,
    AppPaths,
    build_command,
    build_env,
    read_device,
    run_job,
    stage_files,
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
    """Stands in for the CLI subprocess: writes `<stem>.txt` per staged file unless told to fail."""

    def __init__(self, fail_devices: set[str | None] | None = None, exit_code_on_fail: int = 1) -> None:
        self.fail_devices = fail_devices or set()
        self.exit_code_on_fail = exit_code_on_fail
        self.devices: list[str | None] = []
        self.commands: list[list[str]] = []

    def __call__(self, cmd: list[str], env: dict[str, str], on_line: Callable[[str], None]) -> int:
        device = env.get("STT_DEVICE")
        self.devices.append(device)
        self.commands.append(cmd)
        if device in self.fail_devices:
            return self.exit_code_on_fail
        work_dir = Path(cmd[cmd.index("process") + 1])
        for staged in work_dir.iterdir():
            staged.with_suffix(".txt").write_text(f"text of {staged.name}", encoding="utf-8")
        return 0


def test_stage_files_lowercases_ext_and_dedupes_stems(tmp_path: Path) -> None:
    a_mp3, a_wav = tmp_path / "a.MP3", tmp_path / "sub" / "a.wav"
    a_wav.parent.mkdir()
    a_mp3.write_bytes(b"1")
    a_wav.write_bytes(b"2")

    staged = stage_files([a_mp3, a_wav], tmp_path / "work")

    assert staged == {"a": a_mp3, "a__2": a_wav}
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["a.mp3", "a__2.wav"]


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
def test_run_job_retries_failed_gpu_run_on_cpu_and_persists(paths: AppPaths, audio: Path, exit_code: int) -> None:
    # exit 0 without a .txt also counts as failure: the CLI returns 0 even when files fail.
    write_config_value(paths.config_file, "device", "cuda")
    runner = FakeRunner(fail_devices={"cuda"}, exit_code_on_fail=exit_code)

    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == ["cuda", "cpu"]
    assert result.ok and result.fell_back_to_cpu
    assert read_device(paths.config_file) == "cpu"


def test_run_job_on_cpu_does_not_retry(paths: AppPaths, audio: Path) -> None:
    write_config_value(paths.config_file, "device", "cpu")
    runner = FakeRunner(fail_devices={"cpu"})

    result = run_job([audio], GUI_PROFILES["Estonian"], timestamps=True, paths=paths, runner=runner, base_env={})

    assert runner.devices == ["cpu"]
    assert not result.ok and result.missing == [audio]
    assert not audio.with_suffix(".txt").exists()
