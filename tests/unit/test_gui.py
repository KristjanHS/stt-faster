"""Pure helpers behind backend.gui — staging, command, env, result delivery, CPU retry."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.gui import (
    GUI_PROFILES,
    AppPaths,
    build_command,
    build_env,
    default_app_paths,
    find_outputs,
    read_device,
    run_job,
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
    ``fail_stems`` (staged stems) produce no output, as when the CLI fails individual files.
    """

    def __init__(
        self,
        fail_devices: set[str | None] | None = None,
        exit_code_on_fail: int = 1,
        *,
        fail_stems: set[str] | None = None,
        exit_code: int = 0,
        processed: bool = False,
    ) -> None:
        self.fail_devices = fail_devices or set()
        self.exit_code_on_fail = exit_code_on_fail
        self.fail_stems = fail_stems or set()
        self.exit_code = exit_code
        self.processed = processed
        self.devices: list[str | None] = []
        self.commands: list[list[str]] = []
        self.staged_names: list[list[str]] = []

    def __call__(self, cmd: list[str], env: dict[str, str], on_line: Callable[[str], None]) -> int:
        device = env.get("STT_DEVICE")
        self.devices.append(device)
        self.commands.append(cmd)
        work_dir = Path(cmd[cmd.index("process") + 1])
        inputs = sorted(p for p in work_dir.iterdir() if p.is_file())
        self.staged_names.append([p.name for p in inputs])
        if device in self.fail_devices:
            return self.exit_code_on_fail
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
    return {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


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
