from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.steps import loudness


def test_parse_loudnorm_stats_extracts_json() -> None:
    payload = """
        [Parsed_loudnorm_0 @ 0x0] {
            "input_i" : "-27.69",
            "input_tp" : "-5.43",
            "input_lra" : "8.12",
            "input_thresh" : "-38.27",
            "target_offset" : "3.12"
        }
    """

    stats = loudness._parse_loudnorm_stats(payload)  # noqa: SLF001 - exercising helper

    assert stats == {
        "input_i": -27.69,
        "input_tp": -5.43,
        "input_lra": 8.12,
        "input_thresh": -38.27,
        "target_offset": 3.12,
    }


def test_apply_loudnorm_falls_back_on_measurement() -> None:
    calls: list[Dict[str, float] | None] = []

    def fail_measure(_: Path) -> Dict[str, float]:  # noqa: ANN001
        raise StepExecutionError("loudnorm", "analysis failed")

    def record_normalize(
        input_path: Path, output_path: Path, measured: Dict[str, float] | None, sample_rate: int
    ) -> None:
        calls.append(measured)

    result = loudness.apply_loudnorm(
        Path("input.wav"),
        Path("output.wav"),
        sample_rate=16000,
        measure=fail_measure,
        normalize=record_normalize,
    )

    assert calls == [None]  # single-pass
    assert result.name == "loudnorm"


def test_apply_loudnorm_retries_single_pass_after_failure() -> None:
    calls: list[Dict[str, float] | None] = []
    measured = {
        "input_i": -20.0,
        "input_tp": -2.0,
        "input_lra": 5.0,
        "input_thresh": -33.0,
        "target_offset": 0.5,
    }

    def measure(_: Path) -> Dict[str, float]:  # noqa: ANN001
        return measured

    def normalize(input_path: Path, output_path: Path, measured: Dict[str, float] | None, sample_rate: int) -> None:
        calls.append(measured)
        if measured:
            raise StepExecutionError("loudnorm", "first pass failed")

    loudness.apply_loudnorm(
        Path("input.wav"),
        Path("output.wav"),
        sample_rate=16000,
        measure=measure,
        normalize=normalize,
    )

    assert calls == [measured, None]


def test_apply_loudnorm_raises_when_single_pass_fails() -> None:
    def fail_measure(_: Path) -> Dict[str, float]:  # noqa: ANN001
        raise StepExecutionError("loudnorm", "analysis failed")

    def fail_normalize(
        input_path: Path, output_path: Path, measured: Dict[str, float] | None, sample_rate: int
    ) -> None:
        raise StepExecutionError("loudnorm", "normalize failed")

    with pytest.raises(StepExecutionError):
        loudness.apply_loudnorm(
            Path("input.wav"),
            Path("output.wav"),
            sample_rate=16000,
            measure=fail_measure,
            normalize=fail_normalize,
        )


def test_measure_loudness_uses_info_and_parses_stats(tmp_path) -> None:
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"data")

    captured_cmd: list[str] = []

    class _FakeProc:
        def __init__(self, stderr: str):
            self.stderr = stderr

    def fake_run(cmd, check, capture_output, text):  # noqa: ANN001
        nonlocal captured_cmd
        captured_cmd = cmd
        stats = """
            [Parsed_loudnorm_0 @ 0x0] {
                "input_i" : "-30.0",
                "input_tp" : "-6.0",
                "input_lra" : "9.0",
                "input_thresh" : "-40.0",
                "target_offset" : "2.0"
            }
        """
        return _FakeProc(stderr=stats)

    stats = loudness._measure_loudness(fake_audio, runner=fake_run)  # noqa: SLF001 - exercising helper

    assert "-v" in captured_cmd and "info" in captured_cmd
    assert stats == {
        "input_i": -30.0,
        "input_tp": -6.0,
        "input_lra": 9.0,
        "input_thresh": -40.0,
        "target_offset": 2.0,
    }
