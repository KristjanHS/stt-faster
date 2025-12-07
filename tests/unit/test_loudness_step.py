from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from backend.preprocess.errors import StepExecutionError
from backend.preprocess.steps import loudness
from backend.preprocess.steps.denoise_light import _call_reducer


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


def test_create_meter_supports_filter_true_keyword() -> None:
    class MeterWithFilter:
        def __init__(self, sample_rate: int, block_size: float = 0.4, filter_true: bool = False) -> None:
            self.kwargs = {
                "sample_rate": sample_rate,
                "block_size": block_size,
                "filter_true": filter_true,
            }

    pyln = SimpleNamespace(Meter=MeterWithFilter)
    meter = loudness._create_meter(pyln, 44100)

    assert meter.kwargs["filter_true"] is True


def test_create_meter_falls_back_when_filter_true_removed() -> None:
    class MeterWithoutFilter:
        def __init__(self, sample_rate: int, block_size: float = 0.4) -> None:
            self.kwargs = {
                "sample_rate": sample_rate,
                "block_size": block_size,
            }

    class PylnWithoutFilter:
        Meter = MeterWithoutFilter

    meter = loudness._create_meter(PylnWithoutFilter, 48000)

    assert meter.kwargs == {"sample_rate": 48000, "block_size": 0.4}


def test_call_reducer_preserves_noise_clip() -> None:
    def reducer(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    sample = {"y": "wave", "sr": 16000, "noise_clip": "noise"}

    result = _call_reducer(reducer, sample)

    assert result == sample


def test_call_reducer_falls_back_when_noise_clip_removed() -> None:
    call_args: list[dict[str, Any]] = []

    def reducer(**kwargs: Any) -> dict[str, Any]:
        call_args.append(kwargs)
        if "noise_clip" in kwargs and len(call_args) == 1:
            raise TypeError("reduce_noise() got an unexpected keyword argument 'noise_clip'")
        return kwargs

    _call_reducer(reducer, {"y": "wave", "sr": 16000, "noise_clip": "noise"})

    assert len(call_args) == 2
    assert "noise_clip" not in call_args[1]


def test_call_reducer_removes_multiple_unsupported_kwargs() -> None:
    call_args: list[dict[str, Any]] = []

    def reducer(**kwargs: Any) -> dict[str, Any]:
        call_args.append(kwargs.copy())
        if len(call_args) == 1:
            raise TypeError("reduce_noise() got an unexpected keyword argument 'noise_clip'")
        if len(call_args) == 2:
            raise TypeError("reduce_noise() got an unexpected keyword argument 'use_tensor'")
        return kwargs

    _call_reducer(reducer, {"y": "wave", "sr": 16000, "noise_clip": "noise", "use_tensor": False})

    assert len(call_args) == 3
    assert "noise_clip" not in call_args[-1]
    assert "use_tensor" not in call_args[-1]


def test_build_loudnorm_filter_default_preset() -> None:
    """Test that default preset uses I=-20:TP=-2.0:LRA=7."""
    filter_str = loudness._build_loudnorm_filter(preset="default")  # noqa: SLF001 - exercising helper

    assert filter_str.startswith("loudnorm=")
    assert "I=-20.0" in filter_str
    assert "TP=-2.0" in filter_str
    assert "LRA=7.0" in filter_str
    assert "print_format=summary" in filter_str


def test_build_loudnorm_filter_boost_quiet_voices_preset() -> None:
    """Test that boost-quiet-voices preset uses I=-18:TP=-2.0:LRA=6."""
    filter_str = loudness._build_loudnorm_filter(preset="boost-quiet-voices")  # noqa: SLF001 - exercising helper

    assert filter_str.startswith("loudnorm=")
    assert "I=-18.0" in filter_str
    assert "TP=-2.0" in filter_str
    assert "LRA=6.0" in filter_str
    assert "print_format=summary" in filter_str


def test_build_loudnorm_filter_unknown_preset_falls_back_to_default() -> None:
    """Test that unknown preset falls back to default."""
    filter_str = loudness._build_loudnorm_filter(preset="unknown-preset")  # noqa: SLF001 - exercising helper

    assert filter_str.startswith("loudnorm=")
    assert "I=-20.0" in filter_str
    assert "TP=-2.0" in filter_str
    assert "LRA=7.0" in filter_str


def test_measure_loudness_passes_preset_to_filter(tmp_path) -> None:
    """Test that _measure_loudness passes preset to filter builder."""
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

    loudness._measure_loudness(fake_audio, preset="boost-quiet-voices", runner=fake_run)  # noqa: SLF001 - exercising helper

    # Find the -af argument and check the filter string
    af_index = captured_cmd.index("-af")
    filter_str = captured_cmd[af_index + 1]
    assert "I=-18.0" in filter_str
    assert "LRA=6.0" in filter_str
