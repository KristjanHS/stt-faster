"""Stage C guardrail: every registered Step type executes against a small
audio fixture and produces snapshot-stable output.

Stage C flattens `backend/variants/steps.py` (the four Executor classes plus
the per-step `Step` glue collapse to module-level functions + a dispatch
table). The refactor must preserve byte-/property-identical output for every
step type — this test is the gate.

The test enumerates `StepRegistry.get_registered_types()` and for each:
  1. constructs the step (default config, with overrides for configs that
     have required fields),
  2. executes against a transcoded copy of `tests/test_short.mp3`,
  3. asserts non-empty output,
  4. computes a `(sample_count, rms_bucket)` snapshot and compares it against
     `tests/integration/fixtures/step_smoke_snapshots.json`.

Regenerate snapshots after deliberate output-shape changes with:

    UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest \
        tests/integration/test_every_step_smoke.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from backend.preprocess.config import PreprocessConfig
from backend.variants.steps import (
    StepConfig,
    StepRegistry,
    VolumeLimiterStepConfig,
)

_SNAPSHOT_PATH = Path("tests/integration/fixtures/step_smoke_snapshots.json")
_RMS_BUCKET_RESOLUTION_DB = 2  # 2-dB resolution — robust to small ffmpeg drift.

# Step types whose default config has a required field without a default.
# `StepRegistry.create_step(step_type, None)` would fall back to
# `get_default_config()` which constructs the dataclass — that fails when a
# field has no default. Only `volume_limiter` is affected today.
_CONFIG_OVERRIDES: dict[str, StepConfig] = {
    "volume_limiter": VolumeLimiterStepConfig(volume_db=6.0),
}


def _wav_snapshot(wav_path: Path) -> tuple[int, int]:
    """Return (sample_count, rms_bucket) for a 16-bit PCM WAV."""
    with wave.open(str(wav_path), "rb") as w:
        nframes = w.getnframes()
        sampwidth = w.getsampwidth()
        nchannels = w.getnchannels()
        raw = w.readframes(nframes)
    if sampwidth != 2:
        pytest.fail(f"Snapshot expects 16-bit PCM, got {sampwidth * 8}-bit: {wav_path}")
    sample_count = nframes * nchannels
    if sample_count == 0:
        return 0, 0
    samples = struct.unpack(f"<{sample_count}h", raw)
    sum_sq = sum(s * s for s in samples)
    rms = math.sqrt(sum_sq / sample_count)
    if rms <= 0:
        return sample_count, -120
    rms_db = 20.0 * math.log10(rms / 32768.0)
    rms_bucket = int(round(rms_db / _RMS_BUCKET_RESOLUTION_DB))
    return sample_count, rms_bucket


@pytest.fixture(scope="module")
def _input_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Mono 16 kHz 16-bit WAV transcoded from tests/test_short.mp3."""
    src = Path("tests/test_short.mp3")
    if not src.exists():
        pytest.fail("tests/test_short.mp3 required for step smoke test.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("ffmpeg required for step smoke test.")
    wav_path = tmp_path_factory.mktemp("step_smoke_input") / "input.wav"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"ffmpeg input prep failed: {result.stderr}")
    return wav_path


@pytest.fixture(scope="module")
def _preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(target_sample_rate=16_000, target_channels=1)


@pytest.mark.parametrize("step_type", StepRegistry.get_registered_types())
def test_step_executes_and_matches_snapshot(
    step_type: str,
    _input_wav: Path,
    _preprocess_config: PreprocessConfig,
    tmp_path: Path,
) -> None:
    config = _CONFIG_OVERRIDES.get(step_type)
    step = StepRegistry.create_step(step_type, config)
    output = tmp_path / f"{step_type}.wav"

    step.execute(
        input_path=_input_wav,
        output_path=output,
        global_config=_preprocess_config,
        step_index=0,
    )

    assert output.exists(), f"{step_type}: output file not created"
    assert output.stat().st_size > 0, f"{step_type}: output file is empty"

    sample_count, rms_bucket = _wav_snapshot(output)
    assert sample_count > 0, f"{step_type}: WAV has zero samples"

    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        current: dict[str, list[int]] = {}
        if _SNAPSHOT_PATH.exists():
            current = json.loads(_SNAPSHOT_PATH.read_text())
        current[step_type] = [sample_count, rms_bucket]
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n")
        return

    if not _SNAPSHOT_PATH.exists():
        pytest.fail(
            f"Snapshot file {_SNAPSHOT_PATH} missing. "
            "Bootstrap with `UPDATE_SNAPSHOTS=1 pytest "
            "tests/integration/test_every_step_smoke.py`."
        )
    snapshots = json.loads(_SNAPSHOT_PATH.read_text())
    expected = snapshots.get(step_type)
    if expected is None:
        pytest.fail(
            f"No snapshot recorded for step_type={step_type!r}. "
            "Bootstrap with `UPDATE_SNAPSHOTS=1 pytest "
            "tests/integration/test_every_step_smoke.py`."
        )
    assert [sample_count, rms_bucket] == expected, (
        f"{step_type}: snapshot drift — got [{sample_count}, {rms_bucket}], expected {expected}"
    )
