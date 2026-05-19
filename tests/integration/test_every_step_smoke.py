"""Stage C guardrail: every registered Step type executes against a small
audio fixture and produces snapshot-stable output.

Stage C flattens `backend/variants/steps.py` (the four Executor classes plus
the per-step `Step` glue collapse to module-level functions + a dispatch
table). The refactor must preserve property-identical output for every step
type — this test is the gate.

The test enumerates `StepRegistry.get_registered_types()` and for each:
  1. constructs the step (registry-default config),
  2. executes against a transcoded copy of `tests/test_short.mp3`,
  3. asserts non-empty output,
  4. snapshots `(sample_count, rms_bucket)` and compares against
     `tests/integration/fixtures/step_smoke_snapshots.json`.

Known limitation (logged for Stage C.2): the 2-dB RMS bucket does not
discriminate between step types that share a loudness bucket — about half
of the 18 step types share a bucket with at least one neighbour. A
dispatch-table swap between such steps during C.2 would pass the snapshot
silently. The C.2 implementer should additionally spot-check
`scripts/variant_checks/verify_all_variants.py` (named in the plan's gate)
to catch routing errors that escape this gate. A first-N-sample MD5 was
tried but failed on `sox_peak_normalize`, which is not byte-deterministic
between runs.

Regenerate snapshots after deliberate output-shape changes with:

    UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest \\
        tests/integration/test_every_step_smoke.py -p no:xdist
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
from backend.variants.steps import StepRegistry

_TESTS_DIR = Path(__file__).resolve().parent.parent
_SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures" / "step_smoke_snapshots.json"
_INPUT_MP3 = _TESTS_DIR / "test_short.mp3"
_RMS_BUCKET_RESOLUTION_DB = 2  # 2-dB resolution — robust to small ffmpeg drift.


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
    if not _INPUT_MP3.exists():
        pytest.fail(f"{_INPUT_MP3} required for step smoke test.")
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
            str(_INPUT_MP3),
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


@pytest.mark.parametrize("step_type", StepRegistry.get_registered_types())
def test_step_executes_and_matches_snapshot(
    step_type: str,
    _input_wav: Path,
    tmp_path: Path,
) -> None:
    if os.environ.get("UPDATE_SNAPSHOTS") == "1" and os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("UPDATE_SNAPSHOTS=1 races under pytest-xdist; rerun with -p no:xdist")

    preprocess_config = PreprocessConfig(target_sample_rate=16_000, target_channels=1)
    step = StepRegistry.create_step(step_type, None)
    output = tmp_path / f"{step_type}.wav"

    step.execute(
        input_path=_input_wav,
        output_path=output,
        global_config=preprocess_config,
        step_index=0,
    )

    assert output.exists(), f"{step_type}: output file not created"
    assert output.stat().st_size > 0, f"{step_type}: output file is empty"

    sample_count, rms_bucket = _wav_snapshot(output)
    assert sample_count > 0, f"{step_type}: WAV has zero samples"
    actual: list[int] = [sample_count, rms_bucket]

    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        current: dict[str, list[int]] = {}
        if _SNAPSHOT_PATH.exists():
            current = json.loads(_SNAPSHOT_PATH.read_text())
        current[step_type] = actual
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n")
        return

    if not _SNAPSHOT_PATH.exists():
        pytest.fail(
            f"Snapshot file {_SNAPSHOT_PATH} missing. "
            "Bootstrap with `UPDATE_SNAPSHOTS=1 pytest "
            "tests/integration/test_every_step_smoke.py -p no:xdist`."
        )
    snapshots = json.loads(_SNAPSHOT_PATH.read_text())
    expected = snapshots.get(step_type)
    if expected is None:
        pytest.fail(
            f"No snapshot recorded for step_type={step_type!r}. "
            "Bootstrap with `UPDATE_SNAPSHOTS=1 pytest "
            "tests/integration/test_every_step_smoke.py -p no:xdist`."
        )
    assert actual == expected, f"{step_type}: snapshot drift — got {actual}, expected {expected}"
