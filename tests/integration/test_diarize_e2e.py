"""End-to-end diarization test on a real two-speaker fixture.

Plan: docs/plans/2026-05-21-diarization-txt-default-design.md §6.

Skips cleanly when ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` is unset or
when the licensed fixture (`tests/fixtures/audio/two_speakers_10s.wav`)
isn't present locally — see `tests/fixtures/audio/README.md` for
sourcing options.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from backend.transcribe import format_segments_as_text, transcribe

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "audio" / "two_speakers_10s.wav"
TXT_LINE_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\.\d{2} --> \d{2}:\d{2}:\d{2}\.\d{2}\] SPEAKER_\d{2}: ",
)


def _hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


@pytest.mark.slow
@pytest.mark.network
class TestDiarizeE2E:
    """Real pyannote run on a short two-speaker clip."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing_prereqs(self) -> None:
        if not _hf_token_present():
            pytest.skip("HF_TOKEN/HUGGINGFACE_HUB_TOKEN unset — see docs/diarization_setup.md")
        if not FIXTURE_PATH.exists():
            pytest.skip(
                f"Fixture {FIXTURE_PATH} absent — see tests/fixtures/audio/README.md for sourcing",
            )

    def test_two_speakers_detected_and_anchored(self, tmp_path: Path) -> None:
        result = transcribe(str(FIXTURE_PATH), diarize=True, num_speakers=2)
        segments = result["segments"]
        assert segments, "expected at least one whisper segment"

        speakers = [s["speaker"] for s in segments if "speaker" in s]
        assert speakers, "expected at least one segment to carry a speaker label"
        assert speakers[0] == "SPEAKER_00", "anchor_speaker_zero must label first chronologically as SPEAKER_00"

        distinct = set(speakers)
        assert len(distinct) >= 2, f"expected 2+ distinct speakers, got {distinct!r}"
        # Contiguous labelling (Decision §1.5): no gaps in SPEAKER_NN sequence.
        max_n = max(int(label.removeprefix("SPEAKER_")) for label in distinct)
        assert distinct == {f"SPEAKER_{i:02d}" for i in range(max_n + 1)}

        txt_path = tmp_path / "out.txt"
        txt_path.write_text(format_segments_as_text(segments), encoding="utf-8")
        lines = [line for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines, "txt output must contain at least one segment line"
        # Every segment line: "[hh:mm:ss.ff --> hh:mm:ss.ff] SPEAKER_NN: <text>"
        for line in lines:
            assert TXT_LINE_RE.match(line), f"line does not match documented format: {line!r}"
