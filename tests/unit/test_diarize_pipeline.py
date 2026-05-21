"""Unit tests for backend.diarize.pipeline — overlap_assign, anchor, annotate."""

from __future__ import annotations

from typing import Any

from backend.diarize.pipeline import (
    SpeakerTurn,
    anchor_speaker_zero,
    annotate,
    overlap_assign,
)


def _seg(start: float, end: float, text: str = "x") -> dict[str, Any]:
    return {"start": start, "end": end, "text": text}


class TestOverlapAssign:
    def test_clean_alignment_one_turn_per_segment(self) -> None:
        segments = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
        turns = [
            SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
            SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
        ]
        out = overlap_assign(segments, turns)
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[1]["speaker"] == "SPEAKER_01"

    def test_mid_segment_switch_picks_majority(self) -> None:
        # Segment 0..10. Two turns: SPEAKER_00 covers 0..3, SPEAKER_01 covers 3..10.
        # SPEAKER_01 wins (7s vs 3s).
        segments = [_seg(0.0, 10.0)]
        turns = [
            SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
            SpeakerTurn(3.0, 10.0, "SPEAKER_01"),
        ]
        out = overlap_assign(segments, turns)
        assert out[0]["speaker"] == "SPEAKER_01"

    def test_segment_outside_any_turn_has_no_speaker_field(self) -> None:
        segments = [_seg(10.0, 12.0)]
        turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00")]
        out = overlap_assign(segments, turns)
        assert "speaker" not in out[0]

    def test_segment_spanning_three_turns(self) -> None:
        # Segment 0..10. Turns: A 0..2, B 2..7, C 7..10. B wins (5s).
        segments = [_seg(0.0, 10.0)]
        turns = [
            SpeakerTurn(0.0, 2.0, "A"),
            SpeakerTurn(2.0, 7.0, "B"),
            SpeakerTurn(7.0, 10.0, "C"),
        ]
        out = overlap_assign(segments, turns)
        assert out[0]["speaker"] == "B"


class TestAnchorSpeakerZero:
    def test_pyannote_label_speaker_01_becomes_speaker_00(self) -> None:
        # Pyannote sometimes assigns SPEAKER_01 to the first-appearing turn.
        # anchor_speaker_zero renumbers chronologically.
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_03"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        ]
        out = anchor_speaker_zero(segments)
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[1]["speaker"] == "SPEAKER_01"
        assert out[2]["speaker"] == "SPEAKER_00"

    def test_contiguous_labels_for_three_speakers(self) -> None:
        # 3 distinct input labels (non-contiguous) → SPEAKER_00, _01, _02.
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_02"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_05"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_07"},
        ]
        out = anchor_speaker_zero(segments)
        assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def test_segments_without_speaker_pass_through(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0},  # no speaker
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_03"},
        ]
        out = anchor_speaker_zero(segments)
        assert "speaker" not in out[0]
        assert out[1]["speaker"] == "SPEAKER_00"


class TestAnnotate:
    def test_zero_turns_returns_segments_unchanged(self) -> None:
        segments = [_seg(0.0, 1.0), _seg(1.0, 2.0)]
        out = annotate(segments, "fake.wav", num_speakers=2, runner=lambda *_, **__: [])
        assert out == segments
        for seg in out:
            assert "speaker" not in seg

    def test_runner_turns_get_assigned_and_anchored(self) -> None:
        segments = [_seg(0.0, 2.0), _seg(2.0, 4.0)]

        def runner(_path: str, *, num_speakers: int) -> list[SpeakerTurn]:
            assert num_speakers == 2
            # Pyannote returns labels in arbitrary order; SPEAKER_01 covers first turn.
            return [
                SpeakerTurn(0.0, 2.0, "SPEAKER_01"),
                SpeakerTurn(2.0, 4.0, "SPEAKER_00"),
            ]

        out = annotate(segments, "fake.wav", num_speakers=2, runner=runner)
        # After anchor: first-appearing (SPEAKER_01) → SPEAKER_00, second → SPEAKER_01.
        assert out[0]["speaker"] == "SPEAKER_00"
        assert out[1]["speaker"] == "SPEAKER_01"
