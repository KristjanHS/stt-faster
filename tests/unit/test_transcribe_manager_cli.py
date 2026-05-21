"""Argparse smoke tests for scripts/transcribe_manager.py (C3 plumbing)."""

from __future__ import annotations

from scripts.transcribe_manager import create_parser


class TestDiarizeFlag:
    def test_default_is_diarize_true(self) -> None:
        args = create_parser().parse_args(["process", "/tmp"])
        assert args.diarize is True

    def test_no_diarize_flips_to_false(self) -> None:
        args = create_parser().parse_args(["process", "/tmp", "--no-diarize"])
        assert args.diarize is False


class TestNumSpeakersFlag:
    def test_default_is_two(self) -> None:
        args = create_parser().parse_args(["process", "/tmp"])
        assert args.num_speakers == 2

    def test_override_is_parsed_as_int(self) -> None:
        args = create_parser().parse_args(["process", "/tmp", "--num-speakers", "5"])
        assert args.num_speakers == 5
