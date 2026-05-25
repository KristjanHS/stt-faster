"""Unit tests for diarize/num_speakers fields on RunConfig (C3 plumbing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.run_config import RunConfig


class TestDiarizeRoundtrip:
    def test_explicit_values_propagate_through_effective_config_and_dict(self) -> None:
        config = RunConfig(
            input_folder=Path("/tmp"),
            diarize=False,
            num_speakers=3,
        )
        config.validate()

        effective = config.get_effective_config()
        assert effective.diarize is False
        assert effective.num_speakers == 3

        serialized = config.to_dict()
        assert serialized["diarize"] is False
        assert serialized["num_speakers"] == 3

    def test_defaults_are_diarize_on_and_two_speakers(self) -> None:
        config = RunConfig(input_folder=Path("/tmp"))
        config.validate()

        assert config.diarize is True
        assert config.num_speakers == 2

        effective = config.get_effective_config()
        assert effective.diarize is True
        assert effective.num_speakers == 2

        serialized = config.to_dict()
        assert serialized["diarize"] is True
        assert serialized["num_speakers"] == 2


class TestDiarizeValidation:
    def test_num_speakers_zero_raises_value_error(self) -> None:
        config = RunConfig(input_folder=Path("/tmp"), num_speakers=0)
        with pytest.raises(ValueError, match="num_speakers"):
            config.validate()

    def test_num_speakers_one_raises_value_error(self) -> None:
        config = RunConfig(input_folder=Path("/tmp"), num_speakers=1)
        with pytest.raises(ValueError, match="num_speakers"):
            config.validate()
