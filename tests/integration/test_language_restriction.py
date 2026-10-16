"""Real CTranslate2 path for ``restrict_languages`` (``--language en+ja``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from faster_whisper.audio import decode_audio

from backend.transcribe import restrict_languages

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def test_restricted_detection_never_picks_the_unrestricted_winner(
    tiny_whisper_model: WhisperModel, project_root: Path
) -> None:
    audio = decode_audio(str(project_root / "tests/fixtures/audio/two_speakers_10s.wav"))
    unrestricted, _, _ = tiny_whisper_model.detect_language(audio)
    assert unrestricted == "en"  # the fixture is English; ja/ko below must exclude it

    view = restrict_languages(tiny_whisper_model, ("ja", "ko"))
    segments, info = view.transcribe(audio, multilingual=True)
    list(segments)

    assert info.language in {"ja", "ko"}
    assert tiny_whisper_model.detect_language(audio)[0] == "en"
