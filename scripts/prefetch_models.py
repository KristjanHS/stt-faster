#!/usr/bin/env python3
"""Pre-fetch Whisper models used by the Windows transcribe bats.

Downloads CT2 weights for the Estonian and English presets so the first
transcription run on a fresh machine doesn't pay for ~3 GB of HuggingFace
downloads. Pure download via the existing model resolvers - no
``WhisperModel(...)`` construction, so this never probes CUDA on a CPU-only box.

Usage::

    .venv/Scripts/python scripts/prefetch_models.py    # Windows
    .venv/bin/python    scripts/prefetch_models.py    # Linux/WSL
"""

from __future__ import annotations

import logging
import sys

from backend.transcribe import _get_cached_model_path, _get_estonian_model_path

LOGGER = logging.getLogger("stt-faster.prefetch")

ESTONIAN_MODEL = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
ENGLISH_MODEL = "Systran/faster-distil-whisper-large-v3"


def prefetch_all() -> None:
    LOGGER.info("Prefetching Estonian model (ct2/* only): %s", ESTONIAN_MODEL)
    et_path = _get_estonian_model_path(ESTONIAN_MODEL)
    LOGGER.info("  -> %s", et_path)

    LOGGER.info("Prefetching English model (full snapshot): %s", ENGLISH_MODEL)
    en_path = _get_cached_model_path(ENGLISH_MODEL)
    LOGGER.info("  -> %s", en_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        prefetch_all()
    except Exception:
        LOGGER.exception("Model prefetch failed")
        return 1
    LOGGER.info("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
