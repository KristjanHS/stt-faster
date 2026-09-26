#!/usr/bin/env python3
"""Pre-fetch the Whisper models, the pinned speaker-diarization model and the RNNoise model.

Downloads CT2 weights for the Estonian and English presets so the first
transcription run on a fresh machine doesn't pay for ~3 GB of HuggingFace
downloads. Pure download via the existing model resolvers - no
``WhisperModel(...)`` construction, so this never probes CUDA on a CPU-only box.

Usage::

    .venv/Scripts/python scripts/prefetch_models.py    # Windows
    .venv/bin/python    scripts/prefetch_models.py    # Linux/WSL
    .venv/bin/python    scripts/prefetch_models.py --diarization-only
    .venv/bin/python    scripts/prefetch_models.py --rnnoise-only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO

from backend.diarize.model import DIARIZATION_REPO, fetch_model
from backend.transcribe import _get_cached_model_path, _get_estonian_model_path

LOGGER = logging.getLogger("stt-faster.prefetch")

ESTONIAN_MODEL = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
ENGLISH_MODEL = "Systran/faster-distil-whisper-large-v3"
RNNOISE_URL = (
    "https://raw.githubusercontent.com/GregorR/rnnoise-models/"
    "3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/somnolent-hogwash-2018-09-01/sh.rnnn"  # pragma: allowlist secret
)
RNNOISE_SHA256 = "70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0"  # pragma: allowlist secret
RNNOISE_DEST = Path("models/sh.rnnn")  # PreprocessConfig.rnnoise_model default


def prefetch_rnnoise(
    dest: Path = RNNOISE_DEST,
    *,
    opener: Callable[[str], IO[bytes]] = urllib.request.urlopen,
    sha256: str = RNNOISE_SHA256,
) -> Path:
    """Download + verify the pinned sh.rnnn; an existing verified file costs nothing."""
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest
    with opener(RNNOISE_URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"RNNoise download failed its sha256 check: {RNNOISE_URL}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def prefetch_diarization() -> None:
    LOGGER.info("Prefetching speaker-diarization model: %s", DIARIZATION_REPO)
    LOGGER.info("  -> %s", fetch_model())


def prefetch_all() -> None:
    LOGGER.info("Prefetching Estonian model (ct2/* only): %s", ESTONIAN_MODEL)
    et_path = _get_estonian_model_path(ESTONIAN_MODEL)
    LOGGER.info("  -> %s", et_path)

    LOGGER.info("Prefetching English model (full snapshot): %s", ENGLISH_MODEL)
    en_path = _get_cached_model_path(ENGLISH_MODEL)
    LOGGER.info("  -> %s", en_path)

    prefetch_diarization()
    LOGGER.info("Prefetching RNNoise model -> %s", prefetch_rnnoise())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--diarization-only", action="store_true", help="fetch only the speaker model")
    only.add_argument("--rnnoise-only", action="store_true", help="fetch only the RNNoise model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.diarization_only:
            prefetch_diarization()
        elif args.rnnoise_only:
            LOGGER.info("RNNoise model -> %s", prefetch_rnnoise())
        else:
            prefetch_all()
    except Exception:
        LOGGER.exception("Model prefetch failed")
        return 1
    LOGGER.info("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
