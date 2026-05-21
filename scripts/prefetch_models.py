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
import os
import sys

from backend.transcribe import _get_cached_model_path, _get_estonian_model_path

LOGGER = logging.getLogger("stt-faster.prefetch")

ESTONIAN_MODEL = "TalTechNLP/whisper-large-v3-turbo-et-verbatim"
ENGLISH_MODEL = "Systran/faster-distil-whisper-large-v3"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def prefetch_all() -> None:
    LOGGER.info("Prefetching Estonian model (ct2/* only): %s", ESTONIAN_MODEL)
    et_path = _get_estonian_model_path(ESTONIAN_MODEL)
    LOGGER.info("  -> %s", et_path)

    LOGGER.info("Prefetching English model (full snapshot): %s", ENGLISH_MODEL)
    en_path = _get_cached_model_path(ENGLISH_MODEL)
    LOGGER.info("  -> %s", en_path)

    LOGGER.info("Prefetching pyannote diarization model: %s", PYANNOTE_MODEL)
    pyannote_path = prefetch_pyannote(PYANNOTE_MODEL)
    LOGGER.info("  -> %s", pyannote_path)


def prefetch_pyannote(repo_id: str) -> str:
    # Revision pin is required by Bandit B615 and must be a literal SHA in the
    # snapshot_download() call (Bandit only accepts literal 40-char hex strings,
    # not module constants or function parameters). Resolved from HF API 2026-05-21.
    # Cross-ref: backend/diarize/pyannote_runner.py uses Pipeline.from_pretrained
    # without a revision pin — relies on pyannote.audio==3.4.0 lock to resolve the
    # same SHA. Keep this literal in sync if the model card updates a default revision.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    from backend.diarize.errors import DiarizationConfigError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise DiarizationConfigError(
            f"HF_TOKEN is not set. The {repo_id} model is HuggingFace-gated; "
            "see docs/diarization_setup.md for one-time token + model-license setup."
        )
    try:
        return snapshot_download(
            repo_id=repo_id,
            token=token,
            revision="84fd25912480287da0247647c3d2b4853cb3ee5d",  # pragma: allowlist secret
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None) if exc.response is not None else None
        if status == 401:
            raise DiarizationConfigError(
                f"HF_TOKEN was rejected (401) fetching {repo_id}. "
                "Verify the token at https://huggingface.co/settings/tokens; see docs/diarization_setup.md."
            ) from exc
        if status == 403:
            raise DiarizationConfigError(
                f"HuggingFace returned 403 for {repo_id}. Accept the model license at "
                f"https://huggingface.co/{repo_id}; see docs/diarization_setup.md."
            ) from exc
        raise


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
