"""Pyannote pipeline loader + inference, with HF_TOKEN/license error mapping.

The pipeline is constructed in local scope and goes out of scope on return
so its CUDA allocations can be released alongside `torch.cuda.empty_cache()`
when the caller finishes.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.diarize.pipeline import SpeakerTurn

LOGGER = logging.getLogger(__name__)

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def _read_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _release_cuda() -> None:
    """Drop cached CUDA allocations after a model goes out of scope.

    Local scope alone is not enough — PyTorch caches CUDA blocks until
    `empty_cache()` is called. Without this, the sequential whisper→pyannote
    VRAM bound (design Decision §1.10) does not hold.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_pyannote(
    audio_path: str,
    *,
    num_speakers: int = 2,
) -> list[SpeakerTurn]:
    """Run pyannote speaker-diarization-3.1 on the given audio file.

    Returns a list of SpeakerTurn ordered by start time. Raises
    DiarizationConfigError on HF_TOKEN/license failures (batch-aborting) and
    DiarizationRuntimeError on per-file pyannote crashes.
    """
    token = _read_hf_token()
    if not token:
        raise DiarizationConfigError(
            f"HF_TOKEN is not set. The {PYANNOTE_MODEL} model is HuggingFace-gated; "
            "see docs/diarization_setup.md for one-time token + model-license setup."
        )

    try:
        from huggingface_hub.errors import HfHubHTTPError
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DiarizationConfigError(f"pyannote.audio is not installed: {exc}. Run `uv sync` to install.") from exc

    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=token)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            raise DiarizationConfigError(
                f"HF_TOKEN was rejected (401) fetching {PYANNOTE_MODEL}. "
                "Verify the token at https://huggingface.co/settings/tokens; "
                "see docs/diarization_setup.md."
            ) from exc
        if status == 403:
            raise DiarizationConfigError(
                f"HuggingFace returned 403 for {PYANNOTE_MODEL}. Accept the model "
                f"license at https://huggingface.co/{PYANNOTE_MODEL}; "
                "see docs/diarization_setup.md."
            ) from exc
        raise DiarizationConfigError(f"HuggingFace error loading {PYANNOTE_MODEL}: {exc}") from exc
    except Exception as exc:
        raise DiarizationConfigError(f"Failed to load pyannote pipeline {PYANNOTE_MODEL}: {exc}") from exc

    try:
        diarization: Any = pipeline(audio_path, num_speakers=num_speakers)
    except Exception as exc:
        raise DiarizationRuntimeError(f"pyannote inference failed for {audio_path}: {exc}") from exc
    finally:
        del pipeline
        _release_cuda()

    turns: list[SpeakerTurn] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(speaker)))
    turns.sort(key=lambda t: t.start)
    return turns
