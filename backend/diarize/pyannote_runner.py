"""Pyannote pipeline loader + inference, with HF_TOKEN/license error mapping.

The pipeline is constructed in local scope and goes out of scope on return
so its CUDA allocations can be released alongside `torch.cuda.empty_cache()`
when the caller finishes.

Audio is decoded in-process with PyAV and handed to pyannote via its
documented tensor-input API (`{"waveform": (channel, time), "sample_rate": int}`).
This sidesteps `torchcodec`, whose import-time soft-failure emits a noisy
warning but is never reached on the tensor path. See design doc
`docs/plans/2026-05-21-pyannote-community-1-migration-design.md` §2.
"""

from __future__ import annotations

import gc
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, cast

# pyannote.audio 4.x imports torchcodec at module load wrapped in try/except;
# the failure path emits a UserWarning even though our tensor-input flow never
# touches torchcodec at runtime. Filter once before the lazy pyannote import.
warnings.filterwarnings("ignore", message=r".*torchcodec.*", category=UserWarning)

from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.diarize.pipeline import SpeakerTurn

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


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


def _load_audio_tensor(path: str) -> tuple[torch.Tensor, int]:
    """Decode `path` to a mono float32 `(channel, time)` tensor via PyAV.

    PyAV ships its own FFmpeg shared libraries, so this path is independent of
    any system FFmpeg install — the design's key portability lever. Returns
    `(waveform, sample_rate)` shaped to feed pyannote.audio 4.x's tensor-input
    API directly. Pyannote handles any internal resampling via
    `torchaudio.functional.resample` (zero torchcodec references).
    """
    import av  # type: ignore[import-untyped]
    import numpy as np
    import torch

    av_mod = cast(Any, av)  # PyAV ships no type stubs; quarantine the Any at the boundary.

    chunks: list[Any] = []
    sample_rate = 0
    with av_mod.open(path) as container:
        if not container.streams.audio:
            raise DiarizationRuntimeError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate)
        resampler = av_mod.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())

    if not chunks:
        raise DiarizationRuntimeError(f"decoded zero audio frames from {path}")

    waveform_np = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    waveform: torch.Tensor = torch.from_numpy(waveform_np)  # pyright: ignore[reportPrivateImportUsage, reportUnknownMemberType]
    return waveform, sample_rate


def run_pyannote(
    audio_path: str,
    *,
    num_speakers: int = 2,
) -> list[SpeakerTurn]:
    """Run pyannote speaker-diarization-community-1 on the given audio file.

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
    # Non-HF load failures (OSError, CUDA init, etc.) propagate as-is — they are not
    # config errors. processor.py's per-file try/except handles them as file-level
    # failures, not batch aborts.

    try:
        waveform, sample_rate = _load_audio_tensor(audio_path)
        diarization: Any = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )
    except DiarizationRuntimeError:
        raise
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
