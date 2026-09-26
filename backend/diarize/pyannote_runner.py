"""Pyannote pipeline loader + inference; the model loads from a local dir, never the network.

The pipeline is constructed in local scope and goes out of scope on return
so its CUDA allocations can be released alongside `torch.cuda.empty_cache()`
when the caller finishes.

Audio is decoded in-process with PyAV and handed to pyannote via its
documented tensor-input API (`{"waveform": (channel, time), "sample_rate": int}`).
This sidesteps `torchcodec`, whose import-time soft-failure emits a noisy
warning but is never reached on the tensor path. See design doc
`docs/plans/archived/2026-05-21-pyannote-community-1-migration-design.md` §2.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# pyannote.audio 4.x imports torchcodec at module load wrapped in try/except;
# the failure path emits a UserWarning even though our tensor-input flow never
# touches torchcodec at runtime. Filter once before the lazy pyannote import.
warnings.filterwarnings("ignore", message=r".*torchcodec.*", category=UserWarning)
# Pyannote's SAP pooling fires `std(): degrees of freedom is <= 0` whenever a
# pooled window has a single frame — cosmetic upstream artefact, no effect on
# the returned diarization. Suppress so the bat-driven console stays readable.
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom.*", category=UserWarning)
# pyannote.audio 4.x sends usage metrics unless opted out; the README privacy policy promises none.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

from backend.diarize.errors import DiarizationConfigError, DiarizationRuntimeError
from backend.diarize.model import resolve_model_dir
from backend.diarize.pipeline import SpeakerTurn

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)

# Mirrors backend/transcribe.py PROGRESS_LOG_INTERVAL_SECONDS — duplicated
# rather than imported to keep diarize free of cross-module coupling.
PROGRESS_LOG_INTERVAL_SECONDS = 60.0
ProgressCallback = Callable[[str, int | None, int | None], None]
"""``(step_name, completed, total)``; ``completed=None`` marks a step's entry."""


class _DiarizeProgressHook:
    """Pyannote hook that emits per-stage progress via LOGGER.

    Pyannote 4.x calls the hook repeatedly within each stage; depending on
    the stage, calls arrive with or without ``(completed, total)`` ints. The
    surface we want is one log per stage entry + throttled progress lines
    for the long ones — so:

    - First call for a new ``step_name`` logs the entry line ("step, elapsed
      N min") regardless of whether quantities are attached. Quantitative
      detail (often ``0/N`` at entry) is identical information to the entry
      line and is suppressed here.
    - Subsequent same-step calls without quantities (``completed=None`` or
      ``total`` falsy) are dropped — they would re-log the same entry line.
    - Subsequent same-step calls with quantities are throttled to one line
      per ``PROGRESS_LOG_INTERVAL_SECONDS`` for cadence parity with the
      transcription progress logger.
    """

    def __init__(
        self,
        audio_duration: float | None,
        *,
        clock: Callable[[], float] = time.time,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._audio_duration = audio_duration
        self._clock = clock
        self._on_progress = on_progress
        self._start_time = 0.0
        self._last_log_time = 0.0
        self._last_step: str | None = None

    def __enter__(self) -> "_DiarizeProgressHook":
        self._start_time = self._clock()
        self._last_log_time = self._start_time
        self._last_step = None
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def _forward(self, step_name: str, completed: int | None, total: int | None) -> None:
        """Progress is advisory: a failing callback must not abort the diarization run."""
        if self._on_progress is None:
            return
        try:
            self._on_progress(step_name, completed, total)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Diarization progress callback failed", exc_info=True)

    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Any | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        now = self._clock()
        elapsed_min = (now - self._start_time) / 60
        if step_name != self._last_step:
            self._forward(step_name, None, None)
            LOGGER.info(
                "⌛ Diarization progress: %s, elapsed %.1f min",
                step_name,
                elapsed_min,
            )
            self._last_step = step_name
            self._last_log_time = now
            return
        if completed is None or not total:
            return
        self._forward(step_name, completed, total)
        if (now - self._last_log_time) < PROGRESS_LOG_INTERVAL_SECONDS:
            return
        percent = min(completed / total * 100, 999.0)
        LOGGER.info(
            "⌛ Diarization progress: %s %d/%d (%.1f%%), elapsed %.1f min",
            step_name,
            completed,
            total,
            percent,
            elapsed_min,
        )
        self._last_log_time = now


def _import_pipeline_class() -> Any:
    """Lazy `pyannote.audio.Pipeline` import (~5s: torch + sklearn + hf_hub)."""
    from pyannote.audio import Pipeline  # type: ignore[import-untyped]

    return Pipeline


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


def _diarize(
    pipeline: Any,
    waveform: Any,
    sample_rate: int,
    num_speakers: int,
    audio_duration: float | None,
    on_progress: ProgressCallback | None,
) -> Any:
    with _DiarizeProgressHook(audio_duration, on_progress=on_progress) as hook:
        return pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers, hook=hook)


def run_pyannote(
    audio_path: str,
    *,
    num_speakers: int = 2,
    audio_duration: float | None = None,
    env: Mapping[str, str] = os.environ,
    import_pipeline: Callable[[], Any] = _import_pipeline_class,
    on_progress: ProgressCallback | None = None,
    resolve_model: Callable[[Mapping[str, str]], Path] = resolve_model_dir,
    cuda_available: Callable[[], bool] | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote speaker-diarization-community-1 on the given audio file.

    Returns a list of SpeakerTurn ordered by start time. Raises
    DiarizationConfigError when the model is not installed and
    DiarizationRuntimeError on per-file pyannote crashes.

    ``audio_duration`` (seconds, optional) is retained for caller-side context
    around the surrounding 🎙️/✅ bookend lines; the per-stage progress lines
    emitted via pyannote's hook protocol do not render it.
    """
    model_dir = resolve_model(env)
    try:
        pipeline_cls = import_pipeline()
    except ImportError as exc:
        raise DiarizationConfigError(
            f"pyannote.audio is not installed: {exc}. Re-sync with `--extra cpu` or `--extra cu130`."
        ) from exc

    pipeline = pipeline_cls.from_pretrained(str(model_dir))  # type: ignore[reportUnknownMemberType]
    # Load failures (OSError, CUDA init, etc.) propagate as-is — they are not
    # config errors. processor.py's per-file try/except handles them as file-level
    # failures, not batch aborts.
    if pipeline is None:
        raise DiarizationConfigError(
            f"Pipeline.from_pretrained returned None for {model_dir}; see docs/diarization_setup.md."
        )

    import torch

    # STT_DEVICE=cpu (the user's pick, or the GUI's retry after a GPU failure) keeps pyannote off the GPU too.
    wants_cpu = env.get("STT_DEVICE", "").strip().lower().startswith("cpu")
    use_gpu = not wants_cpu and (cuda_available or torch.cuda.is_available)()
    try:
        waveform, sample_rate = _load_audio_tensor(audio_path)
        if use_gpu:
            # pyannote's fix_reproducibility() (core/pipeline.py:__call__) flips
            # TF32 off and warns when CUDA + TF32-on. Setting it ourselves first
            # satisfies pyannote's contract proactively so the branch stays silent.
            # These flags are process-global, but pyannote would assign the same
            # False values on its first pipeline call anyway — we just do it
            # earlier. faster-whisper uses CTranslate2, not torch.matmul, so its
            # throughput is unaffected by these torch.backends flags.
            # See https://github.com/pyannote/pyannote-audio/issues/1370
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                pipeline.to(torch.device("cuda"))  # pyright: ignore[reportPrivateImportUsage]
                LOGGER.info("🚀 Diarization pipeline on GPU (CUDA)")
                diarization = _diarize(pipeline, waveform, sample_rate, num_speakers, audio_duration, on_progress)
            except Exception as exc:  # e.g. a cuDNN clash with ctranslate2's copy in this process
                LOGGER.warning("⚠️ GPU diarization failed (%s); retrying on the CPU", exc)
                pipeline.to(torch.device("cpu"))  # pyright: ignore[reportPrivateImportUsage]
                _release_cuda()
                diarization = _diarize(pipeline, waveform, sample_rate, num_speakers, audio_duration, on_progress)
        else:
            LOGGER.info("🐌 Diarization pipeline on CPU (%s)", "STT_DEVICE=cpu" if wants_cpu else "no CUDA available")
            diarization = _diarize(pipeline, waveform, sample_rate, num_speakers, audio_duration, on_progress)
    except DiarizationRuntimeError:
        raise
    except Exception as exc:
        raise DiarizationRuntimeError(f"pyannote inference failed for {audio_path}: {exc}") from exc
    finally:
        del pipeline
        _release_cuda()

    turns: list[SpeakerTurn] = []
    # pyannote community-1 returns DiarizeOutput; .speaker_diarization is the
    # Annotation that earlier model versions returned directly.
    annotation = diarization.speaker_diarization
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(speaker)))
    turns.sort(key=lambda t: t.start)
    return turns
