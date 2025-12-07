from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404 - used for fixed system tooling (ffmpeg/nvidia-smi)
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics, estimate_snr_db
from backend.preprocess.steps.downmix_resample import downmix_and_resample
from backend.preprocess.steps.loudness import apply_loudnorm
from backend.preprocess.steps.denoise_light import apply_light_denoise

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessResult:
    output_path: Path
    input_info: AudioInfo | None
    metrics: PreprocessMetrics
    profile: str
    cleanup: Callable[[], None] = field(default=lambda: None)


def _gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is available (via nvidia-smi)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi],
            capture_output=True,
            check=False,
            timeout=5,
        )  # nosec B603 B607 - fixed binary path, no untrusted input
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resolve_profile(
    preferred: str | None, *, gpu_check: Callable[[], bool] = _gpu_available
) -> tuple[str, str | None]:
    """Pick GPU profile by default and fall back to CPU with a user-visible note."""
    normalized = (preferred or "auto").lower()
    if normalized not in {"auto", "gpu", "cpu"}:
        normalized = "auto"

    if normalized == "cpu":
        return "cpu", None

    if gpu_check():
        return "gpu", None

    return "cpu", "GPU preprocessing requested but no GPU detected; falling back to CPU."


def preprocess_audio(
    input_path: str | Path,
    config: PreprocessConfig | None = None,
    *,
    inspector: Callable[[Path], AudioInfo] = inspect_audio,
    downmix_fn: Callable[..., StepMetrics] = downmix_and_resample,
    loudnorm_fn: Callable[..., StepMetrics] = apply_loudnorm,
    denoise_fn: Callable[..., StepMetrics] = apply_light_denoise,
    snr_estimator: Callable[[Path, int, int], float | None] = estimate_snr_db,
    gpu_check: Callable[[], bool] = _gpu_available,
    temp_dir_factory: Callable[..., TemporaryDirectory[str]] = TemporaryDirectory,
) -> PreprocessResult:
    """Run the light audio pre-processing pipeline."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspector(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
            profile="disabled",
        )

    profile, profile_note = _resolve_profile(cfg.profile, gpu_check=gpu_check)
    if profile_note:
        LOGGER.info(profile_note)

    input_channels = input_info.channels if input_info else None
    resolved_channels = cfg.target_channels or input_channels or 1
    LOGGER.info(
        "Audio preprocessing enabled (profile=%s, target=%dkHz, channels=%s -> using %d)",
        profile,
        cfg.target_sample_rate,
        str(cfg.target_channels or "auto"),
        resolved_channels,
    )

    temp_dir: TemporaryDirectory[str] = temp_dir_factory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"

    step_metrics: List[StepMetrics] = []
    overall_start = time.time()
    try:
        metrics = _run_pipeline(
            source=source,
            destination=processed_path,
            cfg=cfg,
            step_metrics=step_metrics,
            downmix_fn=downmix_fn,
            loudnorm_fn=loudnorm_fn,
            denoise_fn=denoise_fn,
            resolved_channels=resolved_channels,
        )
    except PreprocessError:
        temp_dir.cleanup()
        raise
    except Exception as exc:  # pragma: no cover - defensive
        temp_dir.cleanup()
        raise PreprocessError(f"Unexpected preprocessing failure: {exc}") from exc

    metrics.total_duration = time.time() - overall_start
    metrics.snr_before = snr_estimator(source, cfg.target_sample_rate, resolved_channels)
    metrics.snr_after = snr_estimator(processed_path, cfg.target_sample_rate, resolved_channels)
    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    if metrics.snr_delta is not None:
        LOGGER.info(
            "SNR delta: %.2f dB -> %.2f dB (Δ %.2f dB)", metrics.snr_before, metrics.snr_after, metrics.snr_delta
        )
    else:
        LOGGER.info("SNR metrics unavailable; see warnings above for decoding issues.")
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        profile=profile,
        cleanup=temp_dir.cleanup,
    )


def _run_pipeline(
    source: Path,
    destination: Path,
    cfg: PreprocessConfig,
    step_metrics: List[StepMetrics],
    *,
    resolved_channels: int,
    downmix_fn: Callable[..., StepMetrics],
    loudnorm_fn: Callable[..., StepMetrics],
    denoise_fn: Callable[..., StepMetrics],
) -> PreprocessMetrics:
    downmixed_path = destination.parent / "downmixed.wav"
    loudnorm_path = destination.parent / "loudnorm.wav"
    try:
        step_metric = downmix_fn(
            input_path=source,
            output_path=downmixed_path,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=resolved_channels,
        )
        step_metrics.append(step_metric)
    except StepExecutionError as exc:
        raise PreprocessError(str(exc)) from exc

    try:
        step_metric = loudnorm_fn(
            input_path=downmixed_path,
            output_path=loudnorm_path,
            sample_rate=cfg.target_sample_rate,
        )
        step_metrics.append(step_metric)
    except StepExecutionError as exc:
        raise PreprocessError(str(exc)) from exc

    try:
        step_metric = denoise_fn(
            input_path=loudnorm_path,
            output_path=destination,
            sample_rate=cfg.target_sample_rate,
        )
        step_metrics.append(step_metric)
    except StepExecutionError as exc:
        raise PreprocessError(str(exc)) from exc

    return PreprocessMetrics(total_duration=0.0, steps=step_metrics)
