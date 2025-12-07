from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import PreprocessError, StepExecutionError
from backend.preprocess.io import AudioInfo, inspect_audio
from backend.preprocess.metrics import PreprocessMetrics, StepMetrics
from backend.preprocess.steps.downmix_resample import downmix_and_resample

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessResult:
    output_path: Path
    input_info: AudioInfo | None
    metrics: PreprocessMetrics
    cleanup: Callable[[], None] = field(default=lambda: None)


def preprocess_audio(input_path: str | Path, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Run the light audio pre-processing pipeline."""
    cfg = config or PreprocessConfig.from_env()
    source = Path(input_path)
    input_info: AudioInfo | None = None

    try:
        input_info = inspect_audio(source)
    except PreprocessError as exc:
        LOGGER.warning("Skipping metadata inspection: %s", exc)

    if not cfg.enabled:
        LOGGER.info("Audio preprocessing disabled; using input as-is.")
        return PreprocessResult(
            output_path=source,
            input_info=input_info,
            metrics=PreprocessMetrics(total_duration=0.0, steps=[]),
        )

    LOGGER.info(
        "Audio preprocessing enabled (profile=%s, target=%dkHz, channels=%d)",
        cfg.profile,
        cfg.target_sample_rate,
        cfg.target_channels,
    )

    temp_dir = tempfile.TemporaryDirectory(prefix="stt-preprocess_", dir=cfg.temp_dir)
    processed_path = Path(temp_dir.name) / "preprocessed.wav"

    step_metrics: List[StepMetrics] = []
    overall_start = time.time()
    try:
        metrics = _run_pipeline(
            source=source,
            destination=processed_path,
            cfg=cfg,
            step_metrics=step_metrics,
        )
    except PreprocessError:
        temp_dir.cleanup()
        raise
    except Exception as exc:  # pragma: no cover - defensive
        temp_dir.cleanup()
        raise PreprocessError(f"Unexpected preprocessing failure: {exc}") from exc

    metrics.total_duration = time.time() - overall_start
    LOGGER.info("Pre-processing completed in %.2fs", metrics.total_duration)
    for metric in metrics.steps:
        LOGGER.info(" - Step %s (%s): %.2fs", metric.name, metric.backend, metric.duration)

    return PreprocessResult(
        output_path=processed_path,
        input_info=input_info,
        metrics=metrics,
        cleanup=temp_dir.cleanup,
    )


def _run_pipeline(
    source: Path,
    destination: Path,
    cfg: PreprocessConfig,
    step_metrics: List[StepMetrics],
) -> PreprocessMetrics:
    try:
        step_metric = downmix_and_resample(
            input_path=source,
            output_path=destination,
            target_sample_rate=cfg.target_sample_rate,
            target_channels=cfg.target_channels,
        )
        step_metrics.append(step_metric)
    except StepExecutionError as exc:
        raise PreprocessError(str(exc)) from exc

    return PreprocessMetrics(total_duration=0.0, steps=step_metrics)
