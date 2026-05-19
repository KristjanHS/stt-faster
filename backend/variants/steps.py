"""Preprocessing step system: typed configs + dispatch table.

A `Step` is a (`step_type`, optional `StepConfig`) pair. Step types map to
handler functions via `STEP_HANDLERS`. Some step types take a typed
`*StepConfig` dataclass (their fields are real parameters); the rest accept
`None` because all parameters come from the global `PreprocessConfig`.

`StepRegistry.create_step(step_type, config)` returns a `Step` ready to call
`.execute(...)`. The smoke test (`tests/integration/test_every_step_smoke.py`)
and `backend/variants/preprocess_steps._execute_step_with_registry` are the
two callers.
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404 - sox invocation with fixed arguments
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

import ffmpeg  # type: ignore[import-untyped]

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics
from backend.variants import preprocess_steps as _preprocess_steps

LOGGER = logging.getLogger(__name__)


# === Typed configs ============================================================
# Only configs carrying real parameter fields survive C.2. Steps whose
# parameters come entirely from `PreprocessConfig` accept `config=None`.


@dataclass
class DenoiseCustomStepConfig:
    """Configuration for custom denoise step."""

    noise_clip_duration_s: float = 5.0
    n_std_thresh_stationary: float = 0.75
    prop_decrease: float = 0.25


@dataclass
class LoudnormOnlyStepConfig:
    """Configuration for loudnorm only step."""

    loudnorm_preset: str = "default"


@dataclass
class LoudnormHighpassStepConfig:
    """Configuration for loudnorm with highpass step."""

    loudnorm_preset: str = "default"


@dataclass
class AresamplLoudnormFixedStepConfig:
    """Configuration for aresampl loudnorm fixed step."""

    integrated_loudness: float = -24.0
    true_peak: float = -2.0
    loudness_range: float = 15.0


@dataclass
class AresamplLoudnormFixed2StepConfig:
    """Configuration for aresampl loudnorm fixed2 step."""

    integrated_loudness: float = -24.0
    true_peak: float = -2.0
    loudness_range: float = 15.0


@dataclass
class VolumeLimiterStepConfig:
    """Configuration for volume limiter step."""

    volume_db: float = 1.5


@dataclass
class PeakNormalize2passStepConfig:
    """Configuration for peak normalize 2pass step."""

    target_db: float = -6.0
    max_gain_db: float = 6.0


@dataclass
class SoxPeakNormalizeStepConfig:
    """Configuration for sox peak normalize step."""

    target_db: float = -3.0


StepConfig = Union[
    DenoiseCustomStepConfig,
    LoudnormOnlyStepConfig,
    LoudnormHighpassStepConfig,
    AresamplLoudnormFixedStepConfig,
    AresamplLoudnormFixed2StepConfig,
    VolumeLimiterStepConfig,
    PeakNormalize2passStepConfig,
    SoxPeakNormalizeStepConfig,
]


# === Inline FFmpeg/SoX handlers ===============================================
# `resample`, `peak_normalize_2pass`, and `sox_peak_normalize` are implemented
# directly here because they don't have a parallel in `preprocess_steps`. The
# rest of the handlers delegate to `_preprocess_steps`.


def _handle_resample(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    step_name = "resample"
    start = time.time()
    try:
        stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(output_path),
            ac=1,
            ar=global_config.target_sample_rate,
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except ffmpeg.Error as exc:  # type: ignore[misc]
        stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
        raise StepExecutionError(step_name, f"ffmpeg resample failed: {stderr}") from exc
    except Exception as exc:
        raise StepExecutionError(step_name, f"ffmpeg resample error: {exc}") from exc
    duration = time.time() - start
    return StepMetrics(name=step_name, backend="ffmpeg", duration=duration)


def _handle_peak_normalize_2pass(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, PeakNormalize2passStepConfig) else PeakNormalize2passStepConfig()
    return _preprocess_steps.peak_normalize_2pass(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        target_db=cfg.target_db,
        max_gain_db=cfg.max_gain_db,
    )


def _handle_sox_peak_normalize(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    step_name = "sox_peak_normalize"
    cfg = config if isinstance(config, SoxPeakNormalizeStepConfig) else SoxPeakNormalizeStepConfig()
    target_sample_rate = global_config.target_sample_rate

    sox_path = shutil.which("sox")
    if not sox_path:
        raise StepExecutionError(step_name, "sox is required but not found on PATH")

    start = time.time()
    try:
        cmd = [
            sox_path,
            str(input_path),
            "-r",
            str(target_sample_rate),
            "-c",
            "1",
            str(output_path),
            "gain",
            "-n",
            str(cfg.target_db),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec B603 - fixed command list, no shell

        # Apply limiter after SoX to suppress intersample overs.
        stream = ffmpeg.input(str(output_path))  # type: ignore[reportUnknownMemberType]
        temp_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
        stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            stream,  # type: ignore[reportUnknownArgumentType]
            str(temp_output),
            af="alimiter=limit=0.98",
            acodec="pcm_s16le",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType]
        temp_output.replace(output_path)
    except subprocess.CalledProcessError as exc:
        raise StepExecutionError(step_name, f"sox failed: {exc.stderr}") from exc
    except ffmpeg.Error as exc:  # type: ignore[misc]
        stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
        raise StepExecutionError(step_name, f"ffmpeg failed: {stderr}") from exc
    except Exception as exc:
        raise StepExecutionError(step_name, f"Error: {exc}") from exc

    duration = time.time() - start
    return StepMetrics(name=step_name, backend="sox+ffmpeg", duration=duration)


# === Delegating handlers ======================================================
# These wrap the audio-processing functions in `backend.variants.preprocess_steps`
# or `backend.preprocess.steps.*`.


def _handle_denoise(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    from backend.preprocess.steps.denoise_light import apply_light_denoise

    return apply_light_denoise(
        input_path=input_path,
        output_path=output_path,
        sample_rate=global_config.target_sample_rate,
    )


def _handle_denoise_custom(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    from backend.preprocess.steps.denoise_light import apply_light_denoise

    cfg = config if isinstance(config, DenoiseCustomStepConfig) else DenoiseCustomStepConfig()
    return apply_light_denoise(
        input_path=input_path,
        output_path=output_path,
        sample_rate=global_config.target_sample_rate,
        noise_clip_duration_s=cfg.noise_clip_duration_s,
        n_std_thresh_stationary=cfg.n_std_thresh_stationary,
        prop_decrease=cfg.prop_decrease,
    )


def _handle_ffmpeg(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

    return run_ffmpeg_pipeline(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        rnnoise_mix=global_config.rnnoise_mix,
        loudnorm_preset=global_config.loudnorm_preset,
        rnnoise_model=global_config.rnnoise_model,
    )


def _handle_loudnorm_only(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, LoudnormOnlyStepConfig) else LoudnormOnlyStepConfig()
    return _preprocess_steps.loudnorm_only(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        loudnorm_preset=cfg.loudnorm_preset,
    )


def _handle_loudnorm_highpass(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, LoudnormHighpassStepConfig) else LoudnormHighpassStepConfig()
    start = time.time()
    metrics = _preprocess_steps.loudnorm_with_highpass(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        loudnorm_preset=cfg.loudnorm_preset,
    )
    LOGGER.info("Loudnorm highpass step completed in %.2fs", time.time() - start)
    return metrics


def _handle_volume_limiter(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, VolumeLimiterStepConfig) else VolumeLimiterStepConfig()
    return _preprocess_steps.volume_with_limiter(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        volume_db=cfg.volume_db,
    )


def _handle_dynaudnorm(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.dynaudnorm_only(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Dynaudnorm step completed in %.2fs", time.time() - start)
    return metrics


def _handle_highlow_aform_loudnorm(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.highlow_aform_loudnorm(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Highlow aform loudnorm step completed in %.2fs", time.time() - start)
    return metrics


def _handle_highlow_nosampl_loudnorm(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.highlow_nosampl_loudnorm(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Highlow nosampl loudnorm step completed in %.2fs", time.time() - start)
    return metrics


def _handle_aresampl_loudnorm_fixed(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, AresamplLoudnormFixedStepConfig) else AresamplLoudnormFixedStepConfig()
    start = time.time()
    metrics = _preprocess_steps.aresampl_loudnorm_fixed(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        target_i=cfg.integrated_loudness,
        target_tp=cfg.true_peak,
        target_lra=cfg.loudness_range,
    )
    LOGGER.info("Aresampl loudnorm fixed step completed in %.2fs", time.time() - start)
    return metrics


def _handle_aresampl_loudnorm_fixed2(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del step_index
    cfg = config if isinstance(config, AresamplLoudnormFixed2StepConfig) else AresamplLoudnormFixed2StepConfig()
    start = time.time()
    metrics = _preprocess_steps.aresampl_loudnorm_fixed2(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
        target_i=cfg.integrated_loudness,
        target_tp=cfg.true_peak,
        target_lra=cfg.loudness_range,
    )
    LOGGER.info("Aresampl loudnorm fixed2 step completed in %.2fs", time.time() - start)
    return metrics


def _handle_loudnorm_2pass_linear(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.loudnorm_2pass_linear(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Loudnorm 2pass linear step completed in %.2fs", time.time() - start)
    return metrics


def _handle_limiter_only(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.limiter_only(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Limiter only step completed in %.2fs", time.time() - start)
    return metrics


def _handle_compressor_limiter(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.compressor_with_limiter(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Compressor limiter step completed in %.2fs", time.time() - start)
    return metrics


def _handle_dynaudnorm_conservative(
    config: StepConfig | None,
    input_path: Path,
    output_path: Path,
    global_config: PreprocessConfig,
    step_index: int,
) -> StepMetrics:
    del config, step_index
    start = time.time()
    metrics = _preprocess_steps.dynaudnorm_conservative(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=global_config.target_sample_rate,
        target_channels=global_config.target_channels or 1,
    )
    LOGGER.info("Dynaudnorm conservative step completed in %.2fs", time.time() - start)
    return metrics


# === Dispatch table ===========================================================
HandlerFn = Callable[[Any, Path, Path, PreprocessConfig, int], StepMetrics]

STEP_HANDLERS: dict[str, HandlerFn] = {
    "resample": _handle_resample,
    "ffmpeg": _handle_ffmpeg,
    "denoise": _handle_denoise,
    "denoise_custom": _handle_denoise_custom,
    "loudnorm_only": _handle_loudnorm_only,
    "loudnorm_highpass": _handle_loudnorm_highpass,
    "dynaudnorm": _handle_dynaudnorm,
    "highlow_aform_loudnorm": _handle_highlow_aform_loudnorm,
    "highlow_nosampl_loudnorm": _handle_highlow_nosampl_loudnorm,
    "aresampl_loudnorm_fixed": _handle_aresampl_loudnorm_fixed,
    "aresampl_loudnorm_fixed2": _handle_aresampl_loudnorm_fixed2,
    "loudnorm_2pass_linear": _handle_loudnorm_2pass_linear,
    "limiter_only": _handle_limiter_only,
    "volume_limiter": _handle_volume_limiter,
    "peak_normalize_2pass": _handle_peak_normalize_2pass,
    "sox_peak_normalize": _handle_sox_peak_normalize,
    "compressor_limiter": _handle_compressor_limiter,
    "dynaudnorm_conservative": _handle_dynaudnorm_conservative,
}


# === Step + Registry =========================================================


@dataclass(frozen=True, slots=True)
class Step:
    """Bound step ready to execute. Created via `StepRegistry.create_step`."""

    step_type: str
    config: StepConfig | None = None

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        handler = STEP_HANDLERS.get(self.step_type)
        if handler is None:
            raise ValueError(f"Unknown step type: {self.step_type}")
        return handler(self.config, input_path, output_path, global_config, step_index)


class StepRegistry:
    """Lookup surface over `STEP_HANDLERS`. Kept as a class for caller compat."""

    @classmethod
    def get_registered_types(cls) -> list[str]:
        return list(STEP_HANDLERS.keys())

    @classmethod
    def create_step(cls, step_type: str, config: StepConfig | None = None) -> Step:
        if step_type not in STEP_HANDLERS:
            raise ValueError(f"Unknown step type: {step_type}")
        return Step(step_type=step_type, config=config)
