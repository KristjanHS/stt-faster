"""New class-based preprocessing step system with typed configurations."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 - sox invocation with fixed arguments
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import ffmpeg  # type: ignore[import-untyped]

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)


class StepExecutor(Protocol):
    """Protocol for step execution with standardized metrics and error handling."""

    def resample(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_name: str,
    ) -> StepMetrics:
        """Execute resampling step."""
        ...

    def loudnorm_only(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        loudnorm_preset: str,
        step_name: str,
    ) -> StepMetrics:
        """Execute loudness normalization only."""
        ...

    def volume_with_limiter(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        volume_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute volume adjustment with limiter."""
        ...

    def peak_normalize_2pass(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        max_gain_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute two-pass peak normalization."""
        ...

    def sox_peak_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute SoX peak normalization."""
        ...

    def apply_light_denoise(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        step_name: str,
        *,
        noise_clip_duration_s: float | None = None,
        n_std_thresh_stationary: float | None = None,
        prop_decrease: float | None = None,
    ) -> StepMetrics:
        """Execute light denoising."""
        ...

    def run_ffmpeg_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        rnnoise_mix: float,
        loudnorm_preset: str,
        rnnoise_model: str | None,
        step_name: str,
    ) -> StepMetrics:
        """Execute FFmpeg pipeline with RNNoise and loudness normalization."""
        ...


class FFmpegExecutor:
    """Executor for FFmpeg-based audio processing steps."""

    def resample(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_name: str,
    ) -> StepMetrics:
        """Execute resampling step."""
        start = time.time()

        try:
            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(output_path),
                ac=1,  # Force mono for Whisper/faster-whisper
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

    def loudnorm_only(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        loudnorm_preset: str,
        step_name: str,
    ) -> StepMetrics:
        """Execute loudness normalization only."""
        from backend.preprocess.config import PreprocessConfig

        start = time.time()

        try:
            target_i, target_tp, target_lra = PreprocessConfig.resolve_loudnorm_params(
                None, None, None, loudnorm_preset
            )

            # Build lightweight filter graph: only resample + loudnorm
            filter_graph = (
                f"aresample=resampler=soxr:osr={target_sample_rate},"
                f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
            )

            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(output_path),
                ac=1,  # Force mono for Whisper/faster-whisper
                af=filter_graph,
                ar=target_sample_rate,
                sample_fmt="s16",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType]

        except ffmpeg.Error as exc:  # type: ignore[misc]
            stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
            raise StepExecutionError(step_name, f"ffmpeg loudnorm failed: {stderr}") from exc
        except Exception as exc:
            raise StepExecutionError(step_name, f"ffmpeg loudnorm error: {exc}") from exc

        duration = time.time() - start
        return StepMetrics(name=step_name, backend="ffmpeg", duration=duration)

    def volume_with_limiter(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        volume_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute volume adjustment with limiter."""
        start = time.time()

        try:
            filter_graph = f"volume={volume_db}dB,alimiter=limit=0.98"

            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(output_path),
                ac=1,  # Force mono
                af=filter_graph,
                ar=target_sample_rate,
                acodec="pcm_s16le",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType]

        except ffmpeg.Error as exc:  # type: ignore[misc]
            stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
            raise StepExecutionError(step_name, f"ffmpeg volume limiter failed: {stderr}") from exc
        except Exception as exc:
            raise StepExecutionError(step_name, f"ffmpeg volume limiter error: {exc}") from exc

        duration = time.time() - start
        return StepMetrics(name=step_name, backend="ffmpeg", duration=duration)

    def peak_normalize_2pass(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        max_gain_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute two-pass peak normalization."""
        import re

        start = time.time()

        try:
            # Pass 1: volumedetect
            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                "null",
                af="volumedetect",
                f="null",
            )
            _, stderr = ffmpeg.run(  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                overwrite_output=True,
                quiet=True,
                capture_stdout=True,
                capture_stderr=True,
            )

            # Parse max_volume from stderr
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr  # type: ignore[reportUnnecessaryIsInstance]
            match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr_str)
            if not match:
                raise StepExecutionError(step_name, "Could not find max_volume in volumedetect output")

            max_volume_db = float(match.group(1))

            # Compute gain: clamp((-target_db - max_volume_db), 0.0, max_gain_db)
            gain_db = max(0.0, min(-target_db - max_volume_db, max_gain_db))

            LOGGER.info(
                "Peak normalize: max_volume=%.2f dBFS, target=%.2f dBFS, applying gain=%.2f dB",
                max_volume_db,
                target_db,
                gain_db,
            )

            # Pass 2: Apply gain + limiter
            if gain_db > 0.0:
                filter_graph = f"volume={gain_db}dB,alimiter=limit=0.98"
            else:
                # No gain needed, just apply limiter
                filter_graph = "alimiter=limit=0.98"

            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(output_path),
                ac=1,  # Force mono
                af=filter_graph,
                ar=target_sample_rate,
                acodec="pcm_s16le",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType]

        except ffmpeg.Error as exc:  # type: ignore[misc]
            stderr = exc.stderr.decode() if exc.stderr else "unknown error"  # type: ignore[union-attr]
            raise StepExecutionError(step_name, f"ffmpeg peak normalize failed: {stderr}") from exc
        except Exception as exc:
            raise StepExecutionError(step_name, f"ffmpeg peak normalize error: {exc}") from exc

        duration = time.time() - start
        return StepMetrics(
            name=step_name,
            backend="ffmpeg",
            duration=duration,
            metadata={"target_db": target_db, "max_gain_db": max_gain_db, "applied_gain_db": gain_db},
        )

    def apply_light_denoise(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        step_name: str,
    ) -> StepMetrics:
        """Execute light denoising."""
        raise NotImplementedError("Light denoising should use PythonExecutor")

    def run_ffmpeg_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        rnnoise_mix: float,
        loudnorm_preset: str,
        rnnoise_model: str | None,
        step_name: str,
    ) -> StepMetrics:
        """Execute FFmpeg pipeline with RNNoise and loudness normalization."""
        raise NotImplementedError("FFmpeg pipeline should use PythonExecutor")

    def sox_peak_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute SoX peak normalization."""
        raise NotImplementedError("SoX operations should use SoxExecutor")


class SoxExecutor:
    """Executor for SoX-based audio processing steps."""

    def sox_peak_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute SoX peak normalization."""
        import shutil

        start = time.time()

        # Check if sox is available
        sox_path = shutil.which("sox")
        if not sox_path:
            raise StepExecutionError(step_name, "sox is required but not found on PATH")

        try:
            # sox in.wav out.wav gain -n -3
            # -n means normalize to the given level
            cmd = [
                sox_path,
                str(input_path),
                "-r",
                str(target_sample_rate),
                "-c",
                "1",  # Force mono
                str(output_path),
                "gain",
                "-n",
                str(target_db),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec B603 - fixed command list, no shell

            # Apply limiter after SoX (as per user note about intersample overs)
            import ffmpeg  # type: ignore[import-untyped]

            stream = ffmpeg.input(str(output_path))  # type: ignore[reportUnknownMemberType]
            temp_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
            stream = ffmpeg.output(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(temp_output),
                af="alimiter=limit=0.98",
                acodec="pcm_s16le",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType]

            # Replace original file
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

    # Implement other methods with NotImplementedError
    def resample(
        self, input_path: Path, output_path: Path, global_config: PreprocessConfig, step_name: str
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Resample should use FFmpegExecutor")

    def loudnorm_only(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        loudnorm_preset: str,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Loudnorm should use FFmpegExecutor")

    def volume_with_limiter(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        volume_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Volume limiter should use FFmpegExecutor")

    def peak_normalize_2pass(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        max_gain_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Peak normalize should use FFmpegExecutor")

    def apply_light_denoise(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        step_name: str,
        *,
        noise_clip_duration_s: float | None = None,
        n_std_thresh_stationary: float | None = None,
        prop_decrease: float | None = None,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Denoise should use PythonExecutor")

    def run_ffmpeg_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        rnnoise_mix: float,
        loudnorm_preset: str,
        rnnoise_model: str | None,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("FFmpeg pipeline should use PythonExecutor")


class PreprocessStepsExecutor:
    """Executor for complex preprocessing functions that need special handling."""

    def loudnorm_only(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        loudnorm_preset: str,
        step_name: str,
    ) -> StepMetrics:
        """Execute loudness normalization only."""
        from backend.variants import preprocess_steps

        return preprocess_steps.loudnorm_only(  # type: ignore[reportUnknownMemberType]
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            loudnorm_preset=loudnorm_preset,
        )

    def volume_with_limiter(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        volume_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute volume adjustment with limiter."""
        from backend.variants import preprocess_steps

        return preprocess_steps.volume_with_limiter(  # type: ignore[reportUnknownMemberType]
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            volume_db=volume_db,
        )

    def peak_normalize_2pass(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        max_gain_db: float,
        step_name: str,
    ) -> StepMetrics:
        """Execute two-pass peak normalization."""
        from backend.variants import preprocess_steps

        return preprocess_steps.peak_normalize_2pass(  # type: ignore[reportUnknownMemberType]
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            target_db=target_db,
            max_gain_db=max_gain_db,
        )

    # Implement other methods with NotImplementedError
    def resample(
        self, input_path: Path, output_path: Path, global_config: PreprocessConfig, step_name: str
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Resample should use FFmpegExecutor")

    def apply_light_denoise(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        step_name: str,
        *,
        noise_clip_duration_s: float | None = None,
        n_std_thresh_stationary: float | None = None,
        prop_decrease: float | None = None,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Denoise should use PythonExecutor")

    def run_ffmpeg_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        rnnoise_mix: float,
        loudnorm_preset: str,
        rnnoise_model: str | None,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("FFmpeg pipeline should use PythonExecutor")

    def sox_peak_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("SoX operations should use SoxExecutor")


class PythonExecutor:
    """Executor for Python-based audio processing steps."""

    def apply_light_denoise(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        step_name: str,
        *,
        noise_clip_duration_s: float | None = None,
        n_std_thresh_stationary: float | None = None,
        prop_decrease: float | None = None,
    ) -> StepMetrics:
        """Execute light denoising with optional custom parameters."""
        from backend.preprocess.steps.denoise_light import apply_light_denoise

        kwargs: dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
            "sample_rate": sample_rate,
        }

        if noise_clip_duration_s is not None:
            kwargs["noise_clip_duration_s"] = noise_clip_duration_s
        if n_std_thresh_stationary is not None:
            kwargs["n_std_thresh_stationary"] = n_std_thresh_stationary
        if prop_decrease is not None:
            kwargs["prop_decrease"] = prop_decrease

        return apply_light_denoise(**kwargs)

    def run_ffmpeg_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        rnnoise_mix: float,
        loudnorm_preset: str,
        rnnoise_model: str | None,
        step_name: str,
    ) -> StepMetrics:
        """Execute FFmpeg pipeline with RNNoise and loudness normalization."""
        from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

        return run_ffmpeg_pipeline(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            rnnoise_mix=rnnoise_mix,
            loudnorm_preset=loudnorm_preset,
            rnnoise_model=rnnoise_model,
        )

    # Implement other methods with NotImplementedError
    def resample(
        self, input_path: Path, output_path: Path, global_config: PreprocessConfig, step_name: str
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Resample should use FFmpegExecutor")

    def loudnorm_only(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        loudnorm_preset: str,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Loudnorm should use FFmpegExecutor")

    def volume_with_limiter(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        volume_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Volume limiter should use FFmpegExecutor")

    def peak_normalize_2pass(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        max_gain_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("Peak normalize should use FFmpegExecutor")

    def sox_peak_normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        target_channels: int,
        target_db: float,
        step_name: str,
    ) -> StepMetrics:  # type: ignore[override]
        raise NotImplementedError("SoX operations should use SoxExecutor")


class StepConfig(ABC):
    """Abstract base class for step configuration classes."""

    pass


# Type variable for step configuration types
StepConfigT = TypeVar("StepConfigT", bound=StepConfig)


@dataclass
class BaseStepConfig(StepConfig):
    """Base configuration class for all steps."""

    pass


@dataclass
class ResampleStepConfig(BaseStepConfig):
    """Configuration for resample step."""

    pass


@dataclass
class FFmpegStepConfig(BaseStepConfig):
    """Configuration for ffmpeg pipeline step."""

    pass


@dataclass
class DenoiseStepConfig(BaseStepConfig):
    """Configuration for denoise step."""

    pass


@dataclass
class DenoiseCustomStepConfig(BaseStepConfig):
    """Configuration for custom denoise step."""

    noise_clip_duration_s: float = 5.0
    n_std_thresh_stationary: float = 0.75
    prop_decrease: float = 0.25


@dataclass
class LoudnormOnlyStepConfig(BaseStepConfig):
    """Configuration for loudnorm only step."""

    loudnorm_preset: str = "default"


@dataclass
class LoudnormHighpassStepConfig(BaseStepConfig):
    """Configuration for loudnorm with highpass step."""

    loudnorm_preset: str = "default"


@dataclass
class DynaudnormStepConfig(BaseStepConfig):
    """Configuration for dynaudnorm step."""

    pass


@dataclass
class HighlowAformLoudnormStepConfig(BaseStepConfig):
    """Configuration for highlow aform loudnorm step."""

    pass


@dataclass
class HighlowNosamplLoudnormStepConfig(BaseStepConfig):
    """Configuration for highlow nosampl loudnorm step."""

    pass


@dataclass
class AresamplLoudnormFixedStepConfig(BaseStepConfig):
    """Configuration for aresampl loudnorm fixed step."""

    integrated_loudness: float = -24.0
    true_peak: float = -2.0
    loudness_range: float = 15.0


@dataclass
class AresamplLoudnormFixed2StepConfig(BaseStepConfig):
    """Configuration for aresampl loudnorm fixed2 step."""

    integrated_loudness: float = -24.0
    true_peak: float = -2.0
    loudness_range: float = 15.0


@dataclass
class Loudnorm2passLinearStepConfig(BaseStepConfig):
    """Configuration for loudnorm 2pass linear step."""

    pass


@dataclass
class LimiterOnlyStepConfig(BaseStepConfig):
    """Configuration for limiter only step."""

    pass


@dataclass
class VolumeLimiterStepConfig(BaseStepConfig):
    """Configuration for volume limiter step."""

    volume_db: float


@dataclass
class PeakNormalize2passStepConfig(BaseStepConfig):
    """Configuration for peak normalize 2pass step."""

    target_db: float = -6.0
    max_gain_db: float = 6.0


@dataclass
class SoxPeakNormalizeStepConfig(BaseStepConfig):
    """Configuration for sox peak normalize step."""

    target_db: float = -3.0


@dataclass
class CompressorLimiterStepConfig(BaseStepConfig):
    """Configuration for compressor limiter step."""

    pass


@dataclass
class DynaudnormConservativeStepConfig(BaseStepConfig):
    """Configuration for dynaudnorm conservative step."""

    pass


class Step(ABC, Generic[StepConfigT]):
    """Abstract base class for preprocessing steps with standardized execution."""

    config: StepConfigT  # Type annotation for type checker

    def __init__(self, config: StepConfigT | None = None, executor: StepExecutor | None = None):
        """Initialize step with configuration and executor.

        Args:
            config: Step-specific configuration. If None, uses default config.
            executor: StepExecutor to use. If None, uses default executor.
        """
        self.config = config or self.get_default_config()
        # Validate that config is the correct type at construction time
        expected_type = type(self.get_default_config())
        if not isinstance(self.config, expected_type):
            raise TypeError(f"Config must be {expected_type.__name__}, got {type(self.config).__name__}")

        self.executor = executor or self.get_default_executor()

    @classmethod
    @abstractmethod
    def get_step_type(cls) -> str:
        """Return the step type string identifier."""
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> StepConfigT:
        """Return default configuration for this step type."""
        pass

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Return default executor for this step type."""
        return FFmpegExecutor()  # type: ignore[return-value]  # Default to FFmpeg executor

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute the preprocessing step using the configured executor.

        Args:
            input_path: Path to input audio file
            output_path: Path where to write output audio file
            global_config: Global preprocessing configuration
            step_index: Index of this step in the pipeline

        Returns:
            StepMetrics with execution results
        """
        step_name = self.get_step_name()

        # Dispatch to the appropriate executor method based on step type
        if self.get_step_type() == "resample":
            return self.executor.resample(
                input_path=input_path,
                output_path=output_path,
                global_config=global_config,
                step_name=step_name,
            )
        elif self.get_step_type() == "loudnorm_only":
            if not isinstance(self.config, LoudnormOnlyStepConfig):
                raise TypeError(f"Expected LoudnormOnlyStepConfig, got {type(self.config).__name__}")
            return self.executor.loudnorm_only(
                input_path=input_path,
                output_path=output_path,
                target_sample_rate=global_config.target_sample_rate,
                target_channels=global_config.target_channels or 1,
                loudnorm_preset=self.config.loudnorm_preset,
                step_name=step_name,
            )
        elif self.get_step_type() == "volume_limiter":
            if not isinstance(self.config, VolumeLimiterStepConfig):
                raise TypeError(f"Expected VolumeLimiterStepConfig, got {type(self.config).__name__}")
            return self.executor.volume_with_limiter(
                input_path=input_path,
                output_path=output_path,
                target_sample_rate=global_config.target_sample_rate,
                target_channels=global_config.target_channels or 1,
                volume_db=self.config.volume_db,
                step_name=step_name,
            )
        elif self.get_step_type() == "peak_normalize_2pass":
            if not isinstance(self.config, PeakNormalize2passStepConfig):
                raise TypeError(f"Expected PeakNormalize2passStepConfig, got {type(self.config).__name__}")
            return self.executor.peak_normalize_2pass(
                input_path=input_path,
                output_path=output_path,
                target_sample_rate=global_config.target_sample_rate,
                target_channels=global_config.target_channels or 1,
                target_db=self.config.target_db,
                max_gain_db=self.config.max_gain_db,
                step_name=step_name,
            )
        elif self.get_step_type() == "sox_peak_normalize":
            if not isinstance(self.config, SoxPeakNormalizeStepConfig):
                raise TypeError(f"Expected SoxPeakNormalizeStepConfig, got {type(self.config).__name__}")
            return self.executor.sox_peak_normalize(
                input_path=input_path,
                output_path=output_path,
                target_sample_rate=global_config.target_sample_rate,
                target_channels=global_config.target_channels or 1,
                target_db=self.config.target_db,
                step_name=step_name,
            )
        elif self.get_step_type() == "denoise":
            return self.executor.apply_light_denoise(
                input_path=input_path,
                output_path=output_path,
                sample_rate=global_config.target_sample_rate,
                step_name=step_name,
            )
        elif self.get_step_type() == "denoise_custom":
            if not isinstance(self.config, DenoiseCustomStepConfig):
                raise TypeError(f"Expected DenoiseCustomStepConfig, got {type(self.config).__name__}")
            return self.executor.apply_light_denoise(  # type: ignore[reportCallIssue]
                input_path=input_path,
                output_path=output_path,
                sample_rate=global_config.target_sample_rate,
                step_name=step_name,
                noise_clip_duration_s=self.config.noise_clip_duration_s,
                n_std_thresh_stationary=self.config.n_std_thresh_stationary,
                prop_decrease=self.config.prop_decrease,
            )
        elif self.get_step_type() == "ffmpeg":
            # For complex FFmpeg pipeline, we need more config
            if not isinstance(self.config, FFmpegStepConfig):
                raise TypeError(f"Expected FFmpegStepConfig, got {type(self.config).__name__}")
            return self.executor.run_ffmpeg_pipeline(
                input_path=input_path,
                output_path=output_path,
                target_sample_rate=global_config.target_sample_rate,
                target_channels=global_config.target_channels or 1,
                rnnoise_mix=global_config.rnnoise_mix,
                loudnorm_preset=global_config.loudnorm_preset,
                rnnoise_model=global_config.rnnoise_model,
                step_name=step_name,
            )
        else:
            raise NotImplementedError(f"Step type '{self.get_step_type()}' not implemented in executor pattern")

    def get_step_name(self) -> str:
        """Return the step name for metrics reporting."""
        return self.get_step_type()


class ResampleStep(Step[ResampleStepConfig]):
    """Simple resampling step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "resample"

    @classmethod
    def get_default_config(cls) -> ResampleStepConfig:
        return ResampleStepConfig()


class FFmpegStep(Step[FFmpegStepConfig]):
    """FFmpeg pipeline step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "ffmpeg"

    @classmethod
    def get_default_config(cls) -> FFmpegStepConfig:
        return FFmpegStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PythonExecutor for complex FFmpeg pipeline."""
        return PythonExecutor()


class DenoiseStep(Step[DenoiseStepConfig]):
    """Light denoising step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "denoise"

    @classmethod
    def get_default_config(cls) -> DenoiseStepConfig:
        return DenoiseStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PythonExecutor for denoising."""
        return PythonExecutor()


class DenoiseCustomStep(Step[DenoiseCustomStepConfig]):
    """Custom denoising step with configurable parameters."""

    @classmethod
    def get_step_type(cls) -> str:
        return "denoise_custom"

    @classmethod
    def get_default_config(cls) -> DenoiseCustomStepConfig:
        return DenoiseCustomStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PythonExecutor for custom denoising."""
        return PythonExecutor()


class LoudnormOnlyStep(Step[LoudnormOnlyStepConfig]):
    """Loudness normalization only step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "loudnorm_only"

    @classmethod
    def get_default_config(cls) -> LoudnormOnlyStepConfig:
        return LoudnormOnlyStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PreprocessStepsExecutor for loudnorm."""
        return PreprocessStepsExecutor()


class VolumeLimiterStep(Step[VolumeLimiterStepConfig]):
    """Volume adjustment with limiter step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "volume_limiter"

    @classmethod
    def get_default_config(cls) -> VolumeLimiterStepConfig:
        return VolumeLimiterStepConfig(volume_db=1.5)  # Default value needed

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PreprocessStepsExecutor for volume limiter."""
        return PreprocessStepsExecutor()


class PeakNormalize2passStep(Step[PeakNormalize2passStepConfig]):
    """Two-pass peak normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "peak_normalize_2pass"

    @classmethod
    def get_default_config(cls) -> PeakNormalize2passStepConfig:
        return PeakNormalize2passStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use PreprocessStepsExecutor for complex 2-pass peak normalization."""
        return PreprocessStepsExecutor()


class StepRegistry:
    """Registry for preprocessing step classes.

    Uses "typed-at-boundary" pattern: registry is internally untyped (Any),
    type safety is regained at the API boundary. This is necessary because
    Python's type system cannot express "mapping from string keys to a family
    of generic types" with full inference.
    """

    _registry: dict[str, type[Step[Any]]] = {}

    @classmethod
    def register(cls, step_class: type[Step[Any]]) -> None:
        """Register a step class by its step type.

        Accepts any Step[ConfigT] type, which is compatible with Step[Any]
        due to generic variance.
        """
        step_type = step_class.get_step_type()
        # pyright: ignore[reportArgumentType]
        cls._registry[step_type] = step_class  # type: ignore[assignment]
        LOGGER.debug("Registered step class: %s -> %s", step_type, step_class.__name__)

    @classmethod
    def get_step_class(cls, step_type: str) -> type[Step[Any]]:
        """Get step class for a given step type."""
        if step_type not in cls._registry:
            raise ValueError(f"Unknown step type: {step_type}")
        return cls._registry[step_type]

    @classmethod
    def create_step(cls, step_type: str, config: StepConfig | None = None) -> Step[Any]:
        """Create a step instance for the given type and config.

        Type safety is regained at this boundary. The registry is dynamic,
        so we use type: ignore for the unavoidable cast.
        """
        step_class = cls.get_step_class(step_type)
        # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportCallIssue]
        return step_class(config)  # type: ignore[arg-type, call-overload]

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of all registered step types."""
        return list(cls._registry.keys())


# Register all step classes
StepRegistry.register(ResampleStep)
StepRegistry.register(FFmpegStep)
StepRegistry.register(DenoiseStep)
StepRegistry.register(DenoiseCustomStep)
StepRegistry.register(LoudnormOnlyStep)
StepRegistry.register(VolumeLimiterStep)
StepRegistry.register(PeakNormalize2passStep)


class LoudnormHighpassStep(Step[LoudnormHighpassStepConfig]):
    """Loudness normalization with highpass step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "loudnorm_highpass"

    @classmethod
    def get_default_config(cls) -> LoudnormHighpassStepConfig:
        return LoudnormHighpassStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute loudnorm with highpass step."""
        from backend.variants.preprocess_steps import loudnorm_with_highpass

        start = time.time()
        metrics = loudnorm_with_highpass(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            loudnorm_preset=self.config.loudnorm_preset,
        )
        LOGGER.info("Loudnorm highpass step completed in %.2fs", time.time() - start)
        return metrics


class DynaudnormStep(Step[DynaudnormStepConfig]):
    """Dynamic audio normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "dynaudnorm"

    @classmethod
    def get_default_config(cls) -> DynaudnormStepConfig:
        return DynaudnormStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute dynaudnorm step."""
        from backend.variants.preprocess_steps import dynaudnorm_only

        start = time.time()
        metrics = dynaudnorm_only(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Dynaudnorm step completed in %.2fs", time.time() - start)
        return metrics


class HighlowAformLoudnormStep(Step[HighlowAformLoudnormStepConfig]):
    """High-low audio format loudness normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "highlow_aform_loudnorm"

    @classmethod
    def get_default_config(cls) -> HighlowAformLoudnormStepConfig:
        return HighlowAformLoudnormStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute highlow aform loudnorm step."""
        from backend.variants.preprocess_steps import highlow_aform_loudnorm

        start = time.time()
        metrics = highlow_aform_loudnorm(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Highlow aform loudnorm step completed in %.2fs", time.time() - start)
        return metrics


class HighlowNosamplLoudnormStep(Step[HighlowNosamplLoudnormStepConfig]):
    """High-low no sample loudness normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "highlow_nosampl_loudnorm"

    @classmethod
    def get_default_config(cls) -> HighlowNosamplLoudnormStepConfig:
        return HighlowNosamplLoudnormStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute highlow nosampl loudnorm step."""
        from backend.variants.preprocess_steps import highlow_nosampl_loudnorm

        start = time.time()
        metrics = highlow_nosampl_loudnorm(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Highlow nosampl loudnorm step completed in %.2fs", time.time() - start)
        return metrics


class AresamplLoudnormFixedStep(Step[AresamplLoudnormFixedStepConfig]):
    """Aresample loudness normalization with fixed parameters step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "aresampl_loudnorm_fixed"

    @classmethod
    def get_default_config(cls) -> AresamplLoudnormFixedStepConfig:
        return AresamplLoudnormFixedStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute aresampl loudnorm fixed step."""
        from backend.variants.preprocess_steps import aresampl_loudnorm_fixed

        start = time.time()
        metrics = aresampl_loudnorm_fixed(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            target_i=self.config.integrated_loudness,
            target_tp=self.config.true_peak,
            target_lra=self.config.loudness_range,
        )
        LOGGER.info("Aresampl loudnorm fixed step completed in %.2fs", time.time() - start)
        return metrics


class AresamplLoudnormFixed2Step(Step[AresamplLoudnormFixed2StepConfig]):
    """Aresample loudness normalization with fixed parameters v2 step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "aresampl_loudnorm_fixed2"

    @classmethod
    def get_default_config(cls) -> AresamplLoudnormFixed2StepConfig:
        return AresamplLoudnormFixed2StepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute aresampl loudnorm fixed2 step."""
        from backend.variants.preprocess_steps import aresampl_loudnorm_fixed2

        start = time.time()
        metrics = aresampl_loudnorm_fixed2(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            target_i=self.config.integrated_loudness,
            target_tp=self.config.true_peak,
            target_lra=self.config.loudness_range,
        )
        LOGGER.info("Aresampl loudnorm fixed2 step completed in %.2fs", time.time() - start)
        return metrics


class Loudnorm2passLinearStep(Step[Loudnorm2passLinearStepConfig]):
    """Two-pass linear loudness normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "loudnorm_2pass_linear"

    @classmethod
    def get_default_config(cls) -> Loudnorm2passLinearStepConfig:
        return Loudnorm2passLinearStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute loudnorm 2pass linear step."""
        from backend.variants.preprocess_steps import loudnorm_2pass_linear

        start = time.time()
        metrics = loudnorm_2pass_linear(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Loudnorm 2pass linear step completed in %.2fs", time.time() - start)
        return metrics


class LimiterOnlyStep(Step[LimiterOnlyStepConfig]):
    """Audio limiter only step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "limiter_only"

    @classmethod
    def get_default_config(cls) -> LimiterOnlyStepConfig:
        return LimiterOnlyStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute limiter only step."""
        from backend.variants.preprocess_steps import limiter_only

        start = time.time()
        metrics = limiter_only(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Limiter only step completed in %.2fs", time.time() - start)
        return metrics


class SoxPeakNormalizeStep(Step[SoxPeakNormalizeStepConfig]):
    """SoX peak normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "sox_peak_normalize"

    @classmethod
    def get_default_config(cls) -> SoxPeakNormalizeStepConfig:
        return SoxPeakNormalizeStepConfig()

    @classmethod
    def get_default_executor(cls) -> StepExecutor:
        """Use SoxExecutor for SoX-based processing."""
        return SoxExecutor()


class CompressorLimiterStep(Step[CompressorLimiterStepConfig]):
    """Audio compression with limiter step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "compressor_limiter"

    @classmethod
    def get_default_config(cls) -> CompressorLimiterStepConfig:
        return CompressorLimiterStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute compressor limiter step."""
        from backend.variants.preprocess_steps import compressor_with_limiter

        start = time.time()
        metrics = compressor_with_limiter(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Compressor limiter step completed in %.2fs", time.time() - start)
        return metrics


class DynaudnormConservativeStep(Step[DynaudnormConservativeStepConfig]):
    """Conservative dynamic audio normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "dynaudnorm_conservative"

    @classmethod
    def get_default_config(cls) -> DynaudnormConservativeStepConfig:
        return DynaudnormConservativeStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute dynaudnorm conservative step."""
        from backend.variants.preprocess_steps import dynaudnorm_conservative

        start = time.time()
        metrics = dynaudnorm_conservative(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
        )
        LOGGER.info("Dynaudnorm conservative step completed in %.2fs", time.time() - start)
        return metrics


# Register all step classes
StepRegistry.register(ResampleStep)
StepRegistry.register(FFmpegStep)
StepRegistry.register(DenoiseStep)
StepRegistry.register(DenoiseCustomStep)
StepRegistry.register(LoudnormOnlyStep)
StepRegistry.register(LoudnormHighpassStep)
StepRegistry.register(DynaudnormStep)
StepRegistry.register(HighlowAformLoudnormStep)
StepRegistry.register(HighlowNosamplLoudnormStep)
StepRegistry.register(AresamplLoudnormFixedStep)
StepRegistry.register(AresamplLoudnormFixed2Step)
StepRegistry.register(Loudnorm2passLinearStep)
StepRegistry.register(LimiterOnlyStep)
StepRegistry.register(VolumeLimiterStep)
StepRegistry.register(PeakNormalize2passStep)
StepRegistry.register(SoxPeakNormalizeStep)
StepRegistry.register(CompressorLimiterStep)
StepRegistry.register(DynaudnormConservativeStep)

LOGGER.info("Registered %d step classes", len(StepRegistry.get_registered_types()))
