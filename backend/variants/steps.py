"""New class-based preprocessing step system with typed configurations."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.errors import StepExecutionError
from backend.preprocess.metrics import StepMetrics

LOGGER = logging.getLogger(__name__)


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
    """Abstract base class for preprocessing steps."""

    config: StepConfigT  # Type annotation for type checker

    def __init__(self, config: StepConfigT | None = None):
        """Initialize step with configuration.

        Args:
            config: Step-specific configuration. If None, uses default config.
        """
        self.config = config or self.get_default_config()
        # Validate that config is the correct type at construction time
        expected_type = type(self.get_default_config())
        if not isinstance(self.config, expected_type):
            raise TypeError(f"Config must be {expected_type.__name__}, got {type(self.config).__name__}")

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

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute the preprocessing step.

        Args:
            input_path: Path to input audio file
            output_path: Path where to write output audio file
            global_config: Global preprocessing configuration
            step_index: Index of this step in the pipeline

        Returns:
            StepMetrics with execution results
        """
        pass


class ResampleStep(Step[ResampleStepConfig]):
    """Simple resampling step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "resample"

    @classmethod
    def get_default_config(cls) -> ResampleStepConfig:
        return ResampleStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute resampling step."""
        import ffmpeg  # type: ignore[import-untyped]

        start = time.time()
        try:
            stream = ffmpeg.input(str(input_path))  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
            stream = ffmpeg.output(  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
                stream,  # type: ignore[reportUnknownArgumentType]
                str(output_path),
                ac=1,  # Force mono for Whisper/faster-whisper
                ar=global_config.target_sample_rate,
                acodec="pcm_s16le",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        except Exception as exc:
            raise StepExecutionError("simple_resample", f"ffmpeg error: {exc}") from exc

        duration = time.time() - start
        LOGGER.info("Resample step completed in %.2fs", duration)
        return StepMetrics(name="simple_resample", backend="ffmpeg", duration=duration)


class FFmpegStep(Step[FFmpegStepConfig]):
    """FFmpeg pipeline step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "ffmpeg"

    @classmethod
    def get_default_config(cls) -> FFmpegStepConfig:
        return FFmpegStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute FFmpeg pipeline step."""
        from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline

        start = time.time()
        LOGGER.info(
            "FFmpeg params: rnnoise_mix=%.2f, loudnorm_preset=%s, rnnoise_model=%s",
            global_config.rnnoise_mix,
            global_config.loudnorm_preset,
            global_config.rnnoise_model or "default",
        )

        metrics = run_ffmpeg_pipeline(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            rnnoise_mix=global_config.rnnoise_mix,
            loudnorm_preset=global_config.loudnorm_preset,
            rnnoise_model=global_config.rnnoise_model,
        )
        LOGGER.info("FFmpeg step completed in %.2fs", time.time() - start)
        return metrics


class DenoiseStep(Step[DenoiseStepConfig]):
    """Light denoising step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "denoise"

    @classmethod
    def get_default_config(cls) -> DenoiseStepConfig:
        return DenoiseStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute denoising step."""
        from backend.preprocess.steps.denoise_light import apply_light_denoise

        start = time.time()
        metrics = apply_light_denoise(
            input_path=input_path,
            output_path=output_path,
            sample_rate=global_config.target_sample_rate,
        )
        LOGGER.info("Denoise step completed in %.2fs", time.time() - start)
        return metrics


class DenoiseCustomStep(Step[DenoiseCustomStepConfig]):
    """Custom denoising step with configurable parameters."""

    @classmethod
    def get_step_type(cls) -> str:
        return "denoise_custom"

    @classmethod
    def get_default_config(cls) -> DenoiseCustomStepConfig:
        return DenoiseCustomStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute custom denoising step."""
        from backend.preprocess.steps.denoise_light import apply_light_denoise

        start = time.time()
        LOGGER.info(
            "Denoise params: noise_clip=%.2fs, n_std=%.2f, prop_decrease=%.2f",
            self.config.noise_clip_duration_s,
            self.config.n_std_thresh_stationary,
            self.config.prop_decrease,
        )

        metrics = apply_light_denoise(
            input_path=input_path,
            output_path=output_path,
            sample_rate=global_config.target_sample_rate,
            noise_clip_duration_s=self.config.noise_clip_duration_s,
            n_std_thresh_stationary=self.config.n_std_thresh_stationary,
            prop_decrease=self.config.prop_decrease,
        )
        LOGGER.info("Denoise custom step completed in %.2fs", time.time() - start)
        return metrics


class LoudnormOnlyStep(Step[LoudnormOnlyStepConfig]):
    """Loudness normalization only step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "loudnorm_only"

    @classmethod
    def get_default_config(cls) -> LoudnormOnlyStepConfig:
        return LoudnormOnlyStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute loudnorm only step."""
        from backend.variants.preprocess_steps import loudnorm_only

        start = time.time()
        metrics = loudnorm_only(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            loudnorm_preset=self.config.loudnorm_preset,
        )
        LOGGER.info("Loudnorm only step completed in %.2fs", time.time() - start)
        return metrics


class VolumeLimiterStep(Step[VolumeLimiterStepConfig]):
    """Volume adjustment with limiter step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "volume_limiter"

    @classmethod
    def get_default_config(cls) -> VolumeLimiterStepConfig:
        return VolumeLimiterStepConfig(volume_db=1.5)  # Default value needed

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute volume limiter step."""
        from backend.variants.preprocess_steps import volume_with_limiter

        start = time.time()
        LOGGER.info("Volume limiter params: volume_db=%.2f", self.config.volume_db)

        metrics = volume_with_limiter(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            volume_db=self.config.volume_db,
        )
        LOGGER.info("Volume limiter step completed in %.2fs", time.time() - start)
        return metrics


class PeakNormalize2passStep(Step[PeakNormalize2passStepConfig]):
    """Two-pass peak normalization step."""

    @classmethod
    def get_step_type(cls) -> str:
        return "peak_normalize_2pass"

    @classmethod
    def get_default_config(cls) -> PeakNormalize2passStepConfig:
        return PeakNormalize2passStepConfig()

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute peak normalize 2pass step."""
        from backend.variants.preprocess_steps import peak_normalize_2pass

        start = time.time()
        LOGGER.info(
            "Peak normalize params: target_db=%.1f, max_gain_db=%.1f", self.config.target_db, self.config.max_gain_db
        )

        metrics = peak_normalize_2pass(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            target_db=self.config.target_db,
            max_gain_db=self.config.max_gain_db,
        )
        LOGGER.info("Peak normalize 2pass step completed in %.2fs", time.time() - start)
        return metrics


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

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        global_config: PreprocessConfig,
        step_index: int,
    ) -> StepMetrics:
        """Execute sox peak normalize step."""
        from backend.variants.preprocess_steps import sox_peak_normalize

        start = time.time()
        LOGGER.info("SoX peak normalize params: target_db=%.1f", self.config.target_db)

        metrics = sox_peak_normalize(
            input_path=input_path,
            output_path=output_path,
            target_sample_rate=global_config.target_sample_rate,
            target_channels=global_config.target_channels or 1,
            target_db=self.config.target_db,
        )
        LOGGER.info("SoX peak normalize step completed in %.2fs", time.time() - start)
        return metrics


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
