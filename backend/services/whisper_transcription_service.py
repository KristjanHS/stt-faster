"""Concrete implementation of TranscriptionService using faster-whisper."""

import logging
from typing import Callable

from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.services.interfaces import TranscriptionRequest, TranscriptionResult
from backend.transcribe import TranscriptionMetrics
from backend.variants.executor import is_minimal_config, transcribe_with_minimal_params

LOGGER = logging.getLogger(__name__)


class WhisperTranscriptionService:
    """Concrete implementation using faster-whisper."""

    def __init__(
        self,
        preprocess_config_provider: Callable[[], PreprocessConfig],
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
        transcription_config_provider: Callable[[], TranscriptionConfig],
    ):
        self._preprocess_config_provider = preprocess_config_provider
        self._preprocess_runner = preprocess_runner
        self._transcription_config_provider = transcription_config_provider

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio and return result with metrics."""
        from backend.transcribe import transcribe  # Import here to avoid circular imports

        metrics_container: dict[str, TranscriptionMetrics] = {}

        def _collect(metrics: TranscriptionMetrics) -> None:
            metrics_container["value"] = metrics

        # Get configurations
        preprocess_config = self._preprocess_config_provider()
        transcription_config = self._transcription_config_provider()

        # Determine if we should use minimal params
        is_minimal = is_minimal_config(transcription_config)

        # Run transcription based on config type
        if is_minimal:
            # For minimal config, use the internal function that omits parameters
            payload = transcribe_with_minimal_params(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config=preprocess_config,
                preprocess_runner=self._preprocess_runner,
                transcription_config=transcription_config,
                metrics_collector=_collect,
            )
        else:
            # Use standard transcription with full config and metrics collection
            payload = transcribe(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config_provider=self._preprocess_config_provider,
                preprocess_runner=self._preprocess_runner,
                transcription_config_provider=self._transcription_config_provider,
                metrics_collector=_collect,
            )

        return TranscriptionResult(metrics=metrics_container.get("value"), payload=payload)
