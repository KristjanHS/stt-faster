"""Concrete implementation of TranscriptionService using faster-whisper."""

import logging
from typing import Callable

from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.preprocess.orchestrator import PreprocessResult
from backend.services.interfaces import TranscriptionRequest, TranscriptionResult
from backend.transcribe import TranscriptionMetrics, transcribe
from backend.variants.executor import (
    is_baseline_config,
    is_minimal_config,
    transcribe_with_baseline_params,
    transcribe_with_minimal_params,
)

LOGGER = logging.getLogger(__name__)


class WhisperTranscriptionService:
    """Concrete implementation using faster-whisper."""

    def __init__(
        self,
        preprocess_config: PreprocessConfig,
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult],
        transcription_config: TranscriptionConfig,
    ):
        self._preprocess_config = preprocess_config
        self._preprocess_runner = preprocess_runner
        self._transcription_config = transcription_config

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio and return result with metrics."""

        metrics_container: dict[str, TranscriptionMetrics] = {}

        def _collect(metrics: TranscriptionMetrics) -> None:
            metrics_container["value"] = metrics

        # Use stored configurations
        preprocess_config = self._preprocess_config
        transcription_config = self._transcription_config

        # Determine transcription path based on config type
        # Check baseline first (baseline is a subset of minimal, but handled separately)
        is_baseline = is_baseline_config(transcription_config)
        is_minimal = is_minimal_config(transcription_config) if not is_baseline else False

        # Run transcription based on config type
        if is_baseline:
            # For baseline config, use baseline transcription (only passes language and vad_filter)
            payload = transcribe_with_baseline_params(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config=preprocess_config,
                preprocess_runner=self._preprocess_runner,
                transcription_config=transcription_config,
                metrics_collector=_collect,
            )
        elif is_minimal:
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
                preprocess_config_provider=lambda: self._preprocess_config,
                preprocess_runner=self._preprocess_runner,
                transcription_config_provider=lambda: self._transcription_config,
                metrics_collector=_collect,
            )

        return TranscriptionResult(metrics=metrics_container.get("value"), payload=payload)
