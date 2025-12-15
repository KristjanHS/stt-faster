"""Variant-aware transcription service implementation."""

import json
import logging
from pathlib import Path
from typing import Any

from backend.preprocess.config import PreprocessConfig
from backend.services.interfaces import TranscriptionRequest, TranscriptionResult, TranscriptionService
from backend.transcribe import TranscriptionMetrics
from backend.variants.executor import (
    create_variant_preprocess_runner,
    create_variant_transcribe_config,
    is_baseline_config,
    is_minimal_config,
)
from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)


class VariantTranscriptionService:
    """Transcription service that uses variant configuration."""

    def __init__(
        self,
        variant: Variant,
        preset: str,
        language: str | None,
        output_format: str,
        base_transcription_service: TranscriptionService | None = None,
    ):
        self.variant = variant
        self.preset = preset
        self.language = language
        self.output_format = output_format

        # Build preprocessing config
        self.preprocess_config = PreprocessConfig()  # Use defaults
        if not any(step.enabled for step in variant.preprocess_steps) and not variant.custom_preprocess_runner:
            self.preprocess_config.enabled = False

        # Create preprocessing runner from variant
        self.preprocess_runner = create_variant_preprocess_runner(
            variant,
            self.preprocess_config,
            base_name=None,
            datetime_suffix=None,
            output_dir=None,
            copy_intermediate=False,
        )

        # Get transcription config based on variant preset
        self.transcription_config = create_variant_transcribe_config(variant)

        # Store base service for fallback if needed
        self.base_transcription_service = base_transcription_service

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio using variant configuration."""
        LOGGER.debug("Using variant %d: %s", self.variant.number, self.variant.name)

        metrics_container: dict[str, TranscriptionMetrics] = {}

        def _collect(metrics: TranscriptionMetrics) -> None:
            metrics_container["value"] = metrics

        # Determine transcription path based on config type
        # Check baseline first (baseline is a subset of minimal, but handled separately)
        is_baseline = is_baseline_config(self.transcription_config)
        is_minimal = is_minimal_config(self.transcription_config) if not is_baseline else False

        # Run transcription based on config type
        if is_baseline:
            # For baseline config, use baseline transcription (only passes language and vad_filter)
            from backend.variants.executor import transcribe_with_baseline_params as transcribe_baseline_fn

            payload = transcribe_baseline_fn(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config=self.preprocess_config,
                preprocess_runner=self.preprocess_runner,
                transcription_config=self.transcription_config,
                metrics_collector=_collect,
            )
        elif is_minimal:
            # For minimal config, use the internal function that omits parameters
            from backend.variants.executor import transcribe_with_minimal_params as transcribe_minimal_fn

            payload = transcribe_minimal_fn(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config=self.preprocess_config,
                preprocess_runner=self.preprocess_runner,
                transcription_config=self.transcription_config,
                metrics_collector=_collect,
            )
        else:
            # Use standard transcription
            from backend.transcribe import transcribe as transcribe_fn

            payload = transcribe_fn(
                path=request.audio_path,
                preset=request.preset,
                language=request.language,
                preprocess_config_provider=lambda: self.preprocess_config,
                preprocess_runner=self.preprocess_runner,
                transcription_config_provider=lambda: self.transcription_config,
                metrics_collector=_collect,
            )

        # Write output in the specified format
        self._write_output(request.output_path, payload, self.output_format)

        return TranscriptionResult(metrics=metrics_container.get("value"), payload=payload)

    def _write_output(self, output_path: str, payload: dict[str, Any], output_format: str) -> list[Path]:
        """Write transcription output in the specified format."""
        created_files: list[Path] = []
        base_path = Path(output_path)

        if output_format in ("txt", "both"):
            txt_path = base_path.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            segments: list[dict[str, Any]] = payload.get("segments", [])
            with open(txt_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    if "text" in segment:
                        text = segment["text"]
                        if isinstance(text, str):
                            f.write(text)
                            f.write("\n")
            created_files.append(txt_path)

        if output_format in ("json", "both"):
            json_path = base_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            created_files.append(json_path)

        return created_files
