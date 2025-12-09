"""File processing logic for transcription automation."""

import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.database import FileMetricRecord, RunRecord, TranscriptionDatabase
from backend.exceptions import DatabaseError
from backend.preprocess.config import PreprocessConfig
from backend.transcribe import DEFAULT_OUTPUT_FORMAT, TranscriptionMetrics, transcribe
from backend.variants.executor import (
    create_variant_preprocess_runner,
    create_variant_transcribe_config,
    transcribe_with_minimal_params,
)
from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}


def _create_variant_transcribe_function(
    variant: Variant,
    preset: str,
    language: str | None,
    output_format: str,
) -> Callable[[str, str, str, str | None], TranscriptionMetrics | None]:
    """Create a transcribe function that uses variant configuration.

    Args:
        variant: Variant definition to use
        preset: Model preset for transcription
        language: Optional language code
        output_format: Output format ("txt", "json", or "both")

    Returns:
        A transcribe function matching TranscriptionProcessor's expected signature:
        (audio: str, output_path: str, preset: str, language: str | None) -> TranscriptionMetrics | None
    """
    # Build preprocessing config
    preprocess_config = PreprocessConfig()  # Use defaults
    if not any(step.enabled for step in variant.preprocess_steps) and not variant.custom_preprocess_runner:
        preprocess_config.enabled = False

    # Create preprocessing runner from variant
    preprocess_runner = create_variant_preprocess_runner(
        variant,
        preprocess_config,
        base_name=None,
        datetime_suffix=None,
        output_dir=None,
        copy_intermediate=False,
    )

    # Get transcription config based on variant preset
    transcription_config = create_variant_transcribe_config(variant)

    def _preprocess_config_provider() -> PreprocessConfig:
        return preprocess_config

    def _transcription_config_provider() -> Any:  # TranscriptionConfig
        return transcription_config

    def _variant_transcribe(
        audio: str, output_path: str, preset: str, language: str | None = None
    ) -> TranscriptionMetrics | None:
        """Transcribe using variant configuration."""
        metrics_container: dict[str, TranscriptionMetrics] = {}

        def _collect(metrics: TranscriptionMetrics) -> None:
            metrics_container["value"] = metrics

        # Run transcription based on variant preset
        if variant.transcription_preset == "minimal":
            # For minimal preset, use the internal function that omits parameters
            payload = transcribe_with_minimal_params(
                path=audio,
                preset=preset,
                language=language,
                preprocess_config=preprocess_config,
                preprocess_runner=preprocess_runner,
                transcription_config=transcription_config,
                transcription_overrides=variant.transcription_overrides,
                metrics_collector=_collect,
            )
        else:
            # Use standard transcription with full config and metrics collection
            payload = transcribe(
                path=audio,
                preset=preset,
                language=language,
                preprocess_config_provider=_preprocess_config_provider,
                preprocess_runner=preprocess_runner,
                transcription_config_provider=_transcription_config_provider,
                metrics_collector=_collect,
            )

        # Write output in the specified format
        if output_format == "txt":
            segments = payload.get("segments", [])
            with open(output_path, "w", encoding="utf-8") as text_file:
                for segment in segments:
                    text_file.write(segment["text"])
                    text_file.write("\n")
        elif output_format == "json":
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(payload, json_file, ensure_ascii=False, indent=2)
        elif output_format == "both":
            # Write both txt and json files
            base_path = Path(output_path)
            txt_path = base_path.with_suffix(".txt")
            json_path = base_path.with_suffix(".json")

            # Write txt
            segments = payload.get("segments", [])
            with open(txt_path, "w", encoding="utf-8") as text_file:
                for segment in segments:
                    text_file.write(segment["text"])
                    text_file.write("\n")

            # Write json
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(payload, json_file, ensure_ascii=False, indent=2)

        return metrics_container.get("value")

    return _variant_transcribe


# Folder names for processed and failed files
PROCESSED_FOLDER_NAME = "processed"
FAILED_FOLDER_NAME = "failed"


@dataclass(slots=True)
class FileProcessingStats:
    """Results captured for each processed file."""

    file_path: str
    status: str
    error_message: str | None = None
    metrics: TranscriptionMetrics | None = None


class TranscriptionProcessor:
    """Processes audio files and manages their lifecycle."""

    def __init__(
        self,
        db: TranscriptionDatabase,
        input_folder: str | Path,
        preset: str = "et-large",
        language: str | None = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        transcribe_fn: Callable[[str, str, str, str | None], TranscriptionMetrics | None] | None = None,
        preprocess_config_provider: Callable[[], PreprocessConfig] | None = None,
        move_fn: Callable[[str, str], str | None] = shutil.move,
        variant: Variant | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            db: Database instance for tracking state
            input_folder: Folder containing audio files to process
            preset: Model preset for transcription (default: 'et-large')
            language: Force specific language code (e.g., 'en', 'et'), None for auto-detect
            output_format: Output format - "txt" or "json" (default: from DEFAULT_OUTPUT_FORMAT)
            preprocess_config_provider: Callable that returns preprocessing configuration
            move_fn: Function to move files (for testing)
            variant: Optional variant to use for transcription (overrides default behavior)
        """
        self.db = db
        self.input_folder = Path(input_folder)
        self.preset = preset
        self.language = language
        self.output_format = output_format
        self._preprocess_config_provider = preprocess_config_provider or PreprocessConfig.from_env

        # If variant is provided and no custom transcribe_fn, create variant-aware transcribe function
        if variant is not None and transcribe_fn is None:
            LOGGER.debug("Using variant %d: %s", variant.number, variant.name)
            transcribe_fn = _create_variant_transcribe_function(
                variant=variant,
                preset=preset,
                language=language,
                output_format=output_format,
            )

        def _default_transcribe(
            audio: str, output_path: str, preset: str, language: str | None = None
        ) -> TranscriptionMetrics | None:
            metrics_container: dict[str, TranscriptionMetrics] = {}

            def _collect(metrics: TranscriptionMetrics) -> None:
                metrics_container["value"] = metrics

            payload = transcribe(
                audio,
                preset,
                language=language,
                preprocess_config_provider=self._preprocess_config_provider,
                metrics_collector=_collect,
            )

            # Write output in the specified format
            if self.output_format == "txt":
                segments = payload.get("segments", [])
                with open(output_path, "w", encoding="utf-8") as text_file:
                    for segment in segments:
                        text_file.write(segment["text"])
                        text_file.write("\n")
            elif self.output_format == "json":
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(payload, json_file, ensure_ascii=False, indent=2)
            elif self.output_format == "both":
                # Write both txt and json files
                base_path = Path(output_path)
                txt_path = base_path.with_suffix(".txt")
                json_path = base_path.with_suffix(".json")

                # Write txt
                segments = payload.get("segments", [])
                with open(txt_path, "w", encoding="utf-8") as text_file:
                    for segment in segments:
                        text_file.write(segment["text"])
                        text_file.write("\n")

                # Write json
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(payload, json_file, ensure_ascii=False, indent=2)

            return metrics_container.get("value")

        self._transcribe: Callable[[str, str, str, str | None], TranscriptionMetrics | None] = (
            transcribe_fn or _default_transcribe
        )
        self._move = move_fn

        # Create subdirectories for processed and failed files
        self.processed_folder = self.input_folder / PROCESSED_FOLDER_NAME
        self.failed_folder = self.input_folder / FAILED_FOLDER_NAME
        self.processed_folder.mkdir(exist_ok=True)
        self.failed_folder.mkdir(exist_ok=True)

        LOGGER.debug("Processor initialized for folder: %s", self.input_folder)

    def scan_folder(self) -> list[str]:
        """Scan the input folder for audio files.

        Returns:
            List of audio file paths found
        """
        audio_files: list[Path] = []

        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.extend(list(self.input_folder.glob(f"*{ext}")))

        # Convert to strings and filter out files in subdirectories
        file_paths = [str(f) for f in audio_files if f.parent.resolve() == self.input_folder.resolve()]

        LOGGER.debug("Found %d audio files in %s", len(file_paths), self.input_folder)
        return file_paths

    def get_files_to_process(self) -> list[str]:
        """Get list of files to process (all files in input folder).

        File location is the source of truth:
        - Files in input folder = need processing
        - Files in processed/ = already done
        - Files in failed/ = previously failed

        Returns:
            List of file paths to process
        """
        audio_files = self.scan_folder()
        LOGGER.debug("Found %d files to process", len(audio_files))
        return audio_files

    def process_file(self, file_path: str) -> FileProcessingStats:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            FileProcessingStats describing the outcome
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            error_msg = f"File not found: {file_path}"
            LOGGER.error(error_msg)
            self.db.update_status(file_path, "failed", error_msg)
            return FileProcessingStats(file_path=file_path, status="failed", error_message=error_msg)

        LOGGER.debug("Processing file: %s", file_path)

        try:
            # Generate output path with appropriate extension
            if self.output_format == "both":
                # For both format, we'll use a base path and create both files
                output_path = file_path_obj.with_suffix(".txt")  # Base path
                output_files = [file_path_obj.with_suffix(".txt"), file_path_obj.with_suffix(".json")]
            else:
                output_ext = ".txt" if self.output_format == "txt" else ".json"
                output_path = file_path_obj.with_suffix(output_ext)
                output_files = [output_path]

            # Perform transcription and capture metrics
            metrics = self._transcribe(str(file_path_obj), str(output_path), self.preset, self.language)

            # Move audio and all output files to processed folder
            if audio_file := file_path_obj:
                if audio_file.exists():
                    dest_audio = self.processed_folder / audio_file.name
                    self._move(str(audio_file), str(dest_audio))
                    LOGGER.debug("Moved audio to: %s", dest_audio)

            for out_file in output_files:
                if out_file.exists():
                    dest_output = self.processed_folder / out_file.name
                    self._move(str(out_file), str(dest_output))
                    LOGGER.debug("Moved output to: %s", dest_output)

            # Log success in database (for history/statistics)
            # Note: File location is source of truth, not database status
            self.db.update_status(file_path, "completed")

            LOGGER.debug("Successfully processed: %s", file_path)
            return FileProcessingStats(file_path=file_path, status="completed", metrics=metrics)

        except Exception as error:
            error_msg = f"{type(error).__name__}: {error}"
            LOGGER.error("Failed to process %s: %s", file_path, error_msg, exc_info=True)
            self.db.update_status(file_path, "failed", error_msg)

            # Move failed file to failed folder
            self._move_to_failed(file_path_obj)
            return FileProcessingStats(file_path=file_path, status="failed", error_message=error_msg)

    def _move_to_processed(self, audio_file: Path, output_file: Path) -> None:
        """Move successfully processed files to the processed subfolder.

        Args:
            audio_file: Path to audio file
            output_file: Path to output file (txt or json)

        Raises:
            Exception: If file move operation fails
        """
        if audio_file.exists():
            dest_audio = self.processed_folder / audio_file.name
            self._move(str(audio_file), str(dest_audio))
            LOGGER.debug("Moved audio to: %s", dest_audio)

        if output_file.exists():
            dest_output = self.processed_folder / output_file.name
            self._move(str(output_file), str(dest_output))
            LOGGER.debug("Moved output to: %s", dest_output)

    def _move_to_failed(self, audio_file: Path) -> None:
        """Move failed files to the failed subfolder.

        Args:
            audio_file: Path to audio file
        """
        try:
            if audio_file.exists():
                dest_audio = self.failed_folder / audio_file.name
                self._move(str(audio_file), str(dest_audio))
                LOGGER.debug("Moved failed file to: %s", dest_audio)

        except Exception as error:
            LOGGER.warning("Failed to move file to failed folder: %s", error)

    def process_all_files(self, file_paths: list[str]) -> dict[str, Any]:
        """Process all files in the provided list.

        Args:
            file_paths: List of file paths to process

        Returns:
            Dictionary with file counts and per-file statistics
        """
        if not file_paths:
            LOGGER.debug("No files to process")
            return {"succeeded": 0, "failed": 0, "file_stats": []}

        LOGGER.debug("Processing %d files", len(file_paths))

        succeeded = 0
        failed = 0
        stats: list[FileProcessingStats] = []

        for file_path in file_paths:
            result = self.process_file(file_path)
            stats.append(result)
            if result.status == "completed":
                succeeded += 1
            else:
                failed += 1

        LOGGER.info(
            "Processing complete: %d succeeded, %d failed",
            succeeded,
            failed,
        )

        return {"succeeded": succeeded, "failed": failed, "file_stats": stats}

    def process_folder(self) -> dict[str, Any]:
        """Main entry point: scan folder and process all files found.

        File location is the source of truth - if file is in the input folder,
        it needs processing. No database checks for "completed" status.

        Returns:
            Dictionary with processing results
        """
        LOGGER.debug("Starting folder processing")
        run_start = time.time()
        config_snapshot = self._preprocess_config_provider()

        # Get all files in the input folder
        files_to_process = self.get_files_to_process()

        # Process them all
        results = self.process_all_files(files_to_process)
        results["files_found"] = len(files_to_process)
        total_processing_time = time.time() - run_start
        results["run_statistics"] = self._record_run_metadata(results, config_snapshot, total_processing_time)

        return results

    def _record_run_metadata(
        self,
        results: dict[str, Any],
        config_snapshot: PreprocessConfig,
        total_processing_time: float,
    ) -> dict[str, Any]:
        """Build and persist run metadata and file metrics."""
        file_stats: list[FileProcessingStats] = results.get("file_stats", [])
        metrics_list = [entry.metrics for entry in file_stats if entry.metrics]

        # Calculate aggregates
        preprocess_time_total = sum(metric.preprocess_duration for metric in metrics_list)
        transcribe_time_total = sum(metric.transcribe_duration for metric in metrics_list)
        audio_durations = [metric.audio_duration for metric in metrics_list if metric.audio_duration is not None]
        total_audio_duration = sum(audio_durations) if audio_durations else None

        speed_values = [metric.speed_ratio for metric in metrics_list if metric.speed_ratio is not None]
        average_speed_ratio = sum(speed_values) / len(speed_values) if speed_values else None

        # Determine representative configuration from the first file if available
        # (Assuming batch processing uses consistent settings)
        if metrics_list:
            sample_metric = metrics_list[0]
            preprocess_profile = sample_metric.preprocess_profile
            target_sample_rate = sample_metric.target_sample_rate
            target_channels = sample_metric.target_channels
            loudnorm_preset = sample_metric.loudnorm_preset
            model_id = sample_metric.model_id
            device = sample_metric.device
            compute_type = sample_metric.compute_type
            beam_size = sample_metric.beam_size
            word_timestamps = sample_metric.word_timestamps
        else:
            # Fallback to config if no files processed successfully
            preprocess_profile = config_snapshot.profile
            target_sample_rate = config_snapshot.target_sample_rate
            target_channels = config_snapshot.target_channels
            loudnorm_preset = config_snapshot.loudnorm_preset
            model_id = None
            device = None
            compute_type = None
            beam_size = None
            word_timestamps = None

        # Create RunRecord
        run_record = RunRecord(
            recorded_at=datetime.now(timezone.utc),
            input_folder=str(self.input_folder),
            preset=self.preset,
            language=self.language,
            preprocess_enabled=config_snapshot.enabled,
            preprocess_profile=preprocess_profile,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            loudnorm_preset=loudnorm_preset,
            model_id=model_id,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            files_found=results.get("files_found", 0),
            succeeded=results.get("succeeded", 0),
            failed=results.get("failed", 0),
            total_processing_time=total_processing_time,
            total_preprocess_time=preprocess_time_total,
            total_transcribe_time=transcribe_time_total,
            total_audio_duration=total_audio_duration,
            speed_ratio=average_speed_ratio,
        )

        try:
            # 1. Record Run and get ID
            run_id = self.db.record_run(run_record)

            # 2. Record File Metrics
            for entry in file_stats:
                if not entry.metrics:
                    # Record failed files with basic info if needed,
                    # but FileMetricRecord ensures we have a path and status.
                    # We can create a skeleton record for failures.
                    # Set required NOT NULL fields from config_snapshot
                    file_record = FileMetricRecord(
                        run_id=run_id,
                        recorded_at=datetime.now(timezone.utc),
                        audio_path=entry.file_path,
                        preset=self.preset,
                        status=entry.status,
                        error_message=entry.error_message,
                        # Required NOT NULL fields
                        total_processing_time=0.0,
                        transcribe_duration=0.0,
                        preprocess_duration=0.0,
                        preprocess_enabled=config_snapshot.enabled,
                        preprocess_profile=config_snapshot.profile,
                        target_sample_rate=config_snapshot.target_sample_rate,
                    )
                else:
                    m = entry.metrics
                    file_record = FileMetricRecord(
                        run_id=run_id,
                        recorded_at=datetime.now(timezone.utc),
                        audio_path=entry.file_path,
                        preset=self.preset,
                        status=entry.status,
                        requested_language=self.language,
                        applied_language=m.applied_language,  # TranscriptionMetrics.applied_language
                        detected_language=m.detected_language,
                        language_probability=m.language_probability,
                        audio_duration=m.audio_duration,
                        total_processing_time=m.total_processing_time,
                        transcribe_duration=m.transcribe_duration,
                        preprocess_duration=m.preprocess_duration,
                        speed_ratio=m.speed_ratio,
                        preprocess_enabled=m.preprocess_enabled,
                        preprocess_profile=m.preprocess_profile,
                        target_sample_rate=m.target_sample_rate,
                        target_channels=m.target_channels,
                        preprocess_snr_before=m.preprocess_snr_before,
                        preprocess_snr_after=m.preprocess_snr_after,
                        preprocess_steps=m.preprocess_steps,
                        rnnoise_model=m.rnnoise_model,
                        rnnoise_mix=m.rnnoise_mix,
                        input_channels=m.input_channels,
                        input_sample_rate=m.input_sample_rate,
                        input_bit_depth=m.input_bit_depth,
                        input_format=m.input_format,
                        volume_adjustment_db=m.volume_adjustment_db,
                        resampler=m.resampler,
                        sample_format=m.sample_format,
                        loudnorm_preset=m.loudnorm_preset,
                        loudnorm_target_i=m.loudnorm_target_i,
                        loudnorm_target_tp=m.loudnorm_target_tp,
                        loudnorm_target_lra=m.loudnorm_target_lra,
                        loudnorm_backend=m.loudnorm_backend,
                        denoise_method=m.denoise_method,
                        denoise_library=m.denoise_library,
                        snr_estimation_method=m.snr_estimation_method,
                        beam_size=m.beam_size,
                        patience=m.patience,
                        word_timestamps=m.word_timestamps,
                        task=m.task,
                        chunk_length=m.chunk_length,
                        vad_filter=m.vad_filter,
                        vad_threshold=m.vad_threshold,
                        vad_min_speech_duration_ms=m.vad_min_speech_duration_ms,
                        vad_max_speech_duration_s=m.vad_max_speech_duration_s,
                        vad_min_silence_duration_ms=m.vad_min_silence_duration_ms,
                        vad_speech_pad_ms=m.vad_speech_pad_ms,
                        temperature=json.dumps(m.temperature) if m.temperature is not None else None,
                        temperature_increment_on_fallback=m.temperature_increment_on_fallback,
                        best_of=m.best_of,
                        compression_ratio_threshold=m.compression_ratio_threshold,
                        logprob_threshold=m.logprob_threshold,
                        no_speech_threshold=m.no_speech_threshold,
                        length_penalty=m.length_penalty,
                        repetition_penalty=m.repetition_penalty,
                        no_repeat_ngram_size=m.no_repeat_ngram_size,
                        suppress_tokens=m.suppress_tokens,
                        condition_on_previous_text=m.condition_on_previous_text,
                        initial_prompt=m.initial_prompt,
                        model_id=m.model_id,
                        device=m.device,
                        compute_type=m.compute_type,
                        output_format=self.output_format,
                        float_precision=m.float_precision,
                    )

                self.db.record_file_metric(file_record)

        except DatabaseError as exc:
            LOGGER.warning("Failed to record run metadata: %s", exc)
            run_id = None

        # Return stats for display/logging
        return {
            "updated_db": True,
            "run_id": run_id,
            "total_processing_time": total_processing_time,
            "total_preprocess_time": preprocess_time_total,
            "total_transcribe_time": transcribe_time_total,
            "average_speed_ratio": average_speed_ratio,
        }
