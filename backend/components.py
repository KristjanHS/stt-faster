"""Separated components extracted from the TranscriptionProcessor god object."""

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from backend.database import RunRecord
from backend.exceptions import DatabaseError
from backend.preprocess.config import PreprocessConfig
from backend.services.interfaces import (
    FileMover,
    OutputWriter,
    StateStore,
    TranscriptionRequest,
    TranscriptionService,
)
from backend.transcribe import TranscriptionMetrics
from backend.types import FileProcessingStats

if TYPE_CHECKING:
    from backend.processor import TranscriptionProcessor

LOGGER = logging.getLogger(__name__)

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}

# Folder names for processed and failed files
PROCESSED_FOLDER_NAME = "processed"
FAILED_FOLDER_NAME = "failed"


class FileMoverPolicy:
    """Handles file movement policies for processed and failed files."""

    def __init__(self, input_folder: str | Path, file_mover: FileMover):
        """Initialize the file mover policy.

        Args:
            input_folder: Base input folder
            file_mover: Service for moving files
        """
        self.input_folder = Path(input_folder)
        self._file_mover = file_mover
        self.processed_folder = self.input_folder / PROCESSED_FOLDER_NAME
        self.failed_folder = self.input_folder / FAILED_FOLDER_NAME

        # Create subdirectories
        self.processed_folder.mkdir(exist_ok=True)
        self.failed_folder.mkdir(exist_ok=True)

    def move_to_processed(self, audio_file: Path, created_files: list[Path]) -> None:
        """Move successfully processed files to the processed subfolder.

        Args:
            audio_file: Path to audio file
            created_files: List of output files created during processing
        """
        if audio_file.exists():
            dest_audio = self.processed_folder / audio_file.name
            self._file_mover.move(str(audio_file), str(dest_audio))
            LOGGER.debug("Moved audio to: %s", dest_audio)

        # Move the files created by the output writer
        for created_file in created_files:
            if created_file.exists():
                dest_output = self.processed_folder / created_file.name
                self._file_mover.move(str(created_file), str(dest_output))
                LOGGER.debug("Moved output to: %s", dest_output)

    def move_to_failed(self, audio_file: Path) -> None:
        """Move failed files to the failed subfolder.

        Args:
            audio_file: Path to audio file
        """
        try:
            if audio_file.exists():
                dest_audio = self.failed_folder / audio_file.name
                self._file_mover.move(str(audio_file), str(dest_audio))
                LOGGER.debug("Moved failed file to: %s", dest_audio)

        except Exception as error:
            LOGGER.warning("Failed to move file to failed folder: %s", error)


class FileProcessor:
    """Handles per-file transcription processing."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        state_store: StateStore,
        output_writer: OutputWriter,
        file_mover_policy: FileMoverPolicy,
        processor_ref: "TranscriptionProcessor",
    ):
        """Initialize the file processor.

        Args:
            transcription_service: Service for transcribing audio files
            state_store: Service for managing transcription state
            output_writer: Service for writing transcription output
            file_mover_policy: Policy for moving processed files
            processor_ref: Reference to the parent TranscriptionProcessor for config access
        """
        self._transcription_service = transcription_service
        self._state_store = state_store
        self._output_writer = output_writer
        self._file_mover_policy = file_mover_policy
        self._processor_ref = processor_ref

    @property
    def preset(self) -> str:
        return self._processor_ref.preset

    @property
    def language(self) -> str | None:
        return self._processor_ref.language

    @property
    def output_format(self) -> str:
        return self._processor_ref.output_format

    @property
    def _disable_file_moving(self) -> bool:
        return self._processor_ref._disable_file_moving  # pyright: ignore[reportPrivateUsage]

    @property
    def _output_base_dir(self) -> Path | None:
        return self._processor_ref._output_base_dir  # pyright: ignore[reportPrivateUsage]

    @property
    def _variant_number(self) -> int | None:
        return self._processor_ref._variant_number  # pyright: ignore[reportPrivateUsage]

    @property
    def _variant_name(self) -> str | None:
        return self._processor_ref._variant_name  # pyright: ignore[reportPrivateUsage]

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
            self._state_store.update_status(file_path, "failed", error_msg)
            return FileProcessingStats(file_path=file_path, status="failed", error_message=error_msg)

        LOGGER.debug("Processing file: %s", file_path)

        try:
            # Generate output path with appropriate extension
            # If _output_base_dir is set (multi-variant mode), write outputs there
            if self._output_base_dir:
                # Preserve directory structure relative to input_folder to prevent collisions
                # when files with the same name exist in different subdirectories
                input_folder_path = self._file_mover_policy.input_folder.resolve()
                file_path_resolved = file_path_obj.resolve()

                try:
                    # Calculate relative path from input_folder to file
                    relative_path = file_path_resolved.relative_to(input_folder_path)
                    # Get parent directory (if file is in subdirectory) and stem
                    if relative_path.parent == Path("."):
                        # File is at root level - output directly to _output_base_dir
                        output_dir = self._output_base_dir
                        base_name = relative_path.stem
                    else:
                        # File is in subdirectory - preserve structure
                        output_dir = self._output_base_dir / relative_path.parent
                        base_name = relative_path.stem
                        # Ensure output directory exists
                        output_dir.mkdir(parents=True, exist_ok=True)
                except ValueError:
                    # File is not under input_folder (shouldn't happen in normal operation)
                    # Fall back to using just the filename
                    LOGGER.warning(
                        "File %s is not under input folder %s, using filename only",
                        file_path,
                        input_folder_path,
                    )
                    output_dir = self._output_base_dir
                    base_name = file_path_obj.stem

                # Add variant prefix if in multi-variant mode
                if self._variant_number is not None:
                    variant_prefix = f"variant_{self._variant_number:03d}_{self._variant_name}_"
                    base_name = f"{variant_prefix}{base_name}"

                if self.output_format == "both":
                    output_path = output_dir / f"{base_name}.txt"
                else:
                    output_ext = ".txt" if self.output_format == "txt" else ".json"
                    output_path = output_dir / f"{base_name}{output_ext}"
            else:
                # Normal mode: write next to input file
                if self.output_format == "both":
                    # For both format, we'll use a base path and create both files
                    output_path = file_path_obj.with_suffix(".txt")  # Base path
                else:
                    output_ext = ".txt" if self.output_format == "txt" else ".json"
                    output_path = file_path_obj.with_suffix(output_ext)

            # Perform transcription and capture metrics
            request = TranscriptionRequest(
                audio_path=str(file_path_obj),
                output_path=str(output_path),
                preset=self.preset,
                language=self.language,
            )
            result = self._transcription_service.transcribe(request)
            metrics = result.metrics

            # Write output using the service
            created_files = self._output_writer.write(str(output_path), result.payload, self.output_format)

            # Move audio and all output files to processed folder (unless disabled)
            if not self._disable_file_moving:
                if audio_file := file_path_obj:
                    if audio_file.exists():
                        dest_audio = self._file_mover_policy.processed_folder / audio_file.name
                        self._file_mover_policy._file_mover.move(str(audio_file), str(dest_audio))  # pyright: ignore[reportPrivateUsage]
                        LOGGER.debug("Moved audio to: %s", dest_audio)

                # Move the files created by the output writer (unless _output_base_dir is set)
                if not self._output_base_dir:
                    for created_file in created_files:
                        if created_file.exists():
                            dest_output = self._file_mover_policy.processed_folder / created_file.name
                            self._file_mover_policy._file_mover.move(str(created_file), str(dest_output))  # pyright: ignore[reportPrivateUsage]
                            LOGGER.debug("Moved output to: %s", dest_output)
                # When _output_base_dir is set (multi-variant mode), outputs stay in that location

            # Log success in database (for history/statistics)
            # Note: File location is source of truth, not database status
            self._state_store.update_status(file_path, "completed")

            LOGGER.debug("Successfully processed: %s", file_path)
            return FileProcessingStats(file_path=file_path, status="completed", metrics=metrics)

        except Exception as error:
            error_msg = f"{type(error).__name__}: {error}"
            LOGGER.error("Failed to process %s: %s", file_path, error_msg, exc_info=True)
            self._state_store.update_status(file_path, "failed", error_msg)

            # Move failed file to failed folder (unless disabled)
            if not self._disable_file_moving:
                self._file_mover_policy.move_to_failed(file_path_obj)
            return FileProcessingStats(file_path=file_path, status="failed", error_message=error_msg)


class RunSummarizer:
    """Handles aggregation and persistence of run metrics."""

    def __init__(self, state_store: StateStore):
        """Initialize the run summarizer.

        Args:
            state_store: Service for managing transcription state
        """
        self._state_store = state_store

    def summarize_run(
        self,
        results: dict[str, Any],
        config_snapshot: PreprocessConfig,
        total_processing_time: float,
        input_folder: str | Path,
        preset: str,
        language: str | None,
        output_format: str,
    ) -> dict[str, Any]:
        """Build and persist run metadata and file metrics.

        Args:
            results: Processing results with file stats
            config_snapshot: Configuration snapshot
            total_processing_time: Total time for the run
            input_folder: Input folder path
            preset: Model preset used
            language: Language setting
            output_format: Output format used

        Returns:
            Dictionary with run statistics
        """
        self._output_format = output_format  # Store for use in _persist_run_data
        """Build and persist run metadata and file metrics.

        Args:
            results: Processing results with file stats
            config_snapshot: Configuration snapshot
            total_processing_time: Total time for the run
            input_folder: Input folder path
            preset: Model preset used
            language: Language setting
            output_format: Output format used

        Returns:
            Dictionary with run statistics
        """
        file_stats: list[FileProcessingStats] = results.get("file_stats", [])
        metrics_list = [entry.metrics for entry in file_stats if entry.metrics]

        # Calculate aggregates
        preprocess_time_total = sum(metric.preprocess_duration for metric in metrics_list)
        transcribe_time_total = sum(metric.transcribe_duration for metric in metrics_list)
        audio_durations = [metric.audio_duration for metric in metrics_list if metric.audio_duration is not None]
        total_audio_duration = sum(audio_durations) if audio_durations else None

        speed_values = [metric.speed_ratio for metric in metrics_list if metric.speed_ratio is not None]
        average_speed_ratio = sum(speed_values) / len(speed_values) if speed_values else None

        # Create RunRecord with aggregated data
        run_record = self._create_run_record(
            results=results,
            config_snapshot=config_snapshot,
            total_processing_time=total_processing_time,
            input_folder=input_folder,
            preset=preset,
            language=language,
            output_format=output_format,
            preprocess_time_total=preprocess_time_total,
            transcribe_time_total=transcribe_time_total,
            total_audio_duration=total_audio_duration,
            average_speed_ratio=average_speed_ratio,
            metrics_list=metrics_list,
        )

        # Persist the run and file metrics
        run_id = self._persist_run_data(run_record, file_stats, config_snapshot, preset, language)

        # Return stats for display/logging
        return {
            "updated_db": True,
            "run_id": run_id,
            "total_processing_time": total_processing_time,
            "total_preprocess_time": preprocess_time_total,
            "total_transcribe_time": transcribe_time_total,
            "average_speed_ratio": average_speed_ratio,
        }

    def _create_run_record(
        self,
        results: dict[str, Any],
        config_snapshot: PreprocessConfig,
        total_processing_time: float,
        input_folder: str | Path,
        preset: str,
        language: str | None,
        output_format: str,
        preprocess_time_total: float,
        transcribe_time_total: float,
        total_audio_duration: float | None,
        average_speed_ratio: float | None,
        metrics_list: list[TranscriptionMetrics],
    ) -> RunRecord:
        """Create a RunRecord from the processing results."""
        import json
        from datetime import datetime, timezone

        # Determine representative configuration from the first file if available
        if metrics_list:
            sample_metric = metrics_list[0]
            # Use sample metric values for the run record
            run_config = self._extract_run_config_from_metrics(sample_metric, config_snapshot)
        else:
            # Fallback to config if no files processed successfully
            run_config = self._extract_run_config_from_config(config_snapshot)

        # Handle temperature conversion
        temperature_str = None
        if run_config["temperature"] is not None:
            temperature = run_config["temperature"]
            temperature_str = json.dumps(temperature) if isinstance(temperature, list) else str(temperature)

        return RunRecord(
            recorded_at=datetime.now(timezone.utc),
            input_folder=str(input_folder),
            preset=preset,
            language=language,
            preprocess_enabled=config_snapshot.enabled,
            **{k: v for k, v in run_config.items() if k != "temperature"},  # Exclude temperature, add it separately
            temperature=temperature_str,
            files_found=results.get("files_found", 0),
            succeeded=results.get("succeeded", 0),
            failed=results.get("failed", 0),
            total_processing_time=total_processing_time,
            total_preprocess_time=preprocess_time_total,
            total_transcribe_time=transcribe_time_total,
            total_audio_duration=total_audio_duration,
            speed_ratio=average_speed_ratio,
        )

    def _extract_run_config_from_metrics(
        self, sample_metric: TranscriptionMetrics, config_snapshot: PreprocessConfig
    ) -> dict[str, Any]:
        """Extract configuration values from metrics."""
        return {
            "preprocess_profile": sample_metric.preprocess_profile,
            "target_sample_rate": sample_metric.target_sample_rate,
            "target_channels": sample_metric.target_channels,
            "loudnorm_preset": sample_metric.loudnorm_preset,
            "volume_adjustment_db": sample_metric.volume_adjustment_db,
            "resampler": sample_metric.resampler,
            "sample_format": sample_metric.sample_format,
            "loudnorm_target_i": sample_metric.loudnorm_target_i,
            "loudnorm_target_tp": sample_metric.loudnorm_target_tp,
            "loudnorm_target_lra": sample_metric.loudnorm_target_lra,
            "loudnorm_backend": sample_metric.loudnorm_backend,
            "denoise_method": sample_metric.denoise_method,
            "denoise_library": sample_metric.denoise_library,
            "rnnoise_model": sample_metric.rnnoise_model,
            "rnnoise_mix": sample_metric.rnnoise_mix,
            "snr_estimation_method": sample_metric.snr_estimation_method,
            "model_id": sample_metric.model_id,
            "device": sample_metric.device,
            "compute_type": sample_metric.compute_type,
            "beam_size": sample_metric.beam_size,
            "patience": sample_metric.patience,
            "word_timestamps": sample_metric.word_timestamps,
            "task": sample_metric.task,
            "chunk_length": sample_metric.chunk_length,
            "vad_filter": sample_metric.vad_filter,
            "vad_threshold": sample_metric.vad_threshold,
            "vad_min_speech_duration_ms": sample_metric.vad_min_speech_duration_ms,
            "vad_max_speech_duration_s": sample_metric.vad_max_speech_duration_s,
            "vad_min_silence_duration_ms": sample_metric.vad_min_silence_duration_ms,
            "vad_speech_pad_ms": sample_metric.vad_speech_pad_ms,
            "temperature": sample_metric.temperature,
            "temperature_increment_on_fallback": sample_metric.temperature_increment_on_fallback,
            "best_of": sample_metric.best_of,
            "compression_ratio_threshold": sample_metric.compression_ratio_threshold,
            "logprob_threshold": sample_metric.logprob_threshold,
            "no_speech_threshold": sample_metric.no_speech_threshold,
            "length_penalty": sample_metric.length_penalty,
            "repetition_penalty": sample_metric.repetition_penalty,
            "no_repeat_ngram_size": sample_metric.no_repeat_ngram_size,
            "suppress_tokens": sample_metric.suppress_tokens,
            "condition_on_previous_text": sample_metric.condition_on_previous_text,
            "initial_prompt": sample_metric.initial_prompt,
        }

    def _extract_run_config_from_config(self, config_snapshot: PreprocessConfig) -> dict[str, Any]:
        """Extract configuration values from config snapshot when no metrics available."""
        return {
            "preprocess_profile": config_snapshot.profile,
            "target_sample_rate": config_snapshot.target_sample_rate,
            "target_channels": config_snapshot.target_channels,
            "loudnorm_preset": config_snapshot.loudnorm_preset,
            "volume_adjustment_db": None,
            "resampler": None,
            "sample_format": None,
            "loudnorm_target_i": None,
            "loudnorm_target_tp": None,
            "loudnorm_target_lra": None,
            "loudnorm_backend": None,
            "denoise_method": None,
            "denoise_library": None,
            "rnnoise_model": None,
            "rnnoise_mix": None,
            "snr_estimation_method": None,
            "model_id": None,
            "device": None,
            "compute_type": None,
            "beam_size": None,
            "patience": None,
            "word_timestamps": None,
            "task": None,
            "chunk_length": None,
            "vad_filter": None,
            "vad_threshold": None,
            "vad_min_speech_duration_ms": None,
            "vad_max_speech_duration_s": None,
            "vad_min_silence_duration_ms": None,
            "vad_speech_pad_ms": None,
            "temperature": None,
            "temperature_increment_on_fallback": None,
            "best_of": None,
            "compression_ratio_threshold": None,
            "logprob_threshold": None,
            "no_speech_threshold": None,
            "length_penalty": None,
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
            "suppress_tokens": None,
            "condition_on_previous_text": None,
            "initial_prompt": None,
        }

    def _persist_run_data(
        self,
        run_record: RunRecord,
        file_stats: list[FileProcessingStats],
        config_snapshot: PreprocessConfig,
        preset: str,
        language: str | None,
    ) -> int | None:
        """Persist run record and individual file metrics to the database."""
        import json
        from datetime import datetime, timezone

        from backend.database import FileMetricRecord

        try:
            # 1. Record Run and get ID
            run_id = self._state_store.record_run(run_record)

            # 2. Record File Metrics
            for entry in file_stats:
                if not entry.metrics:
                    # Record failed files with basic info if needed
                    file_record = FileMetricRecord(
                        run_id=run_id,
                        recorded_at=datetime.now(timezone.utc),
                        audio_path=entry.file_path,
                        preset=preset,
                        status=entry.status,
                        error_message=entry.error_message,
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
                        preset=preset,
                        status=entry.status,
                        requested_language=language,
                        applied_language=m.applied_language,
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
                        output_format=self._output_format,
                        float_precision=m.float_precision,
                    )

                self._state_store.record_file_metric(file_record)

            return run_id

        except DatabaseError as exc:
            LOGGER.error("Failed to record run metadata: %s", exc, exc_info=True)
            return None


class FolderScanner:
    """Handles discovery of audio files in input folders."""

    def __init__(self, input_folder: str | Path):
        """Initialize the scanner.

        Args:
            input_folder: Folder containing audio files to scan
        """
        self.input_folder = Path(input_folder)

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
