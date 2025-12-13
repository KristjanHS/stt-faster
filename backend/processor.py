"""File processing logic for transcription automation."""

import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING, Union

from backend.components import FileMoverPolicy, FileProcessor, FolderScanner, RunSummarizer
from backend.types import FileProcessingStats
from backend.run_config import RunConfig
from backend.services.interfaces import (
    FileMover,
    OutputWriter,
    StateStore,
    TranscriptionService,
)
from backend.transcribe import DEFAULT_OUTPUT_FORMAT

if TYPE_CHECKING:
    from backend.variants.variant import Variant

LOGGER = logging.getLogger(__name__)


class TranscriptionProcessor:
    """Orchestrates audio file transcription using separate components."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        state_store: StateStore,
        file_mover: FileMover,
        output_writer: OutputWriter,
        run_config: RunConfig | None = None,
        # Backward compatibility parameters
        input_folder: str | Path | None = None,
        preset: str = "et-large",
        language: str | None = None,
        output_format: str | None = None,
        variant: Union["Variant", None] = None,
        disable_file_moving: bool = False,
    ) -> None:
        """Initialize the processor with its component dependencies.

        Args:
            transcription_service: Service for transcribing audio files
            state_store: Service for managing transcription state
            file_mover: Service for moving files
            output_writer: Service for writing transcription output
            run_config: Complete run configuration with all settings (new API)
            input_folder: Folder containing audio files to process (legacy API)
            preset: Model preset for transcription (legacy API)
            language: Force specific language code (legacy API)
            output_format: Output format - "txt" or "json" (legacy API)
            variant: Optional variant to use for transcription (legacy API)
            disable_file_moving: If True, don't move files after processing (useful for multi-variant runs)
        """
        # Handle backward compatibility: create RunConfig from legacy parameters if needed
        if run_config is None:
            if input_folder is None:
                raise ValueError("Either run_config or input_folder must be provided")
            run_config = RunConfig.from_env_and_variant(Path(input_folder), variant)
            run_config.model_preset = preset
            run_config.language = language
            run_config.output_format = output_format or DEFAULT_OUTPUT_FORMAT

        self.run_config = run_config
        self.effective_config = run_config.get_effective_config()

        # Extract settings for backward compatibility and convenience
        self.input_folder = self.effective_config.input_folder
        self.preset = self.effective_config.model_preset
        self.language = self.effective_config.language
        self.output_format = self.effective_config.output_format
        self._disable_file_moving = disable_file_moving

        # For backward compatibility with existing code that expects these attributes
        self.db = None  # Will be removed once all references are updated
        self._variant = variant

        self._output_base_dir: Path | None = None  # For multi-variant: base dir for outputs
        self._variant_number: int | None = None  # For multi-variant: variant number for filename prefix
        self._variant_name: str | None = None  # For multi-variant: variant name for filename prefix

        # Initialize components
        self._folder_scanner = FolderScanner(self.input_folder)
        self._file_mover_policy = FileMoverPolicy(self.input_folder, file_mover)
        self._file_processor = FileProcessor(
            transcription_service=transcription_service,
            state_store=state_store,
            output_writer=output_writer,
            file_mover_policy=self._file_mover_policy,
            processor_ref=self,
        )
        self._run_summarizer = RunSummarizer(state_store)

        # For backward compatibility
        self._move = file_mover.move

        LOGGER.debug("Processor initialized for folder: %s", self.input_folder)
        LOGGER.debug("Run configuration: %s", self.run_config.to_dict())

    @property
    def processed_folder(self) -> Path:
        """Get the processed folder path (backward compatibility)."""
        return self._file_mover_policy.processed_folder

    @property
    def failed_folder(self) -> Path:
        """Get the failed folder path (backward compatibility)."""
        return self._file_mover_policy.failed_folder

    def scan_folder(self) -> list[str]:
        """Scan the input folder for audio files.

        Returns:
            List of audio file paths found
        """
        return self._folder_scanner.scan_folder()

    def get_files_to_process(self) -> list[str]:
        """Get list of files to process (all files in input folder).

        File location is the source of truth:
        - Files in input folder = need processing
        - Files in processed/ = already done
        - Files in failed/ = previously failed

        Returns:
            List of file paths to process
        """
        return self._folder_scanner.get_files_to_process()

    def process_file(self, file_path: str) -> FileProcessingStats:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            FileProcessingStats describing the outcome
        """
        return self._file_processor.process_file(file_path)

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

        # Use the effective config snapshot from run_config
        config_snapshot = self.effective_config.preprocess

        # Get all files in the input folder
        files_to_process = self.get_files_to_process()

        # Process them all
        results = self.process_all_files(files_to_process)
        results["files_found"] = len(files_to_process)
        total_processing_time = time.time() - run_start
        results["run_statistics"] = self._run_summarizer.summarize_run(
            results=results,
            config_snapshot=config_snapshot,
            total_processing_time=total_processing_time,
            input_folder=self.input_folder,
            preset=self.preset,
            language=self.language,
            output_format=self.output_format,
        )

        return results
