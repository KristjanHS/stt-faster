"""File processing logic for transcription automation."""

import logging
import shutil
from pathlib import Path
from typing import Any

from backend.transcribe import transcribe_to_json

from .database import TranscriptionDatabase

LOGGER = logging.getLogger(__name__)


class TranscriptionProcessor:
    """Processes audio files and manages their lifecycle."""

    def __init__(
        self,
        db: TranscriptionDatabase,
        input_folder: str | Path,
        preset: str = "distil",
    ) -> None:
        """Initialize the processor.

        Args:
            db: Database instance for tracking state
            input_folder: Folder containing audio files to process
            preset: Model preset for transcription (default: 'distil')
        """
        self.db = db
        self.input_folder = Path(input_folder)
        self.preset = preset

        # Create subdirectories for processed and failed files
        self.processed_folder = self.input_folder / "processed"
        self.failed_folder = self.input_folder / "failed"
        self.processed_folder.mkdir(exist_ok=True)
        self.failed_folder.mkdir(exist_ok=True)

        LOGGER.info("Processor initialized for folder: %s", self.input_folder)

    def scan_folder(self) -> list[str]:
        """Scan the input folder for audio files.

        Returns:
            List of audio file paths found
        """
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}
        audio_files: list[Path] = []

        for ext in audio_extensions:
            audio_files.extend(list(self.input_folder.glob(f"*{ext}")))

        # Convert to strings and filter out files in subdirectories
        file_paths = [str(f) for f in audio_files if f.parent.resolve() == self.input_folder.resolve()]

        LOGGER.info("Found %d audio files in %s", len(file_paths), self.input_folder)
        return file_paths

    def register_files(self) -> int:
        """Scan folder and register any new files in the database.

        Returns:
            Number of new files registered
        """
        audio_files = self.scan_folder()
        new_files = 0

        for file_path in audio_files:
            existing = self.db.get_status(file_path)
            if not existing:
                self.db.add_file(file_path, "pending")
                new_files += 1

        LOGGER.info("Registered %d new files", new_files)
        return new_files

    def process_file(self, file_path: str) -> bool:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if processing succeeded, False otherwise
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            error_msg = f"File not found: {file_path}"
            LOGGER.error(error_msg)
            self.db.update_status(file_path, "failed", error_msg)
            return False

        LOGGER.info("Processing file: %s", file_path)

        try:
            # Generate output JSON path (same name, .json extension)
            json_path = file_path_obj.with_suffix(".json")

            # Call the existing transcription function
            transcribe_to_json(str(file_path_obj), str(json_path), self.preset)

            # Move both audio and JSON to processed folder
            # (must succeed before marking as completed in database)
            self._move_to_processed(file_path_obj, json_path)

            # Update database status only after successful move
            self.db.update_status(file_path, "completed")

            LOGGER.info("Successfully processed: %s", file_path)
            return True

        except Exception as error:
            error_msg = f"{type(error).__name__}: {error}"
            LOGGER.error("Failed to process %s: %s", file_path, error_msg, exc_info=True)
            self.db.update_status(file_path, "failed", error_msg)

            # Move failed file to failed folder
            self._move_to_failed(file_path_obj)
            return False

    def _move_to_processed(self, audio_file: Path, json_file: Path) -> None:
        """Move successfully processed files to the processed subfolder.

        Args:
            audio_file: Path to audio file
            json_file: Path to JSON file

        Raises:
            Exception: If file move operation fails
        """
        if audio_file.exists():
            dest_audio = self.processed_folder / audio_file.name
            shutil.move(str(audio_file), str(dest_audio))
            LOGGER.debug("Moved audio to: %s", dest_audio)

        if json_file.exists():
            dest_json = self.processed_folder / json_file.name
            shutil.move(str(json_file), str(dest_json))
            LOGGER.debug("Moved JSON to: %s", dest_json)

    def _move_to_failed(self, audio_file: Path) -> None:
        """Move failed files to the failed subfolder.

        Args:
            audio_file: Path to audio file
        """
        try:
            if audio_file.exists():
                dest_audio = self.failed_folder / audio_file.name
                shutil.move(str(audio_file), str(dest_audio))
                LOGGER.debug("Moved failed file to: %s", dest_audio)

        except Exception as error:
            LOGGER.warning("Failed to move file to failed folder: %s", error)

    def process_pending(self) -> dict[str, int]:
        """Process all pending files.

        Returns:
            Dictionary with counts of succeeded and failed files
        """
        pending_files = self.db.get_files_by_status("pending")

        if not pending_files:
            LOGGER.info("No pending files to process")
            return {"succeeded": 0, "failed": 0}

        LOGGER.info("Processing %d pending files", len(pending_files))

        succeeded = 0
        failed = 0

        for file_info in pending_files:
            file_path = file_info["file_path"]
            if self.process_file(file_path):
                succeeded += 1
            else:
                failed += 1

        LOGGER.info(
            "Processing complete: %d succeeded, %d failed",
            succeeded,
            failed,
        )

        return {"succeeded": succeeded, "failed": failed}

    def process_folder(self) -> dict[str, Any]:
        """Main entry point: scan folder, register files, and process them.

        Returns:
            Dictionary with processing results
        """
        LOGGER.info("Starting folder processing")

        # Register any new files
        new_files = self.register_files()

        # Process all pending files
        results = self.process_pending()
        results["new_files"] = new_files

        return results
