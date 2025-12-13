"""Factory for creating service instances."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.database import TranscriptionDatabase
from backend.preprocess.config import PreprocessConfig, TranscriptionConfig
from backend.preprocess.orchestrator import PreprocessResult, preprocess_audio
from backend.services.duckdb_state_store import DuckDBStateStore
from backend.services.interfaces import (
    FileMover,
    OutputWriter,
    StateStore,
    TranscriptionService,
)
from backend.services.json_output_writer import JsonOutputWriter
from backend.services.shutil_file_mover import ShutilFileMover
from backend.services.variant_transcription_service import VariantTranscriptionService
from backend.services.whisper_transcription_service import WhisperTranscriptionService

if TYPE_CHECKING:
    from backend.variants.variant import Variant


class ServiceFactory:
    """Factory for creating service instances."""

    @staticmethod
    def create_transcription_service(
        preprocess_config: PreprocessConfig | None = None,
        preprocess_runner: Callable[[str, PreprocessConfig], PreprocessResult] | None = None,
        transcription_config: TranscriptionConfig | None = None,
        variant: "Variant | None" = None,
        preset: str = "et-large",
        language: str | None = None,
        output_format: str = "txt",
    ) -> TranscriptionService:
        """Create transcription service with default implementations."""
        if variant is not None:
            # Create variant-aware service
            return VariantTranscriptionService(
                variant=variant,
                preset=preset,
                language=language,
                output_format=output_format,
            )

        # Create standard service with defaults
        if preprocess_config is None:
            preprocess_config = PreprocessConfig.from_env()
        if preprocess_runner is None:
            preprocess_runner = preprocess_audio
        if transcription_config is None:
            transcription_config = TranscriptionConfig.from_env()

        return WhisperTranscriptionService(
            preprocess_config=preprocess_config,
            preprocess_runner=preprocess_runner,
            transcription_config=transcription_config,
        )

    @staticmethod
    def create_state_store(db_path: str | Path | None = None) -> StateStore:
        """Create state store with default database."""
        db = TranscriptionDatabase(db_path)
        return DuckDBStateStore(db)

    @staticmethod
    def create_file_mover() -> FileMover:
        """Create file mover with default implementation."""
        return ShutilFileMover()

    @staticmethod
    def create_output_writer() -> OutputWriter:
        """Create output writer with default implementation."""
        return JsonOutputWriter()
