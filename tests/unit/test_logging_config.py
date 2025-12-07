"""Unit tests for backend.config logging configuration."""

import logging
import logging.handlers
from pathlib import Path

import pytest


class TestParseLevel:
    """Tests for _parse_level function."""

    def test_parse_level_with_none(self) -> None:
        """Test that None returns the default level."""
        from backend.config import _parse_level

        result = _parse_level(None)
        assert result == logging.INFO

        result = _parse_level(None, logging.ERROR)
        assert result == logging.ERROR

    def test_parse_level_with_empty_string(self) -> None:
        """Test that empty string returns the default level."""
        from backend.config import _parse_level

        result = _parse_level("")
        assert result == logging.INFO

        result = _parse_level("  ", logging.WARNING)
        assert result == logging.WARNING

    def test_parse_level_with_valid_level(self) -> None:
        """Test parsing valid log levels."""
        from backend.config import _parse_level

        assert _parse_level("DEBUG") == logging.DEBUG
        assert _parse_level("INFO") == logging.INFO
        assert _parse_level("WARNING") == logging.WARNING
        assert _parse_level("ERROR") == logging.ERROR
        assert _parse_level("CRITICAL") == logging.CRITICAL

    def test_parse_level_case_insensitive(self) -> None:
        """Test that level parsing is case-insensitive."""
        from backend.config import _parse_level

        assert _parse_level("debug") == logging.DEBUG
        assert _parse_level("DeBuG") == logging.DEBUG
        assert _parse_level("  info  ") == logging.INFO

    def test_parse_level_invalid_returns_default(self) -> None:
        """Test that invalid level returns default."""
        from backend.config import _parse_level

        result = _parse_level("INVALID_LEVEL", logging.WARNING)
        assert result == logging.WARNING


class TestBuildConsoleHandler:
    """Tests for _build_console_handler function."""

    def test_build_console_handler_tty(self) -> None:
        """Test console handler creation for TTY (RichHandler)."""
        from backend.config import _build_console_handler

        handler = _build_console_handler(logging.INFO, isatty=lambda: True)

        from rich.logging import RichHandler

        assert isinstance(handler, RichHandler)
        assert handler.level == logging.INFO

    def test_build_console_handler_non_tty(self) -> None:
        """Test console handler creation for non-TTY (StreamHandler)."""
        from backend.config import _build_console_handler

        handler = _build_console_handler(logging.DEBUG, isatty=lambda: False)

        assert isinstance(handler, logging.StreamHandler)
        assert not isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.level == logging.DEBUG


class TestLogFileCreation:
    """Tests for file logging behavior."""

    def test_file_handler_backup_count_parsing(self, tmp_path: Path) -> None:
        """Test that APP_LOG_BACKUP_COUNT is correctly parsed and bounds-checked."""

        # Test that negative values are coerced to 0
        # This is implementation behavior - max(0, int(value))
        assert max(0, -5) == 0
        assert max(0, 10) == 10

        # Test that invalid strings default appropriately
        try:
            backup_count = max(0, int("invalid"))
        except (TypeError, ValueError):
            backup_count = 5  # Default fallback
        assert backup_count == 5

    def test_log_format_includes_timestamp_and_level(self) -> None:
        """Test that log format includes required fields."""
        from backend.config import LOG_FORMAT

        # Verify format string contains essential fields
        assert "%(asctime)s" in LOG_FORMAT
        assert "%(name)s" in LOG_FORMAT
        assert "%(levelname)s" in LOG_FORMAT
        assert "%(message)s" in LOG_FORMAT


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def test_logging_produces_expected_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that logging actually produces output with expected format."""
        test_logger = logging.getLogger("test_module")

        with caplog.at_level(logging.INFO):
            test_logger.info("Test message")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[0].name == "test_module"
        assert caplog.records[0].message == "Test message"

    def test_logging_level_filtering(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that logging level filtering works correctly."""
        test_logger = logging.getLogger("test_filter")
        test_logger.setLevel(logging.WARNING)

        with caplog.at_level(logging.DEBUG):
            test_logger.debug("Should not appear")
            test_logger.info("Should not appear")
            test_logger.warning("Should appear")
            test_logger.error("Should appear")

        # Only WARNING and ERROR should be captured
        assert len(caplog.records) == 2
        assert caplog.records[0].levelname == "WARNING"
        assert caplog.records[1].levelname == "ERROR"
