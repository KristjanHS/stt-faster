"""Unit tests for backend.main module."""

from __future__ import annotations

import signal
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest


class TestMainFunction:
    """Tests for main function."""

    @patch("backend.main.signal.signal")
    @patch("backend.main.threading.Event")
    def test_registers_signal_handlers_and_blocks(self, mock_event_class: Mock, mock_signal_signal: Mock) -> None:
        """Test that main registers SIGINT and SIGTERM handlers and blocks until signaled."""
        mock_event = Mock()
        # Make is_set return False once (to enter loop), then True (to exit)
        mock_event.is_set.side_effect = [False, True]
        mock_event_class.return_value = mock_event

        with pytest.raises(SystemExit) as exc_info:
            from backend.main import main

            main()

        # Verify both SIGINT and SIGTERM are registered
        assert mock_signal_signal.call_count == 2
        signal_calls = [call_obj[0][0] for call_obj in mock_signal_signal.call_args_list]
        assert signal.SIGINT in signal_calls
        assert signal.SIGTERM in signal_calls

        # Verify clean exit
        assert exc_info.value.code == 0

        # Verify wait is called with 24-hour timeout
        assert mock_event.wait.called
        assert mock_event.wait.call_args[1]["timeout"] == 24 * 60 * 60

    @patch("backend.main.signal.signal")
    @patch("backend.main.threading.Event")
    def test_signal_handler_triggers_shutdown(self, mock_event_class: Mock, mock_signal_signal: Mock) -> None:
        """Test that signal handler sets stop event."""
        mock_event = Mock()
        mock_event_class.return_value = mock_event

        # Capture the signal handler
        captured_handler = None

        def capture_signal_handler(sig: int, handler) -> None:  # type: ignore[no-untyped-def]
            nonlocal captured_handler
            if sig == signal.SIGINT:
                captured_handler = handler

        mock_signal_signal.side_effect = capture_signal_handler

        # Set up event to return True immediately to exit the loop
        mock_event.is_set.return_value = True

        with pytest.raises(SystemExit):
            from backend.main import main

            main()

        # Verify handler was captured and sets stop event when called
        assert captured_handler is not None
        mock_event.reset_mock()
        captured_handler(signal.SIGINT, None)
        mock_event.set.assert_called_once()


class TestHealthcheckMode:
    """Tests for healthcheck command-line option."""

    def test_healthcheck_exits_successfully(self) -> None:
        """Test that --healthcheck mode exits with 0 and logs success."""
        # Run as subprocess to test actual CLI behavior
        result = subprocess.run(
            [sys.executable, "-m", "backend.main", "--healthcheck"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0
        assert "Healthcheck OK" in result.stdout or "Healthcheck OK" in result.stderr
