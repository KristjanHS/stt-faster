"""Unit tests for backend.main module."""

from __future__ import annotations

import logging
import signal

import pytest


class FakeEvent:
    def __init__(self, initially_set: bool = False) -> None:
        self._is_set = initially_set
        self.wait_calls: list[float | int | None] = []
        self.set_called = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self.set_called = True
        self._is_set = True

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)
        # Avoid blocking in tests; mark set so the loop exits.
        self._is_set = True


class RecordingSignalRegistrar:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    def __call__(self, signum: int, handler: object) -> None:
        self.calls.append((signum, handler))


@pytest.fixture()
def fake_event() -> FakeEvent:
    return FakeEvent()


@pytest.fixture()
def fake_event_factory(fake_event: FakeEvent):
    return lambda: fake_event


@pytest.fixture()
def recording_signal_registrar() -> RecordingSignalRegistrar:
    return RecordingSignalRegistrar()


class TestMainFunction:
    """Tests for main function."""

    def test_registers_signal_handlers_and_blocks(
        self,
        fake_event: FakeEvent,
        fake_event_factory,
        recording_signal_registrar: RecordingSignalRegistrar,
    ) -> None:
        """Test that main registers SIGINT and SIGTERM handlers and blocks until signaled."""
        with pytest.raises(SystemExit) as exc_info:
            from backend.main import main

            main(event_factory=fake_event_factory, signal_registrar=recording_signal_registrar)

        # Verify both SIGINT and SIGTERM are registered
        registered_signals = {sig for sig, _ in recording_signal_registrar.calls}
        assert signal.SIGINT in registered_signals
        assert signal.SIGTERM in registered_signals

        # Verify clean exit
        assert exc_info.value.code == 0

        # Verify wait is called with 24-hour timeout
        assert fake_event.wait_calls == [24 * 60 * 60]

    def test_signal_handler_triggers_shutdown(
        self,
        fake_event: FakeEvent,
        fake_event_factory,
        recording_signal_registrar: RecordingSignalRegistrar,
    ) -> None:
        """Test that signal handler sets stop event."""
        with pytest.raises(SystemExit):
            from backend.main import main

            main(event_factory=fake_event_factory, signal_registrar=recording_signal_registrar)

        # Verify handler was captured and sets stop event when called
        handlers = dict(recording_signal_registrar.calls)
        captured_handler = handlers[signal.SIGINT]
        fake_event.set_called = False
        captured_handler(signal.SIGINT, None)
        assert fake_event.set_called


class TestHealthcheckMode:
    """Tests for healthcheck command-line option."""

    def test_healthcheck_exits_successfully(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that --healthcheck mode exits with 0 and logs success."""
        from backend.main import _run_healthcheck

        with caplog.at_level(logging.INFO):
            with pytest.raises(SystemExit) as exc_info:
                _run_healthcheck()

        assert exc_info.value.code == 0
        assert any("Healthcheck OK" in message for message in caplog.messages)
