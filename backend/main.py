"""Minimal keepalive entrypoint for dev/test Docker container.

Purpose:
    Keep the Docker app container running during development and testing.
    This is NOT used in production - end users should use the production
    Dockerfile at the project root.

Usage:
    - Development: `docker compose -f docker/docker-compose.yml up`
    - Testing: `make docker-unit` (runs tests inside container)
    - Healthcheck: `python -m backend.main --healthcheck`

This module does not start any services; it simply blocks until it
receives a termination signal (SIGINT/SIGTERM).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import NoReturn
from types import FrameType


def _setup_logging() -> None:
    # Log to stdout with a concise format
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def main() -> NoReturn:
    """Block indefinitely until SIGINT/SIGTERM is received."""
    _setup_logging()
    log = logging.getLogger("backend.main")

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        names = {getattr(signal, n): n for n in dir(signal) if n.startswith("SIG")}
        log.info("Received signal %s (%s); shutting down.", signum, names.get(signum, "?"))
        stop_event.set()

    # Register graceful shutdown handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    log.info("backend.main started — waiting for signals (PID %s).", str(getattr(sys, "pid", "?")))

    # Park the thread; wake on signal
    # Use a long timeout so Event.wait() can be interrupted by signals.
    while not stop_event.is_set():
        stop_event.wait(timeout=24 * 60 * 60)  # 24 hours

    log.info("backend.main stopped.")
    # Exit explicitly with 0 so Docker reports clean shutdown
    raise SystemExit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backend keepalive and utilities")
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run a quick healthcheck and exit",
    )
    args = parser.parse_args()

    if args.healthcheck:
        _setup_logging()
        logging.getLogger("backend.main").info("Healthcheck OK")
        raise SystemExit(0)

    main()
