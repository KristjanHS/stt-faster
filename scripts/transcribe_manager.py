#!/usr/bin/env python3
"""CLI tool for managing audio transcription automation.

Usage:
    # Process folder with single variant
    .venv/bin/python scripts/transcribe_manager.py process /path/to/audio \\
        --preset et-32 --language et --output-format both --variant 1

    # Process folder with multiple variants (explicit list)
    .venv/bin/python scripts/transcribe_manager.py process /path/to/audio \\
        --preset et-32 --language et --output-format both \\
        --variants "1,6,21,22,23,24,25,26"
"""

import argparse
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from backend.config import setup_logging
from backend.cli.transcription_commands import cmd_process

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging levels based on verbosity.

    Args:
        verbose: If True, show DEBUG logs. If False, show only WARNING+ for noisy modules.
    """
    # First, setup base logging from config
    setup_logging()

    # Then apply CLI-specific overrides
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
        # Reduce noise from internal modules in non-verbose mode
        logging.getLogger("backend.processor").setLevel(logging.WARNING)
        logging.getLogger("backend.model_loader").setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        # Keep important progress loggers at INFO
        logging.getLogger("backend.variants.executor").setLevel(logging.INFO)
        logging.getLogger("backend.variants.preprocess_steps").setLevel(logging.INFO)

    # Update root logger level without resetting handlers
    root = logging.getLogger()
    root.setLevel(level)

    # Ensure console handler uses message-only format (file handler keeps full format)
    for handler in root.handlers:
        if not isinstance(handler, TimedRotatingFileHandler):
            # This is a console handler - ensure it uses message-only format
            handler.setFormatter(logging.Formatter("%(message)s"))


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Manage audio transcription automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process audio files in a folder",
    )
    process_parser.add_argument(
        "input_folder",
        help="Path to folder containing audio files",
    )
    process_parser.add_argument(
        "--preset",
        default="et-large",
        choices=["turbo", "distil", "large8gb", "et-large", "et-32", "small"],
        help="Model preset to use (default: et-large)",
    )
    process_parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g., 'en', 'et', 'ru'). Auto-detect if not specified.",
    )
    process_parser.add_argument(
        "--output-format",
        default="txt",
        choices=["txt", "json", "both"],
        help="Output format for transcripts (default: txt)",
    )
    process_parser.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=True,
        help="Run pyannote speaker diarization (default: enabled). Requires HF_TOKEN.",
    )
    process_parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Disable diarization (omit speaker labels from output).",
    )
    process_parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers to diarize (default: 2).",
    )
    variant_group = process_parser.add_mutually_exclusive_group()
    variant_group.add_argument(
        "--variant",
        type=int,
        default=None,
        help=(
            "Variant number to use (1-26). "
            "Default: 1 (no preprocessing + minimal transcription parameters). "
            "Mutually exclusive with --variants."
        ),
    )
    variant_group.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated list of variant numbers to run (e.g., '1,6,21,22'). "
            "Each variant will be processed sequentially with outputs in separate subfolders. "
            "Mutually exclusive with --variant."
        ),
    )
    process_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed debug logs (default: show only important progress)",
    )

    return parser


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "process":
        return cmd_process(args)
    LOGGER.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
