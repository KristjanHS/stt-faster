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

    # Check status
    .venv/bin/python scripts/transcribe_manager.py status
"""

import argparse
import logging
import sys
from pathlib import Path

from backend.config import setup_logging
from backend.cli.transcription_commands import cmd_process
from backend.database import TranscriptionDatabase

# Setup logging - will be configured based on verbose flag
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
        logging.getLogger("backend.database").setLevel(logging.WARNING)
        logging.getLogger("backend.model_loader").setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        # Keep important progress loggers at INFO
        logging.getLogger("backend.variants.executor").setLevel(logging.INFO)
        logging.getLogger("backend.variants.preprocess_steps").setLevel(logging.INFO)

    # Configure root logger with CLI-specific format
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )


def cmd_status(args: argparse.Namespace) -> int:
    """Display status of all tracked files.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Configure logging based on verbosity
    _configure_logging(getattr(args, "verbose", False))

    try:
        with TranscriptionDatabase(args.db_path) as db:
            summary = db.get_summary()
            all_files = db.get_all_files()

            if not all_files:
                LOGGER.info("No files tracked in database")
                return 0

            LOGGER.info("=" * 60)
            LOGGER.info("Transcription Status Summary")
            LOGGER.info("=" * 60)
            LOGGER.info("Total files: %d", len(all_files))

            for status, count in summary.items():
                LOGGER.info("  %s: %d", status.upper(), count)

            LOGGER.info("=" * 60)

            # Show detailed list if requested
            if args.verbose:
                LOGGER.info("\nDetailed File List:")
                for file_info in all_files:
                    status_marker = (
                        "✓" if file_info["status"] == "completed" else "✗" if file_info["status"] == "failed" else "⏳"
                    )
                    LOGGER.info(
                        "  %s [%s] %s",
                        status_marker,
                        file_info["status"].upper(),
                        Path(file_info["file_path"]).name,
                    )
                    if file_info.get("error_message"):
                        LOGGER.info("      Error: %s", file_info["error_message"])

        return 0

    except Exception as error:
        LOGGER.error("Status check failed: %s", error, exc_info=True)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Manage audio transcription automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database (default: XDG data home, typically ~/.local/share/stt-faster/)",
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

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of tracked files",
    )
    status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed file list",
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
    elif args.command == "status":
        return cmd_status(args)
    else:
        LOGGER.error("Unknown command: %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
