#!/usr/bin/env python3
"""CLI tool for managing audio transcription automation.

Usage:
    # Process folder
    .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe

    # Check status
    .venv/bin/python scripts/transcribe_manager.py status
"""

import argparse
import logging
import sys
from pathlib import Path

from transcription.database import TranscriptionDatabase
from transcription.processor import TranscriptionProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def cmd_process(args: argparse.Namespace) -> int:
    """Process audio files in the specified folder.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    input_folder = Path(args.input_folder)

    if not input_folder.exists():
        LOGGER.error("Input folder does not exist: %s", input_folder)
        return 1

    if not input_folder.is_dir():
        LOGGER.error("Input path is not a directory: %s", input_folder)
        return 1

    LOGGER.info("Starting transcription processing")
    LOGGER.info("Input folder: %s", input_folder)
    LOGGER.info("Model preset: %s", args.preset)

    try:
        with TranscriptionDatabase(args.db_path) as db:
            processor = TranscriptionProcessor(db, input_folder, preset=args.preset)
            results = processor.process_folder()

            LOGGER.info("=" * 60)
            LOGGER.info("Processing Summary:")
            LOGGER.info("  Files found: %d", results.get("files_found", 0))
            LOGGER.info("  Successfully processed: %d", results.get("succeeded", 0))
            LOGGER.info("  Failed: %d", results.get("failed", 0))
            LOGGER.info("=" * 60)

        return 0

    except Exception as error:
        LOGGER.error("Processing failed: %s", error, exc_info=True)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Display status of all tracked files.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
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


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Manage audio transcription automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        default="transcribe_state.db",
        help="Path to SQLite database (default: transcribe_state.db)",
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
        default="distil",
        choices=["turbo", "distil", "large8gb"],
        help="Model preset to use (default: distil)",
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
