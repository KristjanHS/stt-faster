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

from backend.database import TranscriptionDatabase
from backend.processor import TranscriptionProcessor
from backend.variants.registry import get_variant_by_number

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

    # Get variant if specified
    variant = None
    if args.variant is not None:
        variant = get_variant_by_number(args.variant)
        if variant is None:
            LOGGER.error("Invalid variant number: %d", args.variant)
            return 1
        LOGGER.info("Using variant %d: %s", variant.number, variant.name)

    try:
        with TranscriptionDatabase(args.db_path) as db:
            processor = TranscriptionProcessor(
                db,
                input_folder,
                preset=args.preset,
                language=args.language,
                output_format=args.output_format,
                variant=variant,
            )
            results = processor.process_folder()

            LOGGER.info("=" * 60)
            LOGGER.info("Processing Summary:")
            LOGGER.info("  Files found: %d", results.get("files_found", 0))
            LOGGER.info("  Successfully processed: %d", results.get("succeeded", 0))
            LOGGER.info("  Failed: %d", results.get("failed", 0))
            LOGGER.info("=" * 60)

            run_stats = results.get("run_statistics")
            if run_stats:
                LOGGER.info(
                    "Run stats: %.2f s total, %.2f s preprocess, %.2f s transcription, avg %.2f x speed",
                    run_stats.get("total_processing_time", 0),
                    run_stats.get("total_preprocess_time", 0),
                    run_stats.get("total_transcribe_time", 0),
                    float(run_stats.get("average_speed_ratio") or 0),
                )
                if run_stats.get("detected_languages"):
                    LOGGER.info("Detected languages: %s", run_stats["detected_languages"])

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
    process_parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Variant number to use (1-16). Variant 7 uses minimal transcription parameters without VAD filter.",
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
