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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.database import TranscriptionDatabase
from backend.processor import TranscriptionProcessor
from backend.variants.registry import get_variant_by_number

# Rich console for structured output
console = Console()

# Setup logging - will be configured based on verbose flag
LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging levels based on verbosity.

    Args:
        verbose: If True, show DEBUG logs. If False, show only WARNING+ for noisy modules.
    """
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

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )


def cmd_process(args: argparse.Namespace) -> int:
    """Process audio files in the specified folder.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Configure logging based on verbosity
    _configure_logging(getattr(args, "verbose", False))

    input_folder = Path(args.input_folder)

    if not input_folder.exists():
        console.print(f"[red]Error:[/red] Input folder does not exist: {input_folder}")
        return 1

    if not input_folder.is_dir():
        console.print(f"[red]Error:[/red] Input path is not a directory: {input_folder}")
        return 1

    # Get variant - default to variant 7 if not specified
    if args.variant is not None:
        variant = get_variant_by_number(args.variant)
        if variant is None:
            console.print(f"[red]Error:[/red] Invalid variant number: {args.variant}")
            return 1
    else:
        # Default to variant 7 (no preprocessing + minimal transcription parameters)
        variant = get_variant_by_number(7)
        if variant is None:
            console.print("[red]Error:[/red] Failed to load default variant 7")
            return 1

    # Display configuration using Rich
    config_table = Table.grid(padding=(0, 2))
    config_table.add_row("[bold]Input folder:[/bold]", str(input_folder))
    config_table.add_row("[bold]Model preset:[/bold]", args.preset)
    config_table.add_row("[bold]Variant:[/bold]", f"{variant.number}: {variant.name}")
    if args.language:
        config_table.add_row("[bold]Language:[/bold]", args.language)
    config_table.add_row("[bold]Output format:[/bold]", args.output_format)

    console.print("\n[bold]Transcription Configuration[/bold]")
    console.print(Panel(config_table, border_style="blue", padding=(0, 1)))
    console.print()

    LOGGER.debug("Starting transcription processing")
    LOGGER.debug("Input folder: %s", input_folder)
    LOGGER.debug("Model preset: %s", args.preset)
    LOGGER.debug("Using variant %d: %s", variant.number, variant.name)

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

            # Display summary using Rich
            console.print()
            summary_table = Table.grid(padding=(0, 2))
            summary_table.add_row("[bold]Files found:[/bold]", str(results.get("files_found", 0)))
            summary_table.add_row(
                "[bold]Successfully processed:[/bold]", f"[green]{results.get('succeeded', 0)}[/green]"
            )
            summary_table.add_row("[bold]Failed:[/bold]", f"[red]{results.get('failed', 0)}[/red]")

            console.print("[bold]Processing Summary[/bold]")
            console.print(Panel(summary_table, border_style="green", padding=(0, 1)))

            run_stats = results.get("run_statistics")
            if run_stats:
                stats_table = Table.grid(padding=(0, 2))
                stats_table.add_row(
                    "[bold]Total time:[/bold]",
                    f"{run_stats.get('total_processing_time', 0):.2f} s",
                )
                stats_table.add_row(
                    "[bold]Preprocessing:[/bold]",
                    f"{run_stats.get('total_preprocess_time', 0):.2f} s",
                )
                stats_table.add_row(
                    "[bold]Transcription:[/bold]",
                    f"{run_stats.get('total_transcribe_time', 0):.2f} s",
                )
                stats_table.add_row(
                    "[bold]Average speed:[/bold]",
                    f"{float(run_stats.get('average_speed_ratio') or 0):.2f}x realtime",
                )
                if run_stats.get("detected_languages"):
                    stats_table.add_row(
                        "[bold]Detected languages:[/bold]",
                        ", ".join(run_stats["detected_languages"]),
                    )

                console.print()
                console.print("[bold]Run Statistics[/bold]")
                console.print(Panel(stats_table, border_style="cyan", padding=(0, 1)))

        return 0

    except Exception as error:
        console.print(f"[red]Processing failed:[/red] {error}")
        if getattr(args, "verbose", False):
            LOGGER.exception("Full error details:")
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
        help=(
            "Variant number to use (1-16). "
            "Default: 7 (no preprocessing + minimal transcription parameters without VAD filter)."
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
