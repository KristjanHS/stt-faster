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

    # Process folder with conservative sweep preset
    .venv/bin/python scripts/transcribe_manager.py process /path/to/audio \\
        --preset et-32 --language et --output-format both \\
        --variant-sweep conservative

    # Check status
    .venv/bin/python scripts/transcribe_manager.py status
"""

import argparse
import json
import logging
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _get_git_commit_hash() -> str | None:
    """Get the current git commit hash if available.

    Returns:
        Git commit hash as string, or None if not available
    """
    try:
        # Find git executable using shutil.which for security
        git_path = shutil.which("git")
        if not git_path:
            return None

        result = subprocess.run(  # nosec B603 - git path validated via shutil.which
            [git_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_variant_overrides(variant) -> dict[str, Any]:
    """Extract overridden transcription config fields from variant.

    Args:
        variant: Variant instance

    Returns:
        Dictionary of overridden fields (only fields that differ from defaults)
    """
    from backend.variants.transcription_presets import create_minimal_config  # noqa: PLC0415

    # Get baseline minimal config (defaults)
    baseline = create_minimal_config()
    config = variant.transcription_config

    overrides = {}
    # Check which fields are overridden
    for field in ["beam_size", "chunk_length", "no_speech_threshold", "condition_on_previous_text"]:
        baseline_val = getattr(baseline, field, None)
        config_val = getattr(config, field, None)
        if config_val is not None and config_val != baseline_val:
            overrides[field] = config_val

    return overrides


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

    # Determine which variants to run
    variant_numbers: list[int] = []
    if args.variant_sweep == "conservative":
        variant_numbers = [1, 6, 21, 22, 23, 24, 25, 26]
    elif args.variants:
        try:
            variant_numbers = [int(v.strip()) for v in args.variants.split(",")]
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid variant list format: {args.variants}")
            return 1
    elif args.variant is not None:
        variant_numbers = [args.variant]
    else:
        # Default to variant 1
        variant_numbers = [1]

    # Validate all variants exist
    variants = []
    for num in variant_numbers:
        variant = get_variant_by_number(num)
        if variant is None:
            console.print(f"[red]Error:[/red] Invalid variant number: {num}")
            return 1
        variants.append(variant)

    # If single variant, use original behavior (backward compatible)
    if len(variants) == 1:
        variant = variants[0]
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
    else:
        # Multi-variant mode
        console.print(f"\n[bold]Running {len(variants)} variants[/bold]")
        if args.variant_sweep == "conservative":
            console.print("[dim]Using conservative sweep preset[/dim]")

        git_commit = _get_git_commit_hash()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create output root directory
        output_root = input_folder / "variant_outputs"
        output_root.mkdir(exist_ok=True)

        all_results = {}
        for variant in variants:
            overrides = _get_variant_overrides(variant)
            console.print(
                f"\n[cyan]Running variant {variant.number}: {variant.name}[/cyan] "
                f"(overrides: {overrides if overrides else 'none'})"
            )

            # Create variant-specific output folder
            variant_folder = output_root / f"variant_{variant.number:03d}_{variant.name}"
            variant_folder.mkdir(exist_ok=True)

            # Create a temporary input folder for this variant (copy structure)
            # We'll process files and move outputs to variant folder
            try:
                with TranscriptionDatabase(args.db_path) as db:
                    # Create processor with variant-specific output handling
                    # Disable file moving so all variants can process the same input files
                    processor = TranscriptionProcessor(
                        db,
                        input_folder,
                        preset=args.preset,
                        language=args.language,
                        output_format=args.output_format,
                        variant=variant,
                        disable_file_moving=True,
                    )

                    # Override the processed_folder to be variant-specific
                    processor.processed_folder = variant_folder / "processed"
                    processor.failed_folder = variant_folder / "failed"
                    processor._output_base_dir = variant_folder / "processed"  # Write outputs here
                    processor.processed_folder.mkdir(exist_ok=True, parents=True)
                    processor.failed_folder.mkdir(exist_ok=True, parents=True)

                    results = processor.process_folder()
                    all_results[variant.number] = results

                    # Write run_meta.json
                    transcription_config = variant.transcription_config
                    run_meta = {
                        "variant_number": variant.number,
                        "variant_name": variant.name,
                        "transcription_config": {
                            "beam_size": getattr(transcription_config, "beam_size", None),
                            "chunk_length": getattr(transcription_config, "chunk_length", None),
                            "no_speech_threshold": getattr(transcription_config, "no_speech_threshold", None),
                            "condition_on_previous_text": getattr(
                                transcription_config, "condition_on_previous_text", None
                            ),
                        },
                        "preset": args.preset,
                        "language": args.language,
                        "output_format": args.output_format,
                        "git_commit_hash": git_commit,
                        "timestamp": timestamp,
                    }

                    meta_path = variant_folder / "run_meta.json"
                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(run_meta, f, indent=2, ensure_ascii=False)

                    console.print(
                        f"[green]✓[/green] Variant {variant.number} completed: "
                        f"{results.get('succeeded', 0)} succeeded, {results.get('failed', 0)} failed"
                    )

            except Exception as error:
                console.print(f"[red]✗[/red] Variant {variant.number} failed: {error}")
                if getattr(args, "verbose", False):
                    LOGGER.exception("Full error details:")
                all_results[variant.number] = {"succeeded": 0, "failed": 0, "error": str(error)}

        # Display overall summary
        console.print("\n[bold]Overall Summary[/bold]")
        summary_table = Table.grid(padding=(0, 2))
        total_succeeded = sum(r.get("succeeded", 0) for r in all_results.values())
        total_failed = sum(r.get("failed", 0) for r in all_results.values())
        summary_table.add_row("[bold]Total variants:[/bold]", str(len(variants)))
        summary_table.add_row("[bold]Total succeeded:[/bold]", f"[green]{total_succeeded}[/green]")
        summary_table.add_row("[bold]Total failed:[/bold]", f"[red]{total_failed}[/red]")
        summary_table.add_row("[bold]Output directory:[/bold]", str(output_root))
        console.print(Panel(summary_table, border_style="green", padding=(0, 1)))

        return 0 if total_failed == 0 else 1


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
            "Mutually exclusive with --variants and --variant-sweep."
        ),
    )
    variant_group.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated list of variant numbers to run (e.g., '1,6,21,22'). "
            "Each variant will be processed sequentially with outputs in separate subfolders. "
            "Mutually exclusive with --variant and --variant-sweep."
        ),
    )
    variant_group.add_argument(
        "--variant-sweep",
        type=str,
        choices=["conservative"],
        default=None,
        help=(
            "Run a predefined variant sweep. "
            "'conservative' runs variants 1,6,21,22,23,24,25,26. "
            "Mutually exclusive with --variant and --variants."
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
