"""CLI commands for transcription processing."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.cli.ui import (
    console,
    display_config_table,
    display_multi_variant_summary,
    display_processing_summary,
    display_run_statistics,
)
from backend.database import TranscriptionDatabase
from backend.processor import TranscriptionProcessor
from backend.variants.registry import get_conservative_sweep_variants, get_variant_by_number
from backend.variants.variant import Variant

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


def _get_variant_overrides(variant: Variant) -> dict[str, Any]:
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

    overrides: dict[str, Any] = {}
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


def _parse_variant_numbers(args: argparse.Namespace) -> tuple[list[int], str | None]:
    """Parse variant numbers from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (variant_numbers list, error_message or None)
    """
    variant_numbers: list[int] = []
    if args.variant_sweep == "conservative":
        variant_numbers = get_conservative_sweep_variants()
    elif args.variants:
        try:
            variant_numbers = [int(v.strip()) for v in args.variants.split(",")]
        except ValueError:
            return [], f"Invalid variant list format: {args.variants}"
    elif args.variant is not None:
        variant_numbers = [args.variant]
    else:
        # Default to variant 1
        variant_numbers = [1]

    return variant_numbers, None


def _validate_variants(variant_numbers: list[int]) -> tuple[list[Any], str | None]:  # noqa: ANN401
    """Validate that all variant numbers exist.

    Args:
        variant_numbers: List of variant numbers to validate

    Returns:
        Tuple of (variants list, error_message or None)
    """
    variants: list[Variant] = []
    for num in variant_numbers:
        variant = get_variant_by_number(num)
        if variant is None:
            return [], f"Invalid variant number: {num}"
        variants.append(variant)
    return variants, None


def _process_single_variant(
    args: argparse.Namespace,
    input_folder: Path,
    variant: Any,  # noqa: ANN401
) -> int:
    """Process files with a single variant.

    Args:
        args: Parsed command-line arguments
        input_folder: Path to input folder
        variant: Variant instance

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Display configuration using Rich
    display_config_table(
        str(input_folder),
        args.preset,
        variant.number,
        variant.name,
        args.language,
        args.output_format,
    )

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
            display_processing_summary(results)

            run_stats = results.get("run_statistics")
            if run_stats:
                display_run_statistics(run_stats)

        return 0

    except Exception as error:
        console.print(f"[red]Processing failed:[/red] {error}")
        if getattr(args, "verbose", False):
            LOGGER.exception("Full error details:")
        return 1


def _process_multi_variant(
    args: argparse.Namespace,
    input_folder: Path,
    variants: list[Any],  # noqa: ANN401
) -> int:
    """Process files with multiple variants.

    Args:
        args: Parsed command-line arguments
        input_folder: Path to input folder
        variants: List of variant instances

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    console.print(f"\n[bold]Running {len(variants)} variants[/bold]")
    if args.variant_sweep == "conservative":
        console.print("[dim]Using conservative sweep preset[/dim]")

    git_commit = _get_git_commit_hash()
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.isoformat()
    # Create timestamped folder name (filesystem-safe)
    timestamp_folder = timestamp.strftime("%Y-%m-%dT%H-%M-%S")

    # Create single timestamped output directory directly under input folder
    run_folder = input_folder / timestamp_folder
    run_folder.mkdir(exist_ok=True)

    # Collect all variant metadata
    all_variant_metadata: list[dict[str, Any]] = []
    all_results: dict[int, dict[str, Any]] = {}

    for variant in variants:
        overrides = _get_variant_overrides(variant)
        console.print(
            f"\n[cyan]Running variant {variant.number}: {variant.name}[/cyan] "
            f"(overrides: {overrides if overrides else 'none'})"
        )

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

                # Use single run folder for all outputs
                # Files will be prefixed with variant number in the processor
                processor.processed_folder = run_folder
                processor.failed_folder = run_folder / "failed"
                # Set output base dir and store variant info for filename prefixing
                processor._output_base_dir = run_folder  # type: ignore[reportPrivateUsage]
                processor._variant_number = variant.number  # type: ignore[reportPrivateUsage]
                processor._variant_name = variant.name  # type: ignore[reportPrivateUsage]
                processor.processed_folder.mkdir(exist_ok=True, parents=True)
                processor.failed_folder.mkdir(exist_ok=True, parents=True)

                results = processor.process_folder()
                all_results[variant.number] = results

                # Find JSON file(s) created for this variant
                # Pattern: variant_{number:03d}_{name}_*.json
                json_files = list(run_folder.glob(f"variant_{variant.number:03d}_{variant.name}_*.json"))
                json_filename = json_files[0].name if json_files else None

                # Collect variant metadata
                transcription_config = variant.transcription_config
                variant_meta = {
                    "variant_number": variant.number,
                    "variant_name": variant.name,
                    "json_filename": json_filename,  # Store exact filename for easy lookup
                    "transcription_config": {
                        "beam_size": getattr(transcription_config, "beam_size", None),
                        "chunk_length": getattr(transcription_config, "chunk_length", None),
                        "no_speech_threshold": getattr(transcription_config, "no_speech_threshold", None),
                        "logprob_threshold": getattr(transcription_config, "logprob_threshold", None),
                        "vad_filter": getattr(transcription_config, "vad_filter", None),
                        "condition_on_previous_text": getattr(transcription_config, "condition_on_previous_text", None),
                    },
                    "results": {
                        "succeeded": results.get("succeeded", 0),
                        "failed": results.get("failed", 0),
                    },
                }
                all_variant_metadata.append(variant_meta)

                console.print(
                    f"[green]✓[/green] Variant {variant.number} completed: "
                    f"{results.get('succeeded', 0)} succeeded, {results.get('failed', 0)} failed"
                )

        except Exception as error:
            console.print(f"[red]✗[/red] Variant {variant.number} failed: {error}")
            if getattr(args, "verbose", False):
                LOGGER.exception("Full error details:")
            all_results[variant.number] = {"succeeded": 0, "failed": 0, "error": str(error)}
            # Add failed variant to metadata
            all_variant_metadata.append(
                {
                    "variant_number": variant.number,
                    "variant_name": variant.name,
                    "error": str(error),
                    "results": {"succeeded": 0, "failed": 0},
                }
            )

    # Write single run_meta.json with all variants
    run_meta = {
        "timestamp": timestamp_str,
        "git_commit_hash": git_commit,
        "preset": args.preset,
        "language": args.language,
        "output_format": args.output_format,
        "variants": all_variant_metadata,
        "summary": {
            "total_variants": len(variants),
            "total_succeeded": sum(int(r.get("succeeded", 0)) for r in all_results.values()),
            "total_failed": sum(int(r.get("failed", 0)) for r in all_results.values()),
        },
    }

    meta_path = run_folder / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # Display overall summary
    total_succeeded = sum(int(r.get("succeeded", 0)) for r in all_results.values())
    total_failed = sum(int(r.get("failed", 0)) for r in all_results.values())
    display_multi_variant_summary(len(variants), total_succeeded, total_failed, str(run_folder))

    return 0 if total_failed == 0 else 1


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

    # Parse and validate variants
    variant_numbers, error = _parse_variant_numbers(args)
    if error:
        console.print(f"[red]Error:[/red] {error}")
        return 1

    variants, error = _validate_variants(variant_numbers)
    if error:
        console.print(f"[red]Error:[/red] {error}")
        return 1

    # If single variant, use original behavior (backward compatible)
    if len(variants) == 1:
        return _process_single_variant(args, input_folder, variants[0])
    else:
        # Multi-variant mode
        return _process_multi_variant(args, input_folder, variants)
