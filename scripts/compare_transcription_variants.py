#!/usr/bin/env python3
"""Compare different transcription pipeline configurations on the same file.

This script runs 16 different transcription variants using the new declarative variant system.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.variants import execute_variant, get_builtin_variants

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare different transcription pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file to transcribe (Windows paths like C:\\Users\\... are supported)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="et-large",
        help="Model preset to use (default: et-large). All variants will use this preset.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'et', 'en'). If not specified, will auto-detect based on preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Directory to save outputs (optional, defaults to Alt_[datetime]/ relative to input file)"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save comparison results as JSON (optional, defaults to {input_stem}_comparison_{datetime}.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--intermcopy",
        action="store_true",
        default=True,
        help="Copy intermediate preprocessing WAV files to output directory",
    )
    parser.add_argument(
        "--skip-variants",
        type=int,
        nargs="+",
        default=[],
        help="Variant numbers to skip (e.g., --skip-variants 1 2 3)",
    )

    args = parser.parse_args()

    # Setup logging first, before any log messages
    # Default to INFO level if --verbose is not used
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Suppress third-party library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Use format without logger names (cleaner output)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(message)s",
    )

    # Ensure Estonian model, GPU, and 16-bit are used for all variants
    # Set STT_DEVICE environment variable to force GPU with float16 (16-bit)
    if "STT_DEVICE" not in os.environ:
        os.environ["STT_DEVICE"] = "cuda/float16"
        LOGGER.info("Set STT_DEVICE=cuda/float16 to ensure GPU and 16-bit precision")

    # Ensure et-large preset is used (Estonian model)
    if args.preset != "et-large":
        LOGGER.warning(
            "Preset is set to '%s', but Estonian model (et-large) is recommended. Continuing with '%s'...",
            args.preset,
            args.preset,
        )
    else:
        LOGGER.info("Using Estonian model preset: et-large (GPU, 16-bit float16)")

    # Normalize Windows paths - convert Windows paths to WSL paths if needed
    audio_file_path = args.audio_file
    # Convert Windows paths (C:\Users\...) to WSL paths (/mnt/c/Users/...)
    if audio_file_path.startswith("C:\\") or audio_file_path.startswith("C:/"):
        # Replace C:\ or C:/ with /mnt/c/
        audio_file_path = audio_file_path.replace("C:\\", "/mnt/c/").replace("C:/", "/mnt/c/")
        # Replace backslashes with forward slashes
        audio_file_path = audio_file_path.replace("\\", "/")
        LOGGER.info("Converted Windows path to WSL path: %s", audio_file_path)
    elif audio_file_path.startswith(("D:\\", "D:/", "E:\\", "E:/")):
        # Handle other drives (D:, E:, etc.)
        drive_letter = audio_file_path[0].lower()
        audio_file_path = audio_file_path.replace(f"{drive_letter}:\\", f"/mnt/{drive_letter}/").replace(
            f"{drive_letter}:/", f"/mnt/{drive_letter}/"
        )
        audio_file_path = audio_file_path.replace("\\", "/")
        LOGGER.info("Converted Windows path to WSL path: %s", audio_file_path)

    audio_path = Path(audio_file_path).resolve()
    if not audio_path.exists():
        LOGGER.error("Audio file not found: %s", audio_path)
        LOGGER.error("Original path: %s", args.audio_file)
        LOGGER.error("If using Windows path, ensure WSL can access it via /mnt/c/...")
        return 1

    # Generate a single datetime suffix for all variants (they run in sequence)
    datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory: Alt_[datetime]/ relative to input file location
    # If --output-dir is specified, use it; otherwise create Alt_[datetime]/ folder
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Using specified output directory: %s", output_dir)
    else:
        output_dir_name = f"Alt_{datetime_suffix}"
        output_dir = audio_path.parent / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Created output directory: %s", output_dir)

    # Generate base output path from input file name
    audio_stem = audio_path.stem
    output_base_path = output_dir / audio_stem

    # Determine comparison summary JSON path
    if args.output_json:
        comparison_json_path = Path(args.output_json).resolve()
    else:
        # Default: {input_stem}_comparison_{datetime}.json in the Alt_[datetime]/ folder
        comparison_json_path = output_dir / f"{audio_stem}_comparison_{datetime_suffix}.json"

    # Run all variants
    results: list[dict[str, Any]] = []
    summary_variants: list[tuple[str, int]] = []

    # Use new variant system
    builtin_variants = get_builtin_variants()
    # Filter out skipped variants
    if args.skip_variants:
        skip_set = set(args.skip_variants)
        builtin_variants = [v for v in builtin_variants if v.number not in skip_set]
        if skip_set:
            LOGGER.info("Skipping variants: %s", sorted(skip_set))
    summary_variants = [(v.name, v.number) for v in builtin_variants]
    for variant in builtin_variants:
        result = execute_variant(
            variant=variant,
            audio_path=str(audio_path),
            preset=args.preset,
            language=args.language,
            output_dir=output_dir,
            output_base_path=output_base_path,
            datetime_suffix=datetime_suffix,
            copy_intermediate=args.intermcopy,
        )
        results.append(result)

        # Show statistics after each variant in table-like format
        LOGGER.info("")
        LOGGER.info("-" * 80)
        LOGGER.info("Variant %d: %s - Statistics", variant.number, variant.name)
        LOGGER.info("-" * 80)
        if result["status"] == "success":
            transcribe_result = result.get("result", {})
            duration = transcribe_result.get("duration")
            segments = transcribe_result.get("segments", [])
            language = transcribe_result.get("language")
            lang_prob = transcribe_result.get("language_probability")
            elapsed = result["elapsed_seconds"]

            LOGGER.info("Status:         ✅ Success")
            LOGGER.info("Processing:     %.2f seconds", elapsed)
            if duration:
                speed_ratio = duration / elapsed if elapsed > 0 else 0
                LOGGER.info("Audio duration: %.2f seconds (%.2f minutes)", duration, duration / 60)
                LOGGER.info("Speed:          %.2fx realtime", speed_ratio)
            LOGGER.info("Segments:       %d", len(segments))
            if language:
                if lang_prob is not None:
                    LOGGER.info("Language:       %s (confidence: %.1f%%)", language, lang_prob * 100)
                else:
                    LOGGER.info("Language:       %s", language)
            if result.get("output_files"):
                LOGGER.info("Output files:")
                for file_type, file_path in result["output_files"].items():
                    # Show only filename, not full path
                    file_name = Path(file_path).name
                    LOGGER.info("  %-6s: %s", file_type.upper(), file_name)
        else:
            LOGGER.error("Status:         ❌ Failed")
            LOGGER.error("Processing:     %.2f seconds", result["elapsed_seconds"])
            LOGGER.error("Error:          %s", result.get("error", "Unknown error"))

    # Print final summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("FINAL SUMMARY - All Variants")
    LOGGER.info("=" * 80)
    for variant_name, variant_number in summary_variants:
        # Find result for this variant
        result = next((r for r in results if r["variant_number"] == variant_number), None)
        if result:
            status_icon = "✅" if result["status"] == "success" else "❌"
            transcribe_result = result.get("result", {})
            duration = transcribe_result.get("duration")
            segments = transcribe_result.get("segments", [])
            elapsed = result["elapsed_seconds"]

            # Build summary line
            summary_parts = [
                f"{status_icon} Variant {variant_number}: {variant_name}",
                f"Status: {result['status']}",
                f"Time: {elapsed:.1f}s",
            ]
            if duration:
                speed_ratio = duration / elapsed if elapsed > 0 else 0
                summary_parts.append(f"Speed: {speed_ratio:.2f}x")
            summary_parts.append(f"Segments: {len(segments)}")

            LOGGER.info(" | ".join(summary_parts))
            if result["status"] == "error":
                LOGGER.error("   Error: %s", result.get("error", "Unknown error"))
        else:
            LOGGER.warning("⚠️  Variant %d: %s - No result found", variant_number, variant_name)

    # Always save comparison summary JSON
    with comparison_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    LOGGER.info("")
    LOGGER.info("Comparison summary saved to: %s", comparison_json_path.name)

    # Return non-zero if any variant failed
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
