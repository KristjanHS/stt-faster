#!/usr/bin/env python3
"""Generate a comparison report for variant transcription results.

This script reads JSON outputs from variant runs and generates a structured report
with metrics, excerpts, and quality assessments.

Usage:
    .venv/bin/python scripts/generate_variant_report.py /path/to/variant_outputs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def format_time(seconds: float | None) -> str:
    """Format seconds as mm:ss."""
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def find_segments_in_range(segments: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    """Find segments that overlap with the given time range."""
    result = []
    for seg in segments:
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if seg_start is not None and seg_end is not None:
            # Check if segment overlaps with range
            if seg_start < end_s and seg_end > start_s:
                result.append(seg)
    return result


def get_top_no_speech_segments(segments: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """Get top N segments by no_speech_prob."""
    segments_with_prob = [seg for seg in segments if seg.get("no_speech_prob") is not None]
    sorted_segments = sorted(segments_with_prob, key=lambda s: s.get("no_speech_prob", 0.0), reverse=True)
    return sorted_segments[:top_n]


def calculate_transcript_stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate transcript statistics."""
    if not segments:
        return {
            "segment_count": 0,
            "total_chars": 0,
            "total_words": 0,
            "last_end_s": None,
        }

    total_chars = sum(len(seg.get("text", "")) for seg in segments)
    total_words = sum(len(seg.get("text", "").split()) for seg in segments)
    end_times = [float(seg["end"]) for seg in segments if seg.get("end") is not None]
    last_end = max(end_times) if end_times else None

    return {
        "segment_count": len(segments),
        "total_chars": total_chars,
        "total_words": total_words,
        "last_end_s": last_end,
    }


def is_timestamp_folder(folder_name: str) -> bool:
    """Check if folder name matches timestamp pattern YYYY-MM-DDTHH-MM-SS."""
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$"
    return bool(re.match(pattern, folder_name))


def load_run_meta(run_folder: Path) -> dict[str, Any] | None:
    """Load and validate run_meta.json.

    Args:
        run_folder: Path to the run folder containing run_meta.json

    Returns:
        Parsed run_meta dictionary or None if not found/invalid
    """
    run_meta_file = run_folder / "run_meta.json"
    if not run_meta_file.exists():
        return None

    try:
        with run_meta_file.open("r", encoding="utf-8") as f:
            run_meta = json.load(f)
        variants = run_meta.get("variants", [])
        if not variants:
            return None
        return run_meta
    except Exception as e:
        print(f"Error reading {run_meta_file}: {e}", file=sys.stderr)  # noqa: T201
        return None


def load_variant_data(run_folder: Path, variant_number: int) -> dict[str, Any] | None:
    """Load variant data from JSON files and run_meta.json.

    Args:
        run_folder: Path to the timestamped run folder
        variant_number: Variant number to load

    Returns:
        Dictionary with variant data or None if not found
    """
    run_meta = load_run_meta(run_folder)
    if not run_meta:
        return None

    # Find variant metadata
    variants = run_meta.get("variants", [])
    variant_meta = next((v for v in variants if v.get("variant_number") == variant_number), None)
    if not variant_meta:
        return None

    # Use stored JSON filename if available, otherwise fall back to globbing (backward compatibility)
    json_filename = variant_meta.get("json_filename")
    if json_filename:
        json_file = run_folder / json_filename
    else:
        # Fallback: try to find JSON file by pattern (for old run_meta.json without json_filename)
        json_files = list(run_folder.glob(f"variant_{variant_number:03d}_*.json"))
        if not json_files:
            json_files = list(run_folder.glob(f"{variant_number}_*.json"))
            json_files = [f for f in json_files if "_comparison_" not in f.name]
        if not json_files:
            return None
        json_file = json_files[0]

    if not json_file.exists():
        return None

    try:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)  # noqa: T201
        return None

    return {
        "json_data": data,
        "run_meta": run_meta,
        "variant_meta": variant_meta,
        "variant_number": variant_meta.get("variant_number"),
        "variant_name": variant_meta.get("variant_name"),
    }


def generate_variant_report(
    variant_data: dict[str, Any],
    far_speaker_range: tuple[float, float] | None = None,
    silence_range: tuple[float, float] | None = None,
) -> str:
    """Generate a report for a single variant."""
    json_data = variant_data["json_data"]
    variant_meta = variant_data.get("variant_meta", {})
    variant_number = variant_data.get("variant_number", "?")
    variant_name = variant_data.get("variant_name", "unknown")

    segments = json_data.get("segments", [])
    stats = calculate_transcript_stats(segments)

    # Get skip count from JSON metrics (already calculated during transcription)
    metrics = json_data.get("metrics", {})
    no_speech_skips = metrics.get("no_speech_skips_count")
    if no_speech_skips is None:
        # Fallback: recalculate if not in metrics (backward compatibility)
        no_speech_skips = 0
        config = variant_meta.get("transcription_config", {})
        no_speech_threshold = config.get("no_speech_threshold") if config else None
        logprob_threshold = config.get("logprob_threshold") if config else None
        if no_speech_threshold is not None and logprob_threshold is not None:
            for seg in segments:
                no_speech_prob = seg.get("no_speech_prob")
                avg_logprob = seg.get("avg_logprob")
                if (
                    no_speech_prob is not None
                    and avg_logprob is not None
                    and no_speech_prob > no_speech_threshold
                    and avg_logprob <= logprob_threshold
                ):
                    no_speech_skips += 1

    # Runtime - approximate from duration if not in metadata
    runtime_s = None  # Would need to be extracted from logs or database

    lines = []
    lines.append(f"Variant {variant_number} ({variant_name}):")
    lines.append(f"- runtime_s: {runtime_s if runtime_s is not None else 'N/A (check logs/db)'}")
    lines.append(f"- segments_count: {stats['segment_count']}")
    lines.append(f"- transcript_chars: {stats['total_chars']}")
    lines.append(f"- transcript_words: {stats['total_words']}")
    lines.append(f"- last_end_s: {stats['last_end_s']:.1f}" if stats["last_end_s"] else "- last_end_s: N/A")
    lines.append(f"- no_speech_skips_count: {no_speech_skips}")

    # Far speaker excerpt
    if far_speaker_range:
        start_s, end_s = far_speaker_range
        far_segments = find_segments_in_range(segments, start_s, end_s)
        lines.append(f"- Far-speaker excerpt [{format_time(start_s)}–{format_time(end_s)}]:")
        if far_segments:
            for seg in far_segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                avg_lp = seg.get("avg_logprob", "N/A")
                no_speech = seg.get("no_speech_prob", "N/A")
                text = seg.get("text", "").strip()
                lines.append(f"  - ({start:.1f}-{end:.1f}, avg_lp={avg_lp}, no_speech={no_speech}) {text}")
        else:
            lines.append("  - (no segments in this range)")

    # Silence excerpt
    if silence_range:
        start_s, end_s = silence_range
        silence_segments = find_segments_in_range(segments, start_s, end_s)
        lines.append(f"- Silence excerpt [{format_time(start_s)}–{format_time(end_s)}]:")
        if silence_segments:
            for seg in silence_segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                avg_lp = seg.get("avg_logprob", "N/A")
                no_speech = seg.get("no_speech_prob", "N/A")
                text = seg.get("text", "").strip()
                lines.append(f"  - ({start:.1f}-{end:.1f}, avg_lp={avg_lp}, no_speech={no_speech}) {text}")
        else:
            lines.append("  - (no segments in this range)")

    # Top 5 highest no_speech_prob segments
    top_segments = get_top_no_speech_segments(segments, 5)
    if top_segments:
        lines.append("- Top 5 highest no_speech_prob segments:")
        for seg in top_segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            no_speech = seg.get("no_speech_prob", "N/A")
            avg_lp = seg.get("avg_logprob", "N/A")
            text = seg.get("text", "").strip()[:80]  # Truncate long text
            lines.append(f'  - t={start:.0f}-{end:.0f}s no_speech={no_speech:.4f} avg_lp={avg_lp} text="{text}..."')

    # Subjective assessment placeholders
    lines.append("- subjective: hallucinations↓? [Y/N] | quiet speech lost? [Y/N] | better than baseline? [Y/N]")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate variant comparison report from JSON outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "variant_outputs_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to timestamped folder (YYYY-MM-DDTHH-MM-SS) or parent directory (default: auto-detect)",
    )
    parser.add_argument(
        "--far-speaker-range",
        type=str,
        help="Time range for far speaker excerpt (format: start-end in seconds, e.g., '20-60')",
    )
    parser.add_argument(
        "--silence-range",
        type=str,
        help="Time range for silence excerpt (format: start-end in seconds, e.g., '120-160')",
    )
    parser.add_argument(
        "--variants",
        type=str,
        help="Comma-separated list of variant numbers to include (e.g., '1,90,91,92,93,94,95,96')",
    )

    args = parser.parse_args()

    # Auto-detect variant_outputs_dir if not provided
    variant_outputs_dir = args.variant_outputs_dir
    if variant_outputs_dir is None:
        # Default: Look for latest timestamped folder (YYYY-MM-DDTHH-MM-SS) in C:\Users\PC\Downloads\transcribe\
        default_transcribe_dir = Path("/mnt/c/Users/PC/Downloads/transcribe")
        if default_transcribe_dir.exists() and default_transcribe_dir.is_dir():
            # Find folders matching timestamp pattern
            timestamp_folders = [
                f
                for f in default_transcribe_dir.iterdir()
                if f.is_dir() and is_timestamp_folder(f.name) and (f / "run_meta.json").exists()
            ]
            if timestamp_folders:
                # Sort by name (timestamp) descending to get latest
                timestamp_folders.sort(key=lambda p: p.name, reverse=True)
                variant_outputs_dir = timestamp_folders[0]
                print(f"Auto-detected latest timestamp folder: {variant_outputs_dir}", file=sys.stderr)  # noqa: T201
            else:
                print(  # noqa: T201
                    f"Error: No timestamp folders (YYYY-MM-DDTHH-MM-SS) found in {default_transcribe_dir}",
                    file=sys.stderr,
                )  # noqa: T201
                return 1
        else:
            print(f"Error: Default directory not found: {default_transcribe_dir}", file=sys.stderr)  # noqa: T201
            print("Please specify variant_outputs_dir as an argument.", file=sys.stderr)  # noqa: T201
            return 1

    if not variant_outputs_dir.exists():
        print(f"Error: Directory not found: {variant_outputs_dir}", file=sys.stderr)  # noqa: T201
        return 1

    # Parse time ranges
    far_speaker_range = None
    if args.far_speaker_range:
        try:
            # Strip quotes if present (defensive)
            range_str = args.far_speaker_range.strip("\"'")
            start, end = map(float, range_str.split("-"))
            far_speaker_range = (start, end)
        except ValueError:
            print(f"Error: Invalid far-speaker-range format: {args.far_speaker_range}", file=sys.stderr)  # noqa: T201
            return 1

    silence_range = None
    if args.silence_range:
        try:
            # Strip quotes if present (defensive)
            range_str = args.silence_range.strip("\"'")
            start, end = map(float, range_str.split("-"))
            silence_range = (start, end)
        except ValueError:
            print(f"Error: Invalid silence-range format: {args.silence_range}", file=sys.stderr)  # noqa: T201
            return 1

    # Determine run folder: timestamp folder directly or search for timestamped subfolders
    if is_timestamp_folder(variant_outputs_dir.name) and (variant_outputs_dir / "run_meta.json").exists():
        # This is a timestamp folder with run_meta.json - use it directly
        run_folder = variant_outputs_dir
        print(f"Using timestamp folder directly: {run_folder.name}")  # noqa: T201
    else:
        # Find timestamped run folders (format: YYYY-MM-DDTHH-MM-SS) inside variant_outputs_dir
        run_folders = [
            rf
            for rf in variant_outputs_dir.iterdir()
            if rf.is_dir() and is_timestamp_folder(rf.name) and (rf / "run_meta.json").exists()
        ]
        if not run_folders:
            print(f"Error: No timestamp folders (YYYY-MM-DDTHH-MM-SS) found in {variant_outputs_dir}", file=sys.stderr)  # noqa: T201
            return 1
        # Sort by name (timestamp) descending to get latest
        run_folders.sort(key=lambda p: p.name, reverse=True)
        run_folder = run_folders[0]
        print(f"Using timestamp folder: {run_folder.name}")  # noqa: T201
    print()  # noqa: T201

    # Load run_meta.json using unified helper
    run_meta = load_run_meta(run_folder)
    if not run_meta:
        print(f"Error: Could not load run_meta.json from {run_folder}", file=sys.stderr)  # noqa: T201
        return 1

    variants = run_meta.get("variants", [])
    if not variants:
        print("Error: No variants found in run_meta.json", file=sys.stderr)  # noqa: T201
        return 1

    # Filter by variant numbers if specified
    if args.variants:
        requested_numbers = {int(v.strip()) for v in args.variants.split(",")}
        variants = [v for v in variants if v.get("variant_number") in requested_numbers]

    # Generate reports
    print("=" * 80)  # noqa: T201
    print("Variant Comparison Report")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    for variant_meta in variants:
        variant_number = variant_meta.get("variant_number")
        if variant_number is None:
            continue

        variant_data = load_variant_data(run_folder, variant_number)
        if variant_data is None:
            print(f"Warning: Could not load data for variant {variant_number}", file=sys.stderr)  # noqa: T201
            continue

        report = generate_variant_report(variant_data, far_speaker_range, silence_range)
        print(report)  # noqa: T201
        print()  # noqa: T201

    return 0


if __name__ == "__main__":
    sys.exit(main())
