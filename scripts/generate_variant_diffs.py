#!/usr/bin/env python3
"""Generate markdown diff report from transcription variant comparison JSON.

This script reads a comparison JSON file (either a summary with all variants
or individual variant files) and generates a concise markdown report showing
differences between variants.
"""

from __future__ import annotations

import argparse
import json
import logging
from difflib import unified_diff
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Variant descriptions from compare_transcription_variants.py
VARIANT_DESCRIPTIONS = {
    "no_preprocessing": "No preprocessing + project defaults",
    "noprep_minimal": "No preprocessing + minimal defaults",
    "ffmpeg_only": "Only ffmpeg pipeline + project defaults",
    "denoise_only": "Only denoise_light + project defaults",
    "ffmpeg_minimal": "Only ffmpeg pipeline + minimal defaults",
    "full_minimal": "Full preprocessing (ffmpeg + denoise) + minimal defaults",
    "noprep_noparamtrans": "No preprocessing + minimal transcription parameters",
    "normonly_noparamtrans": "Only normalization + minimal transcription parameters",
    "onlyden_noparamtrans": "Only denoise + minimal transcription parameters",
    "noprep_minimal_no_speech_threshold": "No preprocessing + minimal params + no_speech_threshold override",
    "noprep_minimal_chunk_length": "No preprocessing + minimal params + chunk_length override",
    "noprep_minimal_condition_on_previous_text": (
        "No preprocessing + minimal params + condition_on_previous_text override"
    ),
    "noprep_minimal_beam_size": "No preprocessing + minimal params + beam_size override",
}


def extract_text_from_result(result: dict[str, Any]) -> str:
    """Extract full transcription text from a result dictionary.

    Args:
        result: Result dictionary with 'segments' key

    Returns:
        Full transcription text as a single string
    """
    segments = result.get("segments", [])
    texts = [segment.get("text", "").strip() for segment in segments if segment.get("text")]
    return "\n".join(texts)


def extract_text_from_variant(variant_data: dict[str, Any]) -> str:
    """Extract transcription text from a variant result.

    Args:
        variant_data: Variant data dictionary (either from comparison summary or individual file)

    Returns:
        Full transcription text
    """
    # If it's a comparison summary entry, the result is in 'result' key
    if "result" in variant_data:
        return extract_text_from_result(variant_data["result"])
    # If it's an individual variant file, it's the result itself
    return extract_text_from_result(variant_data)


def load_comparison_data(json_path: Path) -> list[dict[str, Any]]:
    """Load comparison data from JSON file.

    Handles both comparison summary files and individual variant files.

    Args:
        json_path: Path to JSON file

    Returns:
        List of variant data dictionaries
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # If it's a list, it's a comparison summary
    if isinstance(data, list):
        return data

    # If it's a dict, it might be a single variant result
    # Check if it has 'segments' (individual variant file) or 'result' (wrapped variant)
    if isinstance(data, dict):
        if "segments" in data:
            # Individual variant file - wrap it
            return [{"variant": "single_variant", "variant_number": 0, "result": data}]
        elif "result" in data:
            # Already wrapped
            return [data]
        elif "variant" in data:
            # Comparison summary entry
            return [data]

    raise ValueError(f"Unexpected JSON structure in {json_path}")


def find_variant_files(directory: Path, base_name: str) -> list[Path]:
    """Find all variant JSON files in a directory matching the base name.

    Args:
        directory: Directory to search
        base_name: Base name to match (e.g., "Recording_3_short")

    Returns:
        List of matching JSON file paths, sorted by variant number
    """
    pattern = f"*_{base_name}_*.json"
    files = list(directory.glob(pattern))
    # Sort by variant number (extracted from filename)
    files.sort(key=lambda p: _extract_variant_number(p.name))
    return files


def _extract_variant_number(filename: str) -> int:
    """Extract variant number from filename (e.g., "1_nopre_projdef_..." -> 1).

    Args:
        filename: Filename to parse

    Returns:
        Variant number, or 999 if not found
    """
    try:
        # Filename format: {variant_number}_{variant_desc}_{base_name}_{datetime}.json
        parts = filename.split("_", 1)
        if parts:
            return int(parts[0])
    except (ValueError, IndexError):
        pass
    return 999


def generate_markdown_report(
    variants: list[dict[str, Any]],
    output_path: Path,
    base_name: str | None = None,
) -> None:
    """Generate markdown diff report from variant data.

    Args:
        variants: List of variant data dictionaries
        output_path: Path to save markdown file
        base_name: Optional base name for the report title
    """
    # Filter successful variants
    successful_variants = [v for v in variants if v.get("status") == "success"]
    if not successful_variants:
        LOGGER.warning("No successful variants found")
        return

    # Extract texts
    variant_texts: dict[int, str] = {}
    variant_info: dict[int, dict[str, Any]] = {}
    for variant in successful_variants:
        variant_num = variant.get("variant_number", 0)
        variant_name = variant.get("variant", "unknown")
        variant_texts[variant_num] = extract_text_from_variant(variant)
        variant_info[variant_num] = {
            "name": variant_name,
            "description": VARIANT_DESCRIPTIONS.get(variant_name, variant_name),
            "elapsed": variant.get("elapsed_seconds", 0),
            "result": variant.get("result", {}),
        }

    # Generate markdown
    lines: list[str] = []
    lines.append("# Transcription Variant Comparison Report")
    lines.append("")
    if base_name:
        lines.append(f"**Audio file:** `{base_name}`")
        lines.append("")
    lines.append(f"**Number of variants:** {len(successful_variants)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | Description | Status | Time (s) | Segments | Language |")
    lines.append("|---------|-------------|--------|----------|----------|----------|")
    for variant_num in sorted(variant_info.keys()):
        info = variant_info[variant_num]
        result = info["result"]
        segments = result.get("segments", [])
        language = result.get("language", "N/A")
        lang_prob = result.get("language_probability")
        lang_str = f"{language}"
        if lang_prob is not None:
            lang_str += f" ({lang_prob * 100:.1f}%)"
        elapsed = info["elapsed"]
        lines.append(f"| {variant_num} | {info['description']} | ✅ | {elapsed:.2f} | {len(segments)} | {lang_str} |")
    lines.append("")

    # Full transcriptions
    lines.append("## Full Transcriptions")
    lines.append("")
    for variant_num in sorted(variant_info.keys()):
        info = variant_info[variant_num]
        text = variant_texts[variant_num]
        lines.append(f"### Variant {variant_num}: {info['description']}")
        lines.append("")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        lines.append("")

    # Pairwise diffs
    lines.append("## Pairwise Differences")
    lines.append("")
    variant_nums = sorted(variant_info.keys())
    for i, variant_num_a in enumerate(variant_nums):
        for variant_num_b in variant_nums[i + 1 :]:
            info_a = variant_info[variant_num_a]
            info_b = variant_info[variant_num_b]
            text_a = variant_texts[variant_num_a]
            text_b = variant_texts[variant_num_b]

            if text_a == text_b:
                lines.append(f"### Variant {variant_num_a} vs {variant_num_b}: **Identical**")
                lines.append("")
                lines.append(f"- **{info_a['description']}**")
                lines.append(f"- **{info_b['description']}**")
                lines.append("")
            else:
                lines.append(f"### Variant {variant_num_a} vs {variant_num_b}")
                lines.append("")
                lines.append(f"- **Variant {variant_num_a}:** {info_a['description']}")
                lines.append(f"- **Variant {variant_num_b}:** {info_b['description']}")
                lines.append("")
                lines.append("**Diff:**")
                lines.append("")
                lines.append("```diff")
                # Generate unified diff
                diff_lines = unified_diff(
                    text_a.splitlines(keepends=True),
                    text_b.splitlines(keepends=True),
                    fromfile=f"Variant {variant_num_a}",
                    tofile=f"Variant {variant_num_b}",
                    lineterm="",
                )
                for diff_line in diff_lines:
                    lines.append(diff_line.rstrip())
                lines.append("```")
                lines.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    LOGGER.info("Markdown report saved to: %s", output_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate markdown diff report from transcription variant comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to comparison JSON file (summary or individual variant)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output markdown file path (default: {json_file_stem}_diffs.md)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="If provided, search for all variant files in this directory matching the base name",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s - %(message)s",
    )

    json_path = Path(args.json_file)
    if not json_path.exists():
        LOGGER.error("JSON file not found: %s", json_path)
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix(".md").with_name(json_path.stem + "_diffs.md")

    # Load data
    if args.directory:
        # Search for all variant files in directory
        directory = Path(args.directory)
        base_name = json_path.stem.split("_for_test_comparison")[0].split("_comparison")[0]
        variant_files = find_variant_files(directory, base_name)
        if not variant_files:
            LOGGER.warning("No variant files found in %s matching %s", directory, base_name)
            LOGGER.info("Falling back to single file mode")
            variants = load_comparison_data(json_path)
        else:
            LOGGER.info("Found %d variant files in %s", len(variant_files), directory)
            variants: list[dict[str, Any]] = []
            for variant_file in variant_files:
                variant_data = load_comparison_data(variant_file)
                variants.extend(variant_data)
    else:
        variants = load_comparison_data(json_path)

    if not variants:
        LOGGER.error("No variant data found in %s", json_path)
        return 1

    LOGGER.info("Loaded %d variants", len(variants))

    # Extract base name for report
    base_name = json_path.stem
    if "_for_test_comparison" in base_name:
        base_name = base_name.split("_for_test_comparison")[0]
    elif "_comparison" in base_name:
        base_name = base_name.split("_comparison")[0]

    # Generate report
    generate_markdown_report(variants, output_path, base_name)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
