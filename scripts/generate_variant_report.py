#!/usr/bin/env python3
"""Generate a comparison report for variant transcription results.

This script reads JSON outputs from variant runs and generates a structured report
with metrics, excerpts, and quality assessments.

Usage:
    .venv/bin/python scripts/generate_variant_report.py /path/to/timestamped_folder
    .venv/bin/python scripts/generate_variant_report.py /path/to/parent_directory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.variants.registry import baseline_variant_num


def format_time(seconds: float | None) -> str:
    """Format seconds as mm:ss."""
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def find_segments_in_range(
    segments: List[Dict[str, Any]],
    start_s: float,
    end_s: float,
    min_overlap_s: float = 3.0,
    min_overlap_ratio: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """Find segments that meaningfully overlap with the given time range.

    Returns:
        Tuple of (overlapping_segments, contained_segments, total_coverage_s)
        - overlapping_segments: Segments with meaningful overlap (>= min_overlap_s or >= min_overlap_ratio)
        - contained_segments: Segments fully inside the range
        - total_coverage_s: Total seconds of coverage by overlapping segments
    """
    range_duration: float = end_s - start_s
    overlapping: List[Dict[str, Any]] = []
    contained: List[Dict[str, Any]] = []
    total_coverage: float = 0.0

    for seg in segments:
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if seg_start is None or seg_end is None:
            continue

        # Compute overlap
        overlap = max(0.0, min(seg_end, end_s) - max(seg_start, start_s))

        # Check if meaningful overlap
        has_meaningful_overlap = overlap >= min_overlap_s or overlap >= (min_overlap_ratio * range_duration)

        if has_meaningful_overlap:
            overlapping.append(seg)
            total_coverage += overlap

            # Check if fully contained
            if seg_start >= start_s and seg_end <= end_s:
                contained.append(seg)

    return overlapping, contained, total_coverage


def extract_text_in_range(segment: Dict[str, Any], start_s: float, end_s: float) -> str:
    """Extract text from segment using word timestamps for accurate slicing.

    If word timestamps are available, returns only words that fall within the range.
    Otherwise, returns the full segment text.

    Args:
        segment: Segment dictionary with optional 'words' list
        start_s: Start time of range
        end_s: End time of range

    Returns:
        Text string containing only words within the time range
    """
    words: List[Dict[str, Any]] | None = segment.get("words")
    if not words:
        # Fallback to full segment text if no word timestamps
        return segment.get("text", "").strip()

    # Extract words that fall within the range
    words_in_range: List[str] = []
    for word in words:
        word_start: float | None = word.get("start")
        word_end: float | None = word.get("end")
        word_text: str = word.get("word", "")

        if word_start is not None and word_end is not None:
            # Word overlaps with range if it starts before end_s and ends after start_s
            if word_start < end_s and word_end > start_s:
                words_in_range.append(word_text)

    if words_in_range:
        return " ".join(words_in_range)
    return ""


def get_top_no_speech_segments(segments: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N segments by no_speech_prob."""
    segments_with_prob: List[Dict[str, Any]] = [seg for seg in segments if seg.get("no_speech_prob") is not None]
    sorted_segments: List[Dict[str, Any]] = sorted(
        segments_with_prob, key=lambda s: s.get("no_speech_prob", 0.0), reverse=True
    )
    return sorted_segments[:top_n]


def calculate_transcript_stats(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def calculate_activation_potential(
    segments: list[dict[str, Any]],
    no_speech_threshold: float | None,
    logprob_threshold: float | None,
) -> dict[str, Any]:
    """Calculate activation potential metrics for skip rule.

    Args:
        segments: List of segment dictionaries
        no_speech_threshold: Threshold for no_speech_prob (None means no threshold)
        logprob_threshold: Threshold for avg_logprob (None means no threshold)

    Returns:
        Dictionary with max_no_speech_prob, min_avg_logprob, count_ns_candidates, count_would_skip
    """
    no_speech_probs: list[float] = []
    avg_logprobs: list[float] = []
    count_ns_candidates = 0
    count_would_skip = 0

    for seg in segments:
        no_speech_prob = seg.get("no_speech_prob")  # type: ignore[reportUnknownMemberType]
        avg_logprob = seg.get("avg_logprob")  # type: ignore[reportUnknownMemberType]

        if no_speech_prob is not None:
            no_speech_probs.append(no_speech_prob)
            if no_speech_threshold is not None and no_speech_prob > no_speech_threshold:
                count_ns_candidates += 1

        if avg_logprob is not None:
            avg_logprobs.append(avg_logprob)

        if no_speech_prob is not None and avg_logprob is not None:
            if no_speech_threshold is not None and logprob_threshold is not None:
                if no_speech_prob > no_speech_threshold and avg_logprob <= logprob_threshold:
                    count_would_skip += 1

    return {
        "max_no_speech_prob": max(no_speech_probs) if no_speech_probs else None,
        "min_avg_logprob": min(avg_logprobs) if avg_logprobs else None,
        "count_ns_candidates": count_ns_candidates,
        "count_would_skip": count_would_skip,
    }


def recalculate_no_speech_skips(
    segments: list[dict[str, Any]],
    no_speech_threshold: float | None,
    logprob_threshold: float | None,
) -> int:
    """Recalculate how many segments would be skipped by the skip rule.

    Rule: skip if no_speech_prob > no_speech_threshold AND avg_logprob <= logprob_threshold

    Args:
        segments: List of segment dictionaries
        no_speech_threshold: Threshold for no_speech_prob (None means no threshold)
        logprob_threshold: Threshold for avg_logprob (None means no threshold)

    Returns:
        Count of segments that would be skipped
    """
    if no_speech_threshold is None or logprob_threshold is None:
        return 0

    count = 0
    for seg in segments:
        no_speech_prob = seg.get("no_speech_prob")
        avg_logprob = seg.get("avg_logprob")

        if no_speech_prob is not None and avg_logprob is not None:
            if no_speech_prob > no_speech_threshold and avg_logprob <= logprob_threshold:
                count += 1

    return count


def calculate_words_in_range(
    segments: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    min_overlap_s: float = 3.0,
    min_overlap_ratio: float = 0.2,
) -> dict[str, Any]:
    """Calculate word-level statistics for a time range.

    Uses the same overlap criteria as find_segments_in_range() to ensure consistency.
    Only counts words from segments with meaningful overlap.

    Returns:
        Dictionary with words_in_range, chars_in_range, unique_words_in_range
        Falls back to segment-level if word timestamps aren't available
    """
    # First, find segments with meaningful overlap (same criteria as excerpts)
    overlapping_segments, _, _ = find_segments_in_range(segments, start_s, end_s, min_overlap_s, min_overlap_ratio)

    words_in_range: list[str] = []
    chars_in_range = 0
    unique_words: set[str] = set()
    text_source = "words"  # Track whether we're using word timestamps or fallback

    for seg in overlapping_segments:
        words = seg.get("words")
        if words:
            # Use word timestamps - but only count words that actually fall in the range
            for word in words:
                word_start = word.get("start")  # type: ignore[reportUnknownMemberType]
                word_end = word.get("end")  # type: ignore[reportUnknownMemberType]
                word_text = word.get("word", "").strip()  # type: ignore[reportUnknownMemberType]

                if word_start is not None and word_end is not None:
                    # Word overlaps with range if it starts before end_s and ends after start_s
                    if word_start < end_s and word_end > start_s:
                        words_in_range.append(word_text)
                        chars_in_range += len(word_text)
                        # Normalize word for uniqueness (lowercase, remove punctuation)
                        normalized = word_text.lower().strip(".,!?;:()[]{}")
                        if normalized:
                            unique_words.add(normalized)
        else:
            # Fallback: use segment text if word timestamps aren't available
            text_source = "segment_fallback"
            seg_text = seg.get("text", "").strip()
            if seg_text:
                # Approximate: count words in overlapping segment
                # Note: this is approximate since we can't slice segment text by time
                seg_words = seg_text.split()
                words_in_range.extend(seg_words)
                chars_in_range += len(seg_text)
                for w in seg_words:
                    normalized = w.lower().strip(".,!?;:()[]{}")
                    if normalized:
                        unique_words.add(normalized)

    return {
        "words_in_range": len(words_in_range),
        "chars_in_range": chars_in_range,
        "unique_words_in_range": len(unique_words),
        "has_word_timestamps": any(seg.get("words") for seg in overlapping_segments),
        "text_source": text_source,
    }


def calculate_silence_metrics(
    segments: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    min_overlap_s: float = 3.0,
    min_overlap_ratio: float = 0.2,
) -> dict[str, Any]:
    """Calculate judge-like metrics for silence range.

    Uses the same overlap criteria as find_segments_in_range() to ensure consistency.
    Only counts words/segments with meaningful overlap.

    Returns:
        Dictionary with silence_word_count, silence_segments_count,
        silence_top_no_speech_prob, silence_avg_logprob_mean, silence_avg_logprob_min
    """
    # First, find segments with meaningful overlap (same criteria as excerpts)
    silence_segments, _, _ = find_segments_in_range(segments, start_s, end_s, min_overlap_s, min_overlap_ratio)

    silence_words: list[str] = []
    no_speech_probs: list[float] = []
    avg_logprobs: list[float] = []
    text_source = "words"  # Track whether we're using word timestamps or fallback

    for seg in silence_segments:
        no_speech_prob = seg.get("no_speech_prob")
        avg_logprob = seg.get("avg_logprob")

        if no_speech_prob is not None:
            no_speech_probs.append(no_speech_prob)
        if avg_logprob is not None:
            avg_logprobs.append(avg_logprob)

        # Count words in this segment that fall in range
        words = seg.get("words")
        if words:
            for word in words:
                word_start = word.get("start")
                word_end = word.get("end")
                if word_start is not None and word_end is not None:
                    if word_start < end_s and word_end > start_s:
                        silence_words.append(word.get("word", ""))  # type: ignore[reportUnknownMemberType]
        else:
            # Fallback: approximate from segment text
            text_source = "segment_fallback"
            seg_text = seg.get("text", "").strip()
            if seg_text:
                silence_words.extend(seg_text.split())

    return {
        "silence_word_count": len(silence_words),
        "silence_segments_count": len(silence_segments),
        "silence_top_no_speech_prob": max(no_speech_probs) if no_speech_probs else None,
        "silence_avg_logprob_mean": sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None,
        "silence_avg_logprob_min": min(avg_logprobs) if avg_logprobs else None,
        "text_source": text_source,
    }


def simulate_counterfactual_skips(
    reference_segments: list[dict[str, Any]],
    no_speech_threshold: float,
    logprob_threshold: float,
) -> tuple[int, list[dict[str, Any]]]:
    """Simulate how many reference segments would be skipped by a given threshold pair.

    Args:
        reference_segments: Segments from reference variant (e.g., Variant 1)
        no_speech_threshold: Threshold for no_speech_prob
        logprob_threshold: Threshold for avg_logprob

    Returns:
        Tuple of (count, list of segments that would be skipped with their details)
    """
    skipped_segments: list[dict[str, Any]] = []

    for seg in reference_segments:
        no_speech_prob = seg.get("no_speech_prob")
        avg_logprob = seg.get("avg_logprob")

        if no_speech_prob is not None and avg_logprob is not None:
            if no_speech_prob > no_speech_threshold and avg_logprob <= logprob_threshold:
                skipped_segments.append(
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip()[:80],
                        "no_speech_prob": no_speech_prob,
                        "avg_logprob": avg_logprob,
                    }
                )

    return len(skipped_segments), skipped_segments


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


def find_best_reference_variant(
    current_variant_meta: dict[str, Any],
    all_variants: list[dict[str, Any]],
    run_folder: Path,
    default_reference: int | None = None,
) -> tuple[int, list[dict[str, Any]] | None]:
    """Find the best reference variant for counterfactual analysis.

    Prefers variants with same vad_filter and chunk_length as current variant.
    Falls back to baseline_variant_num if no better match exists.

    Args:
        current_variant_meta: Metadata for the current variant
        all_variants: List of all variant metadata dictionaries
        run_folder: Path to run folder containing JSON files
        default_reference: Default reference variant number to use as fallback (defaults to baseline_variant_num)

    Returns:
        Tuple of (reference_variant_number, reference_segments or None)
    """
    if default_reference is None:
        default_reference = baseline_variant_num
    current_config = normalize_transcription_config(current_variant_meta.get("transcription_config", {}))
    current_vad = current_config.get("vad_filter")
    current_chunk = current_config.get("chunk_length")

    # Try to find a variant with matching vad_filter and chunk_length
    best_match = None
    best_score = 0

    for variant_meta in all_variants:
        variant_num = variant_meta.get("variant_number")
        if variant_num is None or variant_num == current_variant_meta.get("variant_number"):
            continue

        variant_config = normalize_transcription_config(variant_meta.get("transcription_config", {}))
        variant_vad = variant_config.get("vad_filter")
        variant_chunk = variant_config.get("chunk_length")

        # Score: 2 for vad match, 1 for chunk match
        score = 0
        if current_vad is not None and variant_vad == current_vad:
            score += 2
        if current_chunk is not None and variant_chunk == current_chunk:
            score += 1

        if score > best_score:
            best_score = score
            best_match = variant_num

    # Use best match if found, otherwise fall back to default
    reference_num = best_match if best_match is not None else default_reference

    # Load reference segments
    reference_data = load_variant_data(run_folder, reference_num)
    if reference_data:
        return reference_num, reference_data["json_data"].get("segments", [])

    return reference_num, None


def normalize_transcription_config(transcription_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize transcription config to use consistent key names.

    Handles both faster-whisper standard keys (log_prob_threshold) and legacy keys (logprob_threshold).

    Args:
        transcription_config: Raw config dictionary from variant metadata

    Returns:
        Normalized config dictionary with consistent keys
    """
    normalized = dict(transcription_config)

    # Normalize logprob_threshold key
    if "log_prob_threshold" in normalized:
        normalized["logprob_threshold"] = normalized["log_prob_threshold"]
    elif "logprob_threshold" in normalized:
        # Already normalized
        pass
    else:
        normalized["logprob_threshold"] = None

    return normalized


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
    reference_segments: list[dict[str, Any]] | None = None,
    baseline_stats: dict[str, Any] | None = None,
    reference_variant_number: int | None = None,
) -> str:
    """Generate a report for a single variant."""
    json_data = variant_data["json_data"]
    variant_number = variant_data.get("variant_number", "?")
    variant_name = variant_data.get("variant_name", "unknown")
    variant_meta = variant_data.get("variant_meta", {})

    segments = json_data.get("segments", [])
    stats = calculate_transcript_stats(segments)

    # Extract and normalize config from variant metadata
    raw_config = variant_meta.get("transcription_config", {})
    transcription_config = normalize_transcription_config(raw_config)
    no_speech_threshold = transcription_config.get("no_speech_threshold")
    logprob_threshold = transcription_config.get("logprob_threshold")
    vad_filter = transcription_config.get("vad_filter")
    chunk_length = transcription_config.get("chunk_length")
    condition_on_previous_text = transcription_config.get("condition_on_previous_text")

    # Check if word timestamps are present
    word_timestamps_present = any(seg.get("words") for seg in segments)

    # Get skip count from JSON metrics (for reference, but we'll recalculate)
    metrics = json_data.get("metrics", {})
    no_speech_skips_reported = metrics.get("no_speech_skips_count")

    # Calculate activation potential metrics
    activation = calculate_activation_potential(segments, no_speech_threshold, logprob_threshold)

    # Always recalculate skip counts (segments meeting skip rule)
    segments_meeting_skip_rule_in_output = recalculate_no_speech_skips(segments, no_speech_threshold, logprob_threshold)

    # Runtime - approximate from duration if not in metadata
    runtime_s = None  # Would need to be extracted from logs or database

    lines: list[str] = []
    lines.append(f"Variant {variant_number} ({variant_name}):")

    # Configuration block
    lines.append("- Configuration:")
    lines.append(f"  vad_filter: {vad_filter}")
    lines.append(f"  chunk_length: {chunk_length}")
    lines.append(f"  no_speech_threshold: {no_speech_threshold}")
    # Show which key was found (for debugging)
    logprob_key_used = (
        "log_prob_threshold" if transcription_config.get("log_prob_threshold") is not None else "logprob_threshold"
    )
    lines.append(f"  logprob_threshold ({logprob_key_used}): {logprob_threshold}")
    lines.append(f"  condition_on_previous_text: {condition_on_previous_text}")

    # Word timestamps flag (top-level)
    lines.append(f"- word_timestamps_present: {word_timestamps_present}")
    if not word_timestamps_present:
        lines.append("  ⚠️  WARNING: Word timestamps missing - range word counts are approximate")

    lines.append(f"- runtime_s: {runtime_s if runtime_s is not None else 'N/A (check logs/db)'}")
    lines.append(f"- segments_count: {stats['segment_count']}")
    lines.append(f"- transcript_chars: {stats['total_chars']}")
    lines.append(f"- transcript_words: {stats['total_words']}")
    lines.append(f"- last_end_s: {stats['last_end_s']:.1f}" if stats["last_end_s"] else "- last_end_s: N/A")

    # Diff from baseline (if provided)
    if baseline_stats:
        lines.append("- Diff from baseline:")
        delta_segments = stats["segment_count"] - baseline_stats.get("segment_count", 0)
        delta_words = stats["total_words"] - baseline_stats.get("total_words", 0)
        lines.append(f"  Δsegments_count: {delta_segments:+d}")
        lines.append(f"  Δwords_total: {delta_words:+d}")
        if silence_range:
            silence_word_stats = calculate_words_in_range(segments, silence_range[0], silence_range[1])
            baseline_silence_words = baseline_stats.get("silence_word_count", 0)
            delta_silence = silence_word_stats["words_in_range"] - baseline_silence_words
            lines.append(f"  Δsilence_word_count: {delta_silence:+d}")
        if far_speaker_range:
            far_word_stats = calculate_words_in_range(segments, far_speaker_range[0], far_speaker_range[1])
            baseline_far_words = baseline_stats.get("far_range_words", 0)
            delta_far = far_word_stats["words_in_range"] - baseline_far_words
            lines.append(f"  Δfar_range_words: {delta_far:+d}")

    # Activation potential metrics
    lines.append("- Activation potential:")
    if activation["max_no_speech_prob"] is not None:
        lines.append(f"  max_no_speech_prob: {activation['max_no_speech_prob']:.4f}")
    if activation["min_avg_logprob"] is not None:
        lines.append(f"  min_avg_logprob: {activation['min_avg_logprob']:.3f}")
    ns_thresh_str = str(no_speech_threshold) if no_speech_threshold is not None else "N/A"
    lines.append(
        f"  count_ns_candidates (segments with no_speech_prob > {ns_thresh_str}): {activation['count_ns_candidates']}"
    )
    lines.append(f"  count_would_skip (meets skip rule): {activation['count_would_skip']}")

    # No speech skips - always show all three metrics
    lines.append("- No speech skips:")
    reported_str = str(no_speech_skips_reported) if no_speech_skips_reported is not None else "N/A"
    lines.append(f"  no_speech_skips_reported (from metrics): {reported_str}")

    # Warn if thresholds are missing (one present but not the other)
    if (no_speech_threshold is not None and logprob_threshold is None) or (
        no_speech_threshold is None and logprob_threshold is not None
    ):
        lines.append(
            f"  ⚠️  WARNING: Incomplete threshold config "
            f"(no_speech={no_speech_threshold}, logprob={logprob_threshold}) - "
            f"skip counts may be inaccurate"
        )

    lines.append(f"  segments_meeting_skip_rule_in_output (recalc): {segments_meeting_skip_rule_in_output}")
    # Counterfactual skip simulation (if reference segments provided)
    if reference_segments is not None and no_speech_threshold is not None and logprob_threshold is not None:
        counterfactual_count, counterfactual_segments = simulate_counterfactual_skips(
            reference_segments, no_speech_threshold, logprob_threshold
        )
        ref_label = f"Variant {reference_variant_number}" if reference_variant_number else "reference"
        lines.append(f"  no_speech_skips_counterfactual_on_reference ({ref_label}): {counterfactual_count}")
        if counterfactual_segments:
            lines.append("  Counterfactual segments that would be skipped:")
            for seg_info in counterfactual_segments[:10]:  # Limit to top 10
                lines.append(
                    f"    - t={seg_info['start']:.1f}-{seg_info['end']:.1f}s "
                    f"no_speech={seg_info['no_speech_prob']:.4f} "
                    f"avg_lp={seg_info['avg_logprob']:.3f} "
                    f'text="{seg_info["text"]}..."'
                )
            if len(counterfactual_segments) > 10:
                lines.append(f"    ... and {len(counterfactual_segments) - 10} more")

    # Far speaker excerpt
    if far_speaker_range:
        start_s, end_s = far_speaker_range
        range_duration = end_s - start_s
        far_overlapping, far_contained, far_coverage = find_segments_in_range(segments, start_s, end_s)

        # Word-based range analysis (using same overlap criteria as segments)
        word_stats = calculate_words_in_range(segments, start_s, end_s)

        lines.append(f"- Far-speaker range [{format_time(start_s)}–{format_time(end_s)}]:")
        if word_stats["has_word_timestamps"]:
            lines.append(f"  words_in_range: {word_stats['words_in_range']} (text_source=words)")
            lines.append(f"  chars_in_range: {word_stats['chars_in_range']}")
            lines.append(f"  unique_words_in_range: {word_stats['unique_words_in_range']}")
        else:
            lines.append(f"  coverage_in_range (segment-based fallback): {far_coverage:.1f}s of {range_duration:.1f}s")
            text_src = word_stats.get("text_source", "unknown")
            lines.append(f"  words_in_range: {word_stats['words_in_range']} (text_source={text_src})")
            lines.append(
                "  ⚠️  WARNING: word timestamps not available - word counts are approximate (full segment text used)"
            )
        if far_overlapping:
            lines.append("  Segments overlapping:")
            for seg in far_overlapping:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                avg_lp = seg.get("avg_logprob", "N/A")
                no_speech = seg.get("no_speech_prob", "N/A")
                # Use word timestamps for accurate text slicing
                text = extract_text_in_range(seg, start_s, end_s)
                text_source_marker = "words"
                if not text:
                    # Fallback to full segment text if word extraction failed
                    text = seg.get("text", "").strip()
                    text_source_marker = "segment_fallback"
                # Truncate text to ~80 chars
                text_truncated = text[:80] + "..." if len(text) > 80 else text
                # Calculate overlap for this segment
                overlap = max(0.0, min(end, end_s) - max(start, start_s))
                lines.append(
                    f"    - ({start:.1f}-{end:.1f}, overlap={overlap:.1f}s, "
                    f"avg_lp={avg_lp}, no_speech={no_speech}, text_source={text_source_marker}) {text_truncated}"
                )
        else:
            lines.append("  - (no segments with meaningful overlap in this range)")

        if not far_contained:
            lines.append("  - (no segments fully inside this range - range may be silent or skipped)")

    # Silence excerpt
    if silence_range:
        start_s, end_s = silence_range
        range_duration = end_s - start_s
        silence_overlapping, silence_contained, silence_coverage = find_segments_in_range(segments, start_s, end_s)

        # Word-based range analysis (using same overlap criteria as segments)
        word_stats = calculate_words_in_range(segments, start_s, end_s)

        # Judge-like silence metrics (using same overlap criteria as segments)
        silence_metrics = calculate_silence_metrics(segments, start_s, end_s)

        lines.append(f"- Silence range [{format_time(start_s)}–{format_time(end_s)}]:")
        if word_stats["has_word_timestamps"]:
            lines.append(f"  words_in_range: {word_stats['words_in_range']} (text_source=words)")
            lines.append(f"  chars_in_range: {word_stats['chars_in_range']}")
            lines.append(f"  unique_words_in_range: {word_stats['unique_words_in_range']}")
        else:
            lines.append(
                f"  coverage_in_range (segment-based fallback): {silence_coverage:.1f}s of {range_duration:.1f}s"
            )
            text_src = word_stats.get("text_source", "unknown")
            lines.append(f"  words_in_range: {word_stats['words_in_range']} (text_source={text_src})")
            lines.append(
                "  ⚠️  WARNING: word timestamps not available - word counts are approximate (full segment text used)"
            )
        lines.append("  Silence metrics:")
        silence_text_src = silence_metrics.get("text_source", "unknown")
        lines.append(
            f"    silence_word_count: {silence_metrics['silence_word_count']} (text_source={silence_text_src})"
        )
        lines.append(f"    silence_segments_count: {silence_metrics['silence_segments_count']}")
        if silence_metrics["silence_top_no_speech_prob"] is not None:
            lines.append(f"    silence_top_no_speech_prob: {silence_metrics['silence_top_no_speech_prob']:.4f}")
        if silence_metrics["silence_avg_logprob_mean"] is not None:
            lines.append(f"    silence_avg_logprob_mean: {silence_metrics['silence_avg_logprob_mean']:.3f}")
        if silence_metrics["silence_avg_logprob_min"] is not None:
            lines.append(f"    silence_avg_logprob_min: {silence_metrics['silence_avg_logprob_min']:.3f}")

        if silence_overlapping:
            lines.append("  Segments overlapping:")
            for seg in silence_overlapping:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                avg_lp = seg.get("avg_logprob", "N/A")
                no_speech = seg.get("no_speech_prob", "N/A")
                # Use word timestamps for accurate text slicing
                text = extract_text_in_range(seg, start_s, end_s)
                text_source_marker = "words"
                if not text:
                    # Fallback to full segment text if word extraction failed
                    text = seg.get("text", "").strip()
                    text_source_marker = "segment_fallback"
                # Truncate text to ~80 chars
                text_truncated = text[:80] + "..." if len(text) > 80 else text
                # Calculate overlap for this segment
                overlap = max(0.0, min(end, end_s) - max(start, start_s))
                lines.append(
                    f"    - ({start:.1f}-{end:.1f}, overlap={overlap:.1f}s, "
                    f"avg_lp={avg_lp}, no_speech={no_speech}, text_source={text_source_marker}) {text_truncated}"
                )
        else:
            lines.append("  - (no segments with meaningful overlap in this range)")

        if not silence_contained:
            lines.append("  - (no segments fully inside this range - range may be silent or skipped)")

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

    # Load baseline variant for diff calculations
    # Try baseline variant first (from registry), then variant 1, then first available variant
    baseline_stats = None
    baseline_variant_num_local = None

    # Try baseline variant first
    baseline_data = load_variant_data(run_folder, baseline_variant_num)
    if baseline_data:
        baseline_variant_num_local = baseline_variant_num
    else:
        # Try variant 1 as fallback
        baseline_data = load_variant_data(run_folder, 1)
        if baseline_data:
            baseline_variant_num_local = 1
        else:
            # Try first available variant as last resort
            variant_numbers = sorted([v.get("variant_number") for v in variants if v.get("variant_number") is not None])
            if variant_numbers:
                baseline_data = load_variant_data(run_folder, variant_numbers[0])
                if baseline_data:
                    baseline_variant_num_local = variant_numbers[0]

    if baseline_data:
        baseline_segments = baseline_data["json_data"].get("segments", [])
        baseline_stats_calc = calculate_transcript_stats(baseline_segments)
        baseline_stats = {
            "segment_count": baseline_stats_calc["segment_count"],
            "total_words": baseline_stats_calc["total_words"],
        }
        if silence_range:
            baseline_silence_stats = calculate_words_in_range(baseline_segments, silence_range[0], silence_range[1])
            baseline_stats["silence_word_count"] = baseline_silence_stats["words_in_range"]
        if far_speaker_range:
            baseline_far_stats = calculate_words_in_range(baseline_segments, far_speaker_range[0], far_speaker_range[1])
            baseline_stats["far_range_words"] = baseline_far_stats["words_in_range"]
        print(f"Loaded baseline variant {baseline_variant_num_local} for diff calculations", file=sys.stderr)  # noqa: T201
    else:
        available_variants = sorted([v.get("variant_number") for v in variants if v.get("variant_number") is not None])
        available_str = ", ".join(map(str, available_variants)) if available_variants else "none"
        print(  # noqa: T201
            f"Warning: Could not load baseline variant for diff calculations. "
            f"Tried variants {baseline_variant_num}, 1, and first available. Available variants: {available_str}",
            file=sys.stderr,
        )

    # Generate reports
    report_lines: list[str] = []
    csv_rows: list[dict[str, Any]] = []

    report_lines.append("=" * 80)
    report_lines.append("Variant Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Generated from: {run_folder.name}")
    report_lines.append(f"Run folder: {run_folder}")
    report_lines.append("")

    for variant_meta in variants:
        variant_number = variant_meta.get("variant_number")
        if variant_number is None:
            continue

        variant_data = load_variant_data(run_folder, variant_number)
        if variant_data is None:
            print(f"Warning: Could not load data for variant {variant_number}", file=sys.stderr)  # noqa: T201
            continue

        # Find best reference variant for counterfactual analysis (same vad_filter/chunk_length)
        reference_num, reference_segments = find_best_reference_variant(variant_meta, variants, run_folder)

        # Use reference segments for counterfactual analysis (but not for the reference variant itself)
        ref_segs_for_this_variant = reference_segments if variant_number != reference_num else None

        report = generate_variant_report(
            variant_data,
            far_speaker_range,
            silence_range,
            ref_segs_for_this_variant,
            baseline_stats,
            reference_variant_number=reference_num if ref_segs_for_this_variant else None,
        )
        report_lines.append(report)
        report_lines.append("")

        # Collect data for CSV export
        json_data = variant_data["json_data"]
        segments = json_data.get("segments", [])
        stats = calculate_transcript_stats(segments)
        config = normalize_transcription_config(variant_meta.get("transcription_config", {}))
        activation = calculate_activation_potential(
            segments, config.get("no_speech_threshold"), config.get("logprob_threshold")
        )

        csv_row = {
            "variant_number": variant_number,
            "variant_name": variant_meta.get("variant_name", ""),
            "segment_count": stats["segment_count"],
            "total_words": stats["total_words"],
            "vad_filter": config.get("vad_filter"),
            "chunk_length": config.get("chunk_length"),
            "no_speech_threshold": config.get("no_speech_threshold"),
            "logprob_threshold": config.get("logprob_threshold"),
            "max_no_speech_prob": activation["max_no_speech_prob"],
            "min_avg_logprob": activation["min_avg_logprob"],
            "count_ns_candidates": activation["count_ns_candidates"],
            "count_would_skip": activation["count_would_skip"],
            "word_timestamps_present": any(seg.get("words") for seg in segments),
        }

        if silence_range:
            silence_stats = calculate_words_in_range(segments, silence_range[0], silence_range[1])
            csv_row["silence_word_count"] = silence_stats["words_in_range"]
        if far_speaker_range:
            far_stats = calculate_words_in_range(segments, far_speaker_range[0], far_speaker_range[1])
            csv_row["far_range_words"] = far_stats["words_in_range"]
        if baseline_stats:
            csv_row["delta_segments"] = stats["segment_count"] - baseline_stats.get("segment_count", 0)
            csv_row["delta_words"] = stats["total_words"] - baseline_stats.get("total_words", 0)

        csv_rows.append(csv_row)

    # Write to markdown file
    report_content = "\n".join(report_lines)

    # Create output directory
    output_dir = Path("reports/var_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on run folder name
    report_filename = f"variant_report_{run_folder.name}.md"
    report_path = output_dir / report_filename

    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Variant report saved to: {report_path}", file=sys.stderr)  # noqa: T201
    except Exception as e:
        print(f"Error writing report to {report_path}: {e}", file=sys.stderr)  # noqa: T201
        return 1

    # Write CSV export
    if csv_rows:
        csv_filename = f"variant_report_{run_folder.name}.csv"
        csv_path = output_dir / csv_filename

        # Get all possible keys from all rows
        all_keys: set[str] = set()
        for row in csv_rows:
            all_keys.update(row.keys())

        fieldnames = sorted(all_keys)

        try:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)  # type: ignore[reportUnknownMemberType]
            print(f"CSV export saved to: {csv_path}", file=sys.stderr)  # noqa: T201
        except Exception as e:
            print(f"Error writing CSV to {csv_path}: {e}", file=sys.stderr)  # noqa: T201
            # Don't fail the whole script if CSV export fails

    return 0


if __name__ == "__main__":
    sys.exit(main())
