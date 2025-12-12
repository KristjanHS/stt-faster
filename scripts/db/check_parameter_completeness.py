#!/usr/bin/env python3
"""Check if all parameter values and statistics are inserted at job and file levels.

This script verifies that recent database rows have complete data:
- Job level (runs table): all parameter values and aggregated statistics
- File level (file_metrics table): all parameter values and per-file statistics

Usage:
    .venv/bin/python scripts/db/check_parameter_completeness.py
    .venv/bin/python scripts/db/check_parameter_completeness.py --limit 5
    .venv/bin/python scripts/db/check_parameter_completeness.py --run-id 42
"""
# ruff: noqa: T201  # CLI script - print statements are intentional

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.database import TranscriptionDatabase
from backend.variants.registry import _get_all_variants

# Expected parameter columns for runs table (job level)
RUN_PARAMETER_COLUMNS = {
    # Configuration
    "preset",
    "language",
    "preprocess_enabled",
    "preprocess_profile",
    "target_sample_rate",
    "target_channels",
    "loudnorm_preset",
    # Preprocessing parameters (job level only)
    "volume_adjustment_db",
    "resampler",
    "sample_format",
    "loudnorm_target_i",
    "loudnorm_target_tp",
    "loudnorm_target_lra",
    "loudnorm_backend",
    "denoise_method",
    "denoise_library",
    "rnnoise_model",
    "rnnoise_mix",
    "snr_estimation_method",
    # Model
    "model_id",
    "device",
    "compute_type",
    # Transcription parameters
    "beam_size",
    "patience",
    "word_timestamps",
    "task",
    "chunk_length",
    "vad_filter",
    "vad_threshold",
    "vad_min_speech_duration_ms",
    "vad_max_speech_duration_s",
    "vad_min_silence_duration_ms",
    "vad_speech_pad_ms",
    "temperature",
    "temperature_increment_on_fallback",
    "best_of",
    "compression_ratio_threshold",
    "logprob_threshold",
    "no_speech_threshold",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "suppress_tokens",
    "condition_on_previous_text",
    "initial_prompt",
}

# Expected statistics columns for runs table (job level)
RUN_STATISTICS_COLUMNS = {
    "files_found",
    "succeeded",
    "failed",
    "total_processing_time",
    "total_preprocess_time",
    "total_transcribe_time",
    "total_audio_duration",
    "speed_ratio",
}

# Expected parameter columns for file_metrics table (file level)
# Note: Transcription and preprocessing parameters are only checked at job level, not file level
# File level only checks output parameters and statistics
FILE_PARAMETER_COLUMNS = {
    # Output parameters (file-specific)
    "output_format",
    "float_precision",
}

# Expected statistics columns for file_metrics table (file level)
FILE_STATISTICS_COLUMNS = {
    "status",
    "audio_duration",
    "total_processing_time",
    "transcribe_duration",
    "preprocess_duration",
    "speed_ratio",
    "input_channels",
    "input_sample_rate",
    "input_format",
    "requested_language",
    "applied_language",
    "detected_language",
    "language_probability",
}

# Optional statistics that may legitimately be NULL
OPTIONAL_FILE_STATISTICS = {
    "preprocess_snr_before",  # May be None if SNR estimation wasn't performed
    "preprocess_snr_after",  # May be None if SNR estimation wasn't performed
}


def get_variant_explicit_parameters(variant) -> set[str]:
    """Get set of parameters that were explicitly set and passed to model.transcribe().

    Uses the same approach as verify_all_variants.py - checks what to_kwargs() returns,
    which represents what was actually passed to model.transcribe().

    Args:
        variant: Variant instance

    Returns:
        Set of parameter names that were explicitly set and passed to model.transcribe()
    """
    config = variant.transcription_config

    # Prioritize to_kwargs() - this is what actually gets passed to model.transcribe()
    # (same approach as verify_all_variants.py)
    if hasattr(config, "to_kwargs"):
        kwargs = config.to_kwargs()
        explicit = set(kwargs.keys())

        # Handle VAD parameters - they're stored in a dict but map to individual DB columns
        if "vad_parameters" in kwargs and isinstance(kwargs.get("vad_parameters"), dict):
            vad_params = kwargs["vad_parameters"]
            if vad_params:
                # If vad_parameters dict was set, the individual VAD params were used
                explicit.add("vad_min_speech_duration_ms")
                explicit.add("vad_max_speech_duration_s")
                explicit.add("vad_min_silence_duration_ms")
                explicit.add("vad_speech_pad_ms")
        # vad_threshold might be separate or merged into vad_parameters
        if "vad_threshold" in kwargs or "vad_filter" in kwargs:
            explicit.add("vad_threshold")

        return explicit

    # Fallback: use _explicit_fields if to_kwargs() is not available
    if hasattr(config, "_explicit_fields"):
        explicit = config._explicit_fields.copy()
        # Handle VAD parameters dict
        if "vad_parameters" in explicit or "vad_filter" in explicit:
            if hasattr(config, "vad_parameters") and config.vad_parameters:
                explicit.add("vad_min_speech_duration_ms")
                explicit.add("vad_max_speech_duration_s")
                explicit.add("vad_min_silence_duration_ms")
                explicit.add("vad_speech_pad_ms")
        return explicit

    # Last fallback: assume all parameters from TranscriptionConfig.from_env()
    # This handles the case where the main transcribe function is used (not variants)
    return set(RUN_PARAMETER_COLUMNS)


def find_matching_variant(run: dict) -> tuple[Any | None, set[str]]:
    """Try to find which variant matches this run by checking all variants.

    Args:
        run: Run record from database

    Returns:
        Tuple of (variant, explicit_parameters_set)
        If no variant matches, returns (None, set()) and we assume all parameters should be set
    """
    all_variants = _get_all_variants()

    # Try to match against all variants
    # We check if the run's non-NULL values match what the variant would set
    for variant in all_variants:
        explicit_params = get_variant_explicit_parameters(variant)
        # Check if the run's non-NULL values match what the variant would set
        matches = True
        for param in explicit_params:
            if param in run and run[param] is not None:
                # Parameter was set - check if it matches variant's value
                variant_value = getattr(variant.transcription_config, param, None)
                run_value = run[param]
                # Handle special cases (temperature as JSON string, booleans as ints, etc.)
                if param == "temperature" and variant_value is not None:
                    import json

                    if isinstance(variant_value, list):
                        variant_value_str = json.dumps(variant_value)
                    else:
                        variant_value_str = str(variant_value)
                    if run_value != variant_value_str:
                        matches = False
                        break
                elif isinstance(variant_value, bool) and isinstance(run_value, int):
                    if bool(run_value) != variant_value:
                        matches = False
                        break
                elif run_value != variant_value:
                    matches = False
                    break

        if matches:
            return variant, explicit_params

    # No variant matched - assume it's the main transcribe function which sets all parameters
    # (from TranscriptionConfig.from_env())
    return None, set(RUN_PARAMETER_COLUMNS)


def check_null_values(
    row: dict, expected_columns: set[str], level: str, explicit_params: set[str] | None = None
) -> list[str]:
    """Check which expected columns have NULL values.

    Args:
        row: Database row as dictionary
        expected_columns: Set of column names to check
        level: Level name for reporting (e.g., "job" or "file")
        explicit_params: Optional set of parameters that were explicitly set in the variant.
                        If provided, only check these parameters. If None, check all.

    Returns:
        List of column names that are NULL (only for parameters that were explicitly set)
    """
    missing = []
    # If explicit_params is provided (even if empty), only check those parameters
    # Empty set means baseline variant - no parameters expected, so check nothing
    if explicit_params is not None:
        columns_to_check = explicit_params & expected_columns
    else:
        # None means check all (main transcribe function)
        columns_to_check = expected_columns

    for col in columns_to_check:
        if col not in row:
            missing.append(f"{col} (column missing)")
        elif row[col] is None:
            missing.append(col)
    return missing


def check_run_completeness(db: TranscriptionDatabase, run_id: int) -> tuple[bool, dict]:
    """Check if a run has all parameters and statistics populated.

    Args:
        db: Database connection
        run_id: Run ID to check

    Returns:
        Tuple of (is_complete, details_dict)
    """
    run = db.get_run_by_id(run_id)
    if not run:
        return False, {"error": f"Run {run_id} not found"}

    # Try to find matching variant to determine which parameters should be set
    variant, explicit_params = find_matching_variant(run)

    details = {
        "run_id": run_id,
        "preset": run.get("preset"),
        "recorded_at": run.get("recorded_at"),
        "variant_name": variant.name if variant else "main_transcribe",
        "variant_number": variant.number if variant else None,
        "explicit_parameters": sorted(explicit_params) if explicit_params else None,
        "missing_parameters": [],
        "missing_statistics": [],
    }

    # Check parameters - only validate those that were explicitly set in the variant
    missing_params = check_null_values(run, RUN_PARAMETER_COLUMNS, "job", explicit_params)
    details["missing_parameters"] = missing_params

    # Check statistics (always check all statistics)
    missing_stats = check_null_values(run, RUN_STATISTICS_COLUMNS, "job", None)
    details["missing_statistics"] = missing_stats

    is_complete = len(missing_params) == 0 and len(missing_stats) == 0
    details["complete"] = is_complete

    return is_complete, details


def check_file_metrics_completeness(db: TranscriptionDatabase, run_id: int) -> tuple[bool, dict]:
    """Check if file metrics for a run have all parameters and statistics populated.

    Note: Transcription and preprocessing parameters are only checked at job level.
    File level only checks output parameters and statistics.

    Args:
        db: Database connection
        run_id: Run ID to check

    Returns:
        Tuple of (is_complete, details_dict)
    """
    if not db.conn:
        return False, {"error": "Database connection not initialized"}

    # Get all file metrics for this run
    cursor = db.conn.execute("SELECT * FROM file_metrics WHERE run_id = ? ORDER BY id", [run_id])
    rows = cursor.fetchall()

    if not rows:
        return True, {
            "run_id": run_id,
            "file_count": 0,
            "complete": True,
            "message": "No file metrics found for this run",
        }

    columns = [desc[0] for desc in cursor.description]
    file_metrics = [dict(zip(columns, row)) for row in rows]

    details = {
        "run_id": run_id,
        "file_count": len(file_metrics),
        "files_checked": [],
        "all_complete": True,
    }

    # Check each file
    for i, file_metric in enumerate(file_metrics):
        file_details = {
            "file_index": i + 1,
            "audio_path": file_metric.get("audio_path"),
            "status": file_metric.get("status"),
            "missing_parameters": [],
            "missing_statistics": [],
        }

        # Check parameters - only output parameters at file level
        # Transcription and preprocessing parameters are only checked at job level, not file level
        missing_params = check_null_values(file_metric, FILE_PARAMETER_COLUMNS, "file", None)
        file_details["missing_parameters"] = missing_params

        # Check statistics - exclude optional ones that may legitimately be NULL
        missing_stats = check_null_values(file_metric, FILE_STATISTICS_COLUMNS, "file", None)
        # Filter out optional statistics that are expected to be NULL in some cases
        missing_stats = [s for s in missing_stats if s not in OPTIONAL_FILE_STATISTICS]
        file_details["missing_statistics"] = missing_stats

        file_complete = len(missing_params) == 0 and len(missing_stats) == 0
        file_details["complete"] = file_complete

        if not file_complete:
            details["all_complete"] = False

        details["files_checked"].append(file_details)

    details["complete"] = details["all_complete"]
    return details["complete"], details


def print_run_report(run_details: dict) -> None:
    """Print a formatted report for a run check."""
    print(f"\n{'=' * 80}")
    print(f"RUN ID: {run_details['run_id']}")
    print(f"Preset: {run_details.get('preset', 'N/A')}")
    print(f"Recorded at: {run_details.get('recorded_at', 'N/A')}")
    print(f"{'=' * 80}")

    if "error" in run_details:
        print(f"❌ ERROR: {run_details['error']}")
        return

    # Job level check
    print("\n📊 JOB LEVEL (runs table):")
    variant_name = run_details.get("variant_name", "unknown")
    variant_number = run_details.get("variant_number")
    explicit_params = run_details.get("explicit_parameters", [])

    if variant_number:
        print(f"  Detected variant: {variant_number} ({variant_name})")
        if explicit_params:
            print(f"  Parameters expected to be set: {len(explicit_params)}")
        else:
            print("  Baseline variant - no transcription parameters explicitly set (using library defaults)")
    else:
        print("  Using main transcribe function (all parameters should be set)")

    missing_params = run_details.get("missing_parameters", [])
    missing_stats = run_details.get("missing_statistics", [])

    if missing_params:
        print(f"  ⚠️  Missing parameters ({len(missing_params)}):")
        for param in missing_params:
            print(f"     - {param}")
    else:
        if explicit_params:
            print(f"  ✅ All expected parameters present ({len(explicit_params)} checked)")
        elif variant_number:
            print("  ✅ Baseline variant - no parameters expected (all NULL values are expected)")
        else:
            print("  ✅ All parameters present")

    if missing_stats:
        print(f"  ⚠️  Missing statistics ({len(missing_stats)}):")
        for stat in missing_stats:
            print(f"     - {stat}")
    else:
        print("  ✅ All statistics present")

    if run_details.get("complete"):
        print("  ✅ JOB LEVEL: COMPLETE")
    else:
        print("  ❌ JOB LEVEL: INCOMPLETE")


def print_file_metrics_report(file_details: dict) -> None:
    """Print a formatted report for file metrics check."""
    print("\n📁 FILE LEVEL (file_metrics table):")
    print(f"  Total files: {file_details['file_count']}")

    if file_details["file_count"] == 0:
        print("  ℹ️  No file metrics to check")
        return

    incomplete_count = 0
    for file_info in file_details["files_checked"]:
        if not file_info.get("complete", False):
            incomplete_count += 1
            print(f"\n  📄 File {file_info['file_index']}: {file_info['audio_path']}")
            print(f"     Status: {file_info['status']}")

            missing_params = file_info.get("missing_parameters", [])
            if missing_params:
                print(f"     ⚠️  Missing parameters ({len(missing_params)}):")
                for param in missing_params[:10]:  # Show first 10
                    print(f"        - {param}")
                if len(missing_params) > 10:
                    print(f"        ... and {len(missing_params) - 10} more")

            missing_stats = file_info.get("missing_statistics", [])
            if missing_stats:
                print(f"     ⚠️  Missing statistics ({len(missing_stats)}):")
                for stat in missing_stats[:10]:  # Show first 10
                    print(f"        - {stat}")
                if len(missing_stats) > 10:
                    print(f"        ... and {len(missing_stats) - 10} more")

    if incomplete_count == 0:
        print("  ✅ All files have complete parameters and statistics")
    else:
        print(f"\n  ⚠️  {incomplete_count} out of {file_details['file_count']} files are incomplete")

    if file_details.get("complete"):
        print("  ✅ FILE LEVEL: COMPLETE")
    else:
        print("  ❌ FILE LEVEL: INCOMPLETE")


def check_recent_runs(limit: int = 5) -> int:
    """Check recent runs for parameter and statistics completeness.

    Args:
        limit: Number of recent runs to check

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        db = TranscriptionDatabase()
        runs = db.get_run_history(limit=limit)

        if not runs:
            print("No runs found in database")
            db.close()
            return 0

        print(f"Checking {len(runs)} recent run(s) for parameter and statistics completeness...")

        all_complete = True
        for run in runs:
            run_id = run.get("id")
            if not run_id:
                continue

            # Check job level
            job_complete, job_details = check_run_completeness(db, run_id)
            print_run_report(job_details)

            # Check file level - only checks output parameters and statistics
            # (transcription and preprocessing parameters are checked at job level only)
            file_complete, file_details = check_file_metrics_completeness(db, run_id)
            print_file_metrics_report(file_details)

            if not job_complete or not file_complete:
                all_complete = False

        db.close()

        print(f"\n{'=' * 80}")
        if all_complete:
            print("✅ ALL CHECKS PASSED: All recent runs have complete data")
            return 0
        else:
            print("❌ SOME CHECKS FAILED: Some runs are missing parameters or statistics")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def check_specific_run(run_id: int) -> int:
    """Check a specific run for parameter and statistics completeness.

    Args:
        run_id: Run ID to check

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        db = TranscriptionDatabase()

        # Check job level
        job_complete, job_details = check_run_completeness(db, run_id)
        print_run_report(job_details)

        # Check file level - only checks output parameters and statistics
        # (transcription and preprocessing parameters are checked at job level only)
        file_complete, file_details = check_file_metrics_completeness(db, run_id)
        print_file_metrics_report(file_details)

        db.close()

        print(f"\n{'=' * 80}")
        if job_complete and file_complete:
            print(f"✅ RUN {run_id}: COMPLETE")
            return 0
        else:
            print(f"❌ RUN {run_id}: INCOMPLETE")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check if all parameter values and statistics are inserted at job and file levels"
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=5,
        help="Number of recent runs to check (default: 5)",
    )
    parser.add_argument(
        "--run-id",
        "-r",
        type=int,
        default=None,
        help="Check a specific run by ID instead of recent runs",
    )
    args = parser.parse_args()

    if args.run_id:
        return check_specific_run(args.run_id)
    else:
        return check_recent_runs(args.limit)


if __name__ == "__main__":
    sys.exit(main())
