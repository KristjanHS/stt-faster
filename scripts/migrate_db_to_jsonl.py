#!/usr/bin/env python3
"""One-shot exporter: DuckDB ``transcribe_state.duckdb`` → ``runs.jsonl``.

Reads every row of ``runs`` (wide), LEFT JOINs ``run_configs`` and
``run_metrics`` (the narrow newer tables — they win on conflict), and embeds
``file_metrics`` rows per run as a ``files[]`` array. ``run_parameters`` is
used only to backfill values not already covered by the wide ``runs`` row or
the narrow tables (older runs predate ``run_configs`` / ``run_metrics``).

The ``transcriptions`` table is intentionally skipped — confirmed empty in
production per ``docs/plans/2026-05-20-stage-G-db-to-jsonl.md`` §2.

WAL note (plan §5 R-6): if WAL replay fails on open, move the ``.wal`` file
aside (``mv transcribe_state.duckdb.wal transcribe_state.duckdb.wal.bak``)
before running this script.

Output is byte-deterministic: ``json.dumps`` with ``sort_keys=True`` and
``separators=(",", ":")`` at every encode boundary. Re-running against the
same DB yields identical bytes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from backend.database.schema import convert_parameter_value, get_default_db_path

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-set definitions (kept local so the script stays self-contained)
# ---------------------------------------------------------------------------

# Preprocessing keys grouped at the run level.
_PREPROCESS_FLAT_KEYS = (
    "enabled",  # mapped from preprocess_enabled
    "profile",
    "target_sample_rate",
    "target_channels",
    "snr_estimation_method",
    "volume_adjustment_db",
    "resampler",
    "sample_format",
)
_LOUDNORM_KEYS = (
    "preset",  # mapped from loudnorm_preset
    "target_i",  # loudnorm_target_i
    "target_tp",  # loudnorm_target_tp
    "target_lra",  # loudnorm_target_lra
    "backend",  # loudnorm_backend
)
_DENOISE_KEYS = (
    "method",  # denoise_method
    "library",  # denoise_library
    "rnnoise_model",
    "rnnoise_mix",
)

# Transcription params placed under ``params`` (with VAD sub-object).
_PARAM_KEYS = (
    "beam_size",
    "best_of",
    "patience",
    "task",
    "chunk_length",
    "word_timestamps",
    "temperature",
    "temperature_increment_on_fallback",
    "compression_ratio_threshold",
    "logprob_threshold",
    "no_speech_threshold",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "suppress_tokens",
    "condition_on_previous_text",
    "initial_prompt",
)
_VAD_KEY_MAP = {
    "filter": "vad_filter",
    "threshold": "vad_threshold",
    "min_speech_ms": "vad_min_speech_duration_ms",
    "max_speech_s": "vad_max_speech_duration_s",
    "min_silence_ms": "vad_min_silence_duration_ms",
    "speech_pad_ms": "vad_speech_pad_ms",
}


def _row_to_dict(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Materialise the last-executed cursor as a list of column-keyed dicts."""
    description = cursor.description or []
    column_names = [desc[0] for desc in description]
    rows = cursor.fetchall()
    return [dict(zip(column_names, row, strict=False)) for row in rows]


def _iso_utc(value: Any) -> str | None:
    """Render a timestamp as ISO-8601 UTC with ``Z`` suffix.

    Accepts ``datetime`` (naive → treated as UTC; aware → converted to UTC)
    or a pre-formatted string (passed through). Returns ``None`` for ``None``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            dt = value.replace(tzinfo=timezone.utc)
        else:
            dt = value.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        # Best-effort: if it parses, normalise; otherwise pass through.
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return str(value)


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return bool(value)


def _fill_missing(target: dict[str, Any], key: str, value: Any) -> None:
    """Set ``target[key] = value`` only if the current slot is missing/None."""
    if value is None:
        return
    if target.get(key) is None:
        target[key] = value


def _load_run_parameters(conn: duckdb.DuckDBPyConnection, run_id: int) -> dict[tuple[str, str], Any]:
    """Fetch the ``run_parameters`` k/v rows for a single run, converted to
    their original types via :func:`convert_parameter_value`. Keyed by
    ``(category, name)``."""
    cursor = conn.execute(
        "SELECT category, name, value, value_type FROM run_parameters WHERE run_id = ? ORDER BY category, name",
        [run_id],
    )
    out: dict[tuple[str, str], Any] = {}
    for category, name, value, value_type in cursor.fetchall():
        if value is None:
            continue
        out[(category, name)] = convert_parameter_value(value, value_type)
    return out


def _file_record_from_row(file_row: dict[str, Any]) -> dict[str, Any]:
    """Map a single ``file_metrics`` row into the nested per-file JSON shape.

    Mirrors the run-level nesting so per-file overrides are visible. Keeps
    ``segment_count`` as ``None`` — the source schema doesn't track it yet
    (the field is reserved for future runs).
    """
    preprocess: dict[str, Any] = {
        "enabled": _coerce_bool(file_row.get("preprocess_enabled")),
        "profile": file_row.get("preprocess_profile"),
        "target_sample_rate": file_row.get("target_sample_rate"),
        "target_channels": file_row.get("target_channels"),
        "snr_estimation_method": file_row.get("snr_estimation_method"),
        "snr_before": file_row.get("preprocess_snr_before"),
        "snr_after": file_row.get("preprocess_snr_after"),
        "volume_adjustment_db": file_row.get("volume_adjustment_db"),
        "resampler": file_row.get("resampler"),
        "sample_format": file_row.get("sample_format"),
        "loudnorm": {
            "preset": file_row.get("loudnorm_preset"),
            "target_i": file_row.get("loudnorm_target_i"),
            "target_tp": file_row.get("loudnorm_target_tp"),
            "target_lra": file_row.get("loudnorm_target_lra"),
            "backend": file_row.get("loudnorm_backend"),
        },
        "denoise": {
            "method": file_row.get("denoise_method"),
            "library": file_row.get("denoise_library"),
            "rnnoise_model": file_row.get("rnnoise_model"),
            "rnnoise_mix": file_row.get("rnnoise_mix"),
        },
        "input": {
            "channels": file_row.get("input_channels"),
            "sample_rate": file_row.get("input_sample_rate"),
            "format": file_row.get("input_format"),
        },
    }

    model: dict[str, Any] = {
        "id": file_row.get("model_id"),
        "device": file_row.get("device"),
        "compute_type": file_row.get("compute_type"),
    }

    params: dict[str, Any] = {key: file_row.get(key) for key in _PARAM_KEYS}
    params["vad"] = {json_key: file_row.get(src_key) for json_key, src_key in _VAD_KEY_MAP.items()}
    # Booleans in DuckDB sometimes come back as ints; normalise the obvious ones.
    if params.get("word_timestamps") is not None:
        params["word_timestamps"] = _coerce_bool(params["word_timestamps"])
    if params.get("condition_on_previous_text") is not None:
        params["condition_on_previous_text"] = _coerce_bool(params["condition_on_previous_text"])
    if params["vad"].get("filter") is not None:
        params["vad"]["filter"] = _coerce_bool(params["vad"]["filter"])

    record: dict[str, Any] = {
        "path": file_row.get("audio_path"),
        "status": file_row.get("status"),
        "audio_duration_s": file_row.get("audio_duration"),
        "preprocess_time_s": file_row.get("preprocess_duration"),
        "transcribe_time_s": file_row.get("transcribe_duration"),
        "total_processing_time_s": file_row.get("total_processing_time"),
        "speed_ratio": file_row.get("speed_ratio"),
        # segment_count is not in the file_metrics schema; reserved for future runs.
        "segment_count": None,
        "requested_language": file_row.get("requested_language"),
        "applied_language": file_row.get("applied_language"),
        "detected_language": file_row.get("detected_language"),
        "language_probability": file_row.get("language_probability"),
        "error_message": file_row.get("error_message"),
        "preprocess": preprocess,
        "model": model,
        "params": params,
        "output_format": file_row.get("output_format"),
        "float_precision": file_row.get("float_precision"),
    }
    return record


def _build_run_record(
    run_row: dict[str, Any],
    file_rows: list[dict[str, Any]],
    parameters: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    """Assemble one JSONL record from the joined runs/run_configs/run_metrics
    row, child ``file_metrics`` rows, and the per-run k/v ``run_parameters``
    backfill map."""

    # --- model ------------------------------------------------------------
    model: dict[str, Any] = {
        "id": run_row.get("model_id"),
        "device": run_row.get("device"),
        "compute_type": run_row.get("compute_type"),
    }
    # Backfill from run_parameters where missing (older runs).
    _fill_missing(model, "id", parameters.get(("model", "model_id")))
    _fill_missing(model, "device", parameters.get(("model", "device")))
    _fill_missing(model, "compute_type", parameters.get(("model", "compute_type")))

    # --- preprocess -------------------------------------------------------
    preprocess: dict[str, Any] = {
        "enabled": _coerce_bool(run_row.get("preprocess_enabled")),
        "profile": run_row.get("preprocess_profile"),
        "target_sample_rate": run_row.get("target_sample_rate"),
        "target_channels": run_row.get("target_channels"),
        "snr_estimation_method": run_row.get("snr_estimation_method"),
        "volume_adjustment_db": run_row.get("volume_adjustment_db"),
        "resampler": run_row.get("resampler"),
        "sample_format": run_row.get("sample_format"),
        "loudnorm": {
            "preset": run_row.get("loudnorm_preset"),
            "target_i": run_row.get("loudnorm_target_i"),
            "target_tp": run_row.get("loudnorm_target_tp"),
            "target_lra": run_row.get("loudnorm_target_lra"),
            "backend": run_row.get("loudnorm_backend"),
        },
        "denoise": {
            "method": run_row.get("denoise_method"),
            "library": run_row.get("denoise_library"),
            "rnnoise_model": run_row.get("rnnoise_model"),
            "rnnoise_mix": run_row.get("rnnoise_mix"),
        },
    }
    # Backfill preprocess group from run_parameters.
    # The category "preprocess" with name == "profile" maps to preprocess.profile;
    # other names map directly into the top-level group or its sub-objects.
    for (category, name), pval in parameters.items():
        if category != "preprocess":
            continue
        if name == "profile":
            _fill_missing(preprocess, "profile", pval)
        elif name in _PREPROCESS_FLAT_KEYS:
            _fill_missing(preprocess, name, pval)
        elif name == "loudnorm_preset":
            _fill_missing(preprocess["loudnorm"], "preset", pval)
        elif name == "loudnorm_target_i":
            _fill_missing(preprocess["loudnorm"], "target_i", pval)
        elif name == "loudnorm_target_tp":
            _fill_missing(preprocess["loudnorm"], "target_tp", pval)
        elif name == "loudnorm_target_lra":
            _fill_missing(preprocess["loudnorm"], "target_lra", pval)
        elif name == "loudnorm_backend":
            _fill_missing(preprocess["loudnorm"], "backend", pval)
        elif name == "denoise_method":
            _fill_missing(preprocess["denoise"], "method", pval)
        elif name == "denoise_library":
            _fill_missing(preprocess["denoise"], "library", pval)
        elif name == "rnnoise_model":
            _fill_missing(preprocess["denoise"], "rnnoise_model", pval)
        elif name == "rnnoise_mix":
            _fill_missing(preprocess["denoise"], "rnnoise_mix", pval)
        # Unknown names within "preprocess": pass through under the top group
        # to avoid silent loss.
        else:
            _fill_missing(preprocess, name, pval)

    # --- params (transcription config) -----------------------------------
    params: dict[str, Any] = {key: run_row.get(key) for key in _PARAM_KEYS}
    params["vad"] = {json_key: run_row.get(src_key) for json_key, src_key in _VAD_KEY_MAP.items()}
    if params.get("word_timestamps") is not None:
        params["word_timestamps"] = _coerce_bool(params["word_timestamps"])
    if params.get("condition_on_previous_text") is not None:
        params["condition_on_previous_text"] = _coerce_bool(params["condition_on_previous_text"])
    if params["vad"].get("filter") is not None:
        params["vad"]["filter"] = _coerce_bool(params["vad"]["filter"])

    # Backfill from run_parameters where missing.
    for (category, name), pval in parameters.items():
        if category != "transcription":
            continue
        if name in _PARAM_KEYS:
            _fill_missing(params, name, pval)
        elif name in _VAD_KEY_MAP:
            # legacy name path: name matches the source column name; map back.
            json_key = next(jk for jk, src in _VAD_KEY_MAP.items() if src == name)
            _fill_missing(params["vad"], json_key, pval)
        else:
            # Some legacy runs stored vad_* under the JSON key directly.
            if name in {"filter", "threshold", "min_speech_ms", "max_speech_s", "min_silence_ms", "speech_pad_ms"}:
                _fill_missing(params["vad"], name, pval)
            else:
                _fill_missing(params, name, pval)

    # --- totals -----------------------------------------------------------
    totals: dict[str, Any] = {
        "files_found": run_row.get("files_found"),
        "succeeded": run_row.get("succeeded"),
        "failed": run_row.get("failed"),
        "processing_time_s": run_row.get("total_processing_time"),
        "preprocess_time_s": run_row.get("total_preprocess_time"),
        "transcribe_time_s": run_row.get("total_transcribe_time"),
        "audio_duration_s": run_row.get("total_audio_duration"),
        "speed_ratio": run_row.get("speed_ratio"),
    }

    files = [_file_record_from_row(fr) for fr in file_rows]

    record: dict[str, Any] = {
        "id": run_row.get("id"),
        "recorded_at": _iso_utc(run_row.get("recorded_at")),
        "input_folder": run_row.get("input_folder"),
        "preset": run_row.get("preset"),
        "language": run_row.get("language"),
        "preprocess": preprocess,
        "model": model,
        "params": params,
        "totals": totals,
        "files": files,
    }
    return record


def _fetch_runs(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Return one wide row per run with newer narrow tables joined.

    SELECTs every column from ``runs`` (wide) via ``r.*``, then layers
    ``run_configs`` and ``run_metrics`` columns with ``COALESCE``-style
    precedence (newer narrow data wins). Done in two passes — first the
    wide row, then a per-run lookup — so we never null-stomp a wide-row
    value when the narrow row is missing (plan §5 R-1).
    """
    cursor = conn.execute("SELECT * FROM runs ORDER BY id ASC")
    runs = _row_to_dict(cursor)

    for run in runs:
        run_id = run.get("id")
        if run_id is None:
            continue
        cfg_cursor = conn.execute(
            "SELECT model_id, device, compute_type FROM run_configs WHERE run_id = ?",
            [run_id],
        )
        cfg_rows = _row_to_dict(cfg_cursor)
        if cfg_rows:
            cfg = cfg_rows[0]
            # Narrow table wins on conflict, but only when its value is non-null.
            for key in ("model_id", "device", "compute_type"):
                if cfg.get(key) is not None:
                    run[key] = cfg[key]
                elif key not in run:
                    run[key] = None

        metrics_cursor = conn.execute(
            "SELECT total_preprocess_time, total_transcribe_time FROM run_metrics WHERE run_id = ?",
            [run_id],
        )
        metrics_rows = _row_to_dict(metrics_cursor)
        if metrics_rows:
            metrics = metrics_rows[0]
            for key in ("total_preprocess_time", "total_transcribe_time"):
                if metrics.get(key) is not None:
                    run[key] = metrics[key]
                elif key not in run:
                    run[key] = None
    return runs


def _fetch_file_metrics(conn: duckdb.DuckDBPyConnection, run_id: int) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT * FROM file_metrics WHERE run_id = ? ORDER BY id ASC",
        [run_id],
    )
    return _row_to_dict(cursor)


def _encode_records(records: Iterable[dict[str, Any]]) -> bytes:
    """Encode each record on its own line with stable key ordering."""
    encoded_lines: list[str] = []
    for record in records:
        encoded_lines.append(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return ("\n".join(encoded_lines) + "\n").encode("utf-8") if encoded_lines else b""


def export(
    db_path: Path,
    output: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, str | None, str | None]:
    """Run the export. Returns ``(run_count, byte_count, earliest, latest)``.

    When ``dry_run`` is True the file is NOT written.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        runs = _fetch_runs(conn)
        out_records: list[dict[str, Any]] = []
        for run in runs:
            run_id = run.get("id")
            if not isinstance(run_id, int):
                LOGGER.warning("Skipping run row with non-integer id: %r", run_id)
                continue
            file_rows = _fetch_file_metrics(conn, run_id)
            parameters = _load_run_parameters(conn, run_id)
            out_records.append(_build_run_record(run, file_rows, parameters))
    finally:
        conn.close()

    data = _encode_records(out_records)
    earliest: str | None = None
    latest: str | None = None
    if out_records:
        timestamps: list[str] = [r["recorded_at"] for r in out_records if isinstance(r.get("recorded_at"), str)]
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(output.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, output)

    return len(out_records), len(data), earliest, latest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the DuckDB file (default: ~/.local/share/stt-faster/transcribe_state.duckdb).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: ~/.local/share/stt-faster/runs.jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records but do not write the output file; print summary only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = _parse_args(argv)

    db_path = args.db_path or get_default_db_path()
    output = args.output or (Path.home() / ".local" / "share" / "stt-faster" / "runs.jsonl")
    if args.output is None:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            output = Path(xdg) / "stt-faster" / "runs.jsonl"

    n, byte_count, earliest, latest = export(db_path, output, dry_run=args.dry_run)
    print(  # noqa: T201
        f"{n} runs exported, {byte_count} bytes, earliest={earliest}, latest={latest}",
        file=sys.stderr,
    )
    if args.dry_run:
        print("(dry-run: no file written)", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
