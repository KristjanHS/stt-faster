"""Round-trip guardrail for the runs-schema normalization migration.

Stage D prerequisite (audit §2.1): exercises the database lifecycle that depends on
`_migration_006_normalize_runs_schema` so the migration file can be moved into
`backend/database/migrations/migration_006.py` untouched in Stage D.2 without silent
breakage.

What this test covers:
- A fresh `TranscriptionDatabase` initializes its schema (the normalized-tables path
  exercised in production for new DBs) and accepts a `record_run` write.
- Reopening the same DB file re-runs `_init_db` and the full `MIGRATIONS` list against
  an already-migrated database, and the recorded run is recoverable via
  `get_run_by_id` and `get_run_history`. This is the path every production restart
  takes, and it is gated on every migration being idempotent on a populated DB.

What this test does NOT cover:
The migration 006 body's wide-row data-migration loop is not directly exercised here.
A pre-006 synthetic fixture exposed a latent bug in that loop (the `run_parameters`
table is created without a sequence default, so the INSERT at database.py:789 fails
with a NOT NULL constraint). In production the first guard at database.py:477
short-circuits before reaching that INSERT, so the bug never fires — but it means the
wide-row data-migration path is effectively dead code. Fixing that is explicitly out
of scope for Stage D (the audit guardrail says "move migration_006 untouched"); a
separate PR with a synthetic-wide-shape test should land the body fix.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from backend.database import (
    MIGRATIONS,
    RunRecord,
    TranscriptionDatabase,
)


def _build_record(recorded_at: datetime) -> RunRecord:
    return RunRecord(
        recorded_at=recorded_at,
        input_folder="/data/audio",
        preset="turbo",
        language="en",
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        loudnorm_preset="ebu",
        volume_adjustment_db=-2.0,
        resampler="soxr",
        sample_format="s16",
        loudnorm_target_i=-23.0,
        loudnorm_target_tp=-2.0,
        loudnorm_target_lra=7.0,
        loudnorm_backend="ffmpeg",
        denoise_method="rnnoise",
        denoise_library="rnnoise",
        rnnoise_model="models/sh.rnnn",
        rnnoise_mix=0.6,
        snr_estimation_method="estimate_snr_db",
        model_id="base",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        patience=1.0,
        word_timestamps=True,
        task="transcribe",
        chunk_length=30,
        vad_filter=True,
        vad_threshold=0.5,
        vad_min_speech_duration_ms=250,
        vad_max_speech_duration_s=30.0,
        vad_min_silence_duration_ms=100,
        vad_speech_pad_ms=400,
        temperature="[0.0,0.2,0.4]",
        temperature_increment_on_fallback=0.2,
        best_of=5,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        length_penalty=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        suppress_tokens="-1",
        condition_on_previous_text=True,
        initial_prompt="Hello",
        files_found=3,
        succeeded=2,
        failed=1,
        total_processing_time=90.5,
        total_preprocess_time=20.5,
        total_transcribe_time=70.0,
        total_audio_duration=180.0,
        speed_ratio=2.0,
    )


def test_run_round_trips_through_reopen(tmp_path: Path) -> None:
    """A run written to a fresh DB is recoverable after closing and re-running migrations."""
    db_path = tmp_path / "round_trip.duckdb"
    recorded_at = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)
    expected = _build_record(recorded_at)

    db = TranscriptionDatabase(str(db_path))
    try:
        run_id = db.record_run(expected)
        assert run_id > 0
    finally:
        assert db.conn is not None
        db.conn.close()

    # Reopening re-runs _init_db and the full MIGRATIONS list on a populated DB,
    # so this asserts every migration is idempotent against post-006 schema.
    reopened = TranscriptionDatabase(str(db_path))
    try:
        history = reopened.get_run_history()
        assert len(history) == 1, "round-trip: history must contain the recorded run"

        reloaded = reopened.get_run_by_id(run_id)
        assert reloaded is not None
        assert reloaded["preset"] == expected.preset
        assert reloaded["language"] == expected.language
        assert reloaded["files_found"] == expected.files_found
        assert reloaded["succeeded"] == expected.succeeded
        assert reloaded["failed"] == expected.failed
        assert reloaded["model_id"] == expected.model_id
        assert reloaded["device"] == expected.device
        assert reloaded["compute_type"] == expected.compute_type
        assert reloaded["total_preprocess_time"] == expected.total_preprocess_time
        assert reloaded["total_transcribe_time"] == expected.total_transcribe_time
        # Parameters reconstructed from run_parameters
        assert reloaded["beam_size"] == expected.beam_size
        assert reloaded["task"] == expected.task
        assert reloaded["loudnorm_preset"] == expected.loudnorm_preset
        assert reloaded["rnnoise_mix"] == expected.rnnoise_mix
        assert reloaded["initial_prompt"] == expected.initial_prompt
    finally:
        assert reopened.conn is not None
        reopened.conn.close()


def test_migration_006_in_registered_migrations() -> None:
    """Migration 006 is in MIGRATIONS — D.2 must preserve the registration after splitting the file."""
    versions = [m.version for m in MIGRATIONS]
    assert 6 in versions, "migration 006 must remain registered"
    migration_six = next(m for m in MIGRATIONS if m.version == 6)
    assert "normalize" in migration_six.name.lower() or "runs" in migration_six.name.lower()
