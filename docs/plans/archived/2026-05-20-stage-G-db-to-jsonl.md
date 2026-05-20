# Stage G — DuckDB removal + JSONL replacement

**Branch / PR**: TBD (single PR, mirrors Stage D.2 atomic shape)
**Status**: Design approved 2026-05-20. Implementation pending.
**Prereq**: None — independent of Stage E.2.

## 0. Context

The transcription pipeline has shipped Stages A → F.2 of `docs/plans/backend_simplification_rollout.md`. Stage E.2 (file-isolated `_run_single_filter` + `RunSummarizer` refactor in `components.py`) remains. Stage G is a new top-level rip-out independent of E.2.

Empirical trigger: in the 2026-05-20 design session the production DuckDB at `~/.local/share/stt-faster/transcribe_state.duckdb` was found unreadable (WAL replay failure on every connect, traced to a CREATE/ALTER with a function-default expression that DuckDB 1.4.2 can't bind during replay). The user had not noticed the breakage despite the DB having been broken for ≥12h. This was treated as the strongest possible empirical signal that no read path is load-bearing in the actual workflow.

The WAL was moved aside (`transcribe_state.duckdb.wal.bak`) for one-time inspection of the 465 historical runs. All 6 readers (`stt-faster db recent`, `db show`, `transcribe_manager status`, `scripts/db/show_recent_runs.py`, `show_run.py`, `check_orphaned_metrics.py`, `check_parameter_completeness.py`) work after the WAL move. The data is preserved for export.

## 1. Goal

Replace `TranscriptionDatabase` + DuckDB + migrations + state-store Protocol seam with a single append-only `runs.jsonl` log. Preserve the 465 historical runs by one-shot export. Delete ~3,750 LOC of production code + ~500 LOC of tests + the `duckdb` dependency.

## 2. Pre-state inventory

### Tables in production DB (post-WAL-recovery)

| Table | Rows | Notes |
|-------|-----:|-------|
| `runs` | 465 | Wide row, 55 columns. Authoritative for run-level data. |
| `file_metrics` | 471 | 1 per run (single-file batches dominate). |
| `run_parameters` | 5,429 | Key/value form of same params (cols: `name`, `value_type`, `value`, `category`). Denormalized duplicate of `runs`. |
| `run_configs` | 338 | Narrower per-run config added in migration 005+. 127 older runs missing this row. |
| `run_metrics` | 338 | Narrower per-run metrics added in migration 005+. Same gap as `run_configs`. |
| `transcriptions` | **0** | Per-file status table — `add_file` is never called in prod, `update_status` UPDATEs empty table. Dead since project start. |
| `schema_version` | 6 | Migration history 001–006. |

### History coverage

- **Earliest run**: 2025-12-08
- **Latest run**: 2026-05-18
- **Presets**: et-large 443 (95%), et-subs 15, turbo 6, et-32 1
- **DB size**: 143 MB on disk (~2 MB of actual data, ~141 MB of UPDATE-induced block bloat — DuckDB doesn't reclaim).

### Schema drift / dead-code findings

- **4 overlapping tables for same per-run data** (`runs` wide + `run_parameters` k/v + `run_configs` narrow + `run_metrics` narrow). Migration 006 was the consolidation attempt; per `backend_simplification_rollout.md` Stage D.1 it has a latent bug (the wide-row data-migration loop is dead code that only doesn't fire because of an L477 short-circuit guard).
- **Empty `transcriptions` table**: per-file status tracking has never worked. `transcribe_manager status` reports "No files tracked in database" against a 465-run DB.
- **Init-error swallow**: `backend/database/__init__.py:108-110` catches all migration exceptions as `LOGGER.warning(...)`. This is why the WAL break went unnoticed.
- **DuckDB-as-OLTP mismatch**: the workload (write-after-write per CLI, ~500 rows total, reads = `SELECT * ORDER BY recorded_at DESC LIMIT N`) does not match DuckDB's analytical-scan strengths. Drives the 70× block bloat.

## 3. Target state

### Data files

```
~/.local/share/stt-faster/
├── runs.jsonl                     # NEW: one JSON line per run
└── (transcribe_state.duckdb*       # DELETED after migration script runs)
```

### Code surface

- `backend/database/` → **deleted** (entire package, ~1,400 LOC + 640 LOC migrations)
- `backend/services/duckdb_state_store.py` → **deleted**
- `backend/services/interfaces.py` → `StateStore` Protocol either deleted OR re-pointed at a tiny `JsonlRunLog` class (decided below)
- `backend/cli/db.py` → shrunk to ~80 LOC reader (tail + grep over JSONL)
- `scripts/db/check_orphaned_metrics.py` → deleted (no orphans possible in append-only JSONL)
- `scripts/db/check_parameter_completeness.py` → deleted (every record contains all fields by construction)
- `scripts/db/show_run.py`, `show_recent_runs.py` → deleted (CLI `db show`/`db recent` covers it)
- `scripts/transcribe_manager.py status` subcommand → deleted (was always broken)
- `backend/components.py` → write path changed from 3 calls (`record_run` / `record_file_metric` / `update_status`) to 1 call (`run_log.append(run_record_dict)`) at end of run
- `pyproject.toml` → remove `duckdb` dep

## 4. Design

### 4.1 JSONL record shape

One JSON object per line in `runs.jsonl`. Nested groups by concern; `files: [...]` array embeds per-file metrics. Approved shape:

```json
{
  "id": 465,
  "recorded_at": "2026-05-18T15:06:38.247419Z",
  "input_folder": "/mnt/c/Users/PC/Downloads/transcribe",
  "preset": "turbo",
  "language": "en",
  "preprocess": {
    "enabled": true,
    "profile": "cpu",
    "target_sample_rate": 16000,
    "target_channels": 1,
    "loudnorm": {"preset": "...", "target_i": -16, "target_tp": -1.5, "target_lra": 11, "backend": "..."},
    "denoise": {"method": null, "library": null, "rnnoise_model": null, "rnnoise_mix": null},
    "snr_estimation_method": null,
    "volume_adjustment_db": null,
    "resampler": null,
    "sample_format": null
  },
  "model": {
    "id": "Systran/faster-distil-whisper-large-v3",
    "device": "cuda",
    "compute_type": "float16"
  },
  "params": {
    "beam_size": 5,
    "best_of": null,
    "patience": null,
    "task": "transcribe",
    "chunk_length": 30,
    "word_timestamps": true,
    "vad": {"filter": true, "threshold": 0.5, "min_speech_ms": 250, "max_speech_s": 30, "min_silence_ms": 2000, "speech_pad_ms": 400},
    "temperature": "0.0,0.2,0.4,0.6,0.8,1.0",
    "temperature_increment_on_fallback": 0.2,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "suppress_tokens": "-1",
    "condition_on_previous_text": false,
    "initial_prompt": null
  },
  "totals": {
    "files_found": 1,
    "succeeded": 1,
    "failed": 0,
    "processing_time_s": 301.80,
    "preprocess_time_s": 1.42,
    "transcribe_time_s": 297.28,
    "audio_duration_s": 6930.77,
    "speed_ratio": 23.31
  },
  "files": [
    {
      "path": "x.mp3",
      "status": "completed",
      "audio_duration_s": 6930.77,
      "preprocess_time_s": 1.42,
      "transcribe_time_s": 297.28,
      "segment_count": 142,
      "detected_language": "en",
      "error_message": null
    }
  ]
}
```

**Nesting rationale**: top-level scalars stay flat (`id`, `recorded_at`, `preset`) for trivial grep; logical groups (`preprocess`, `model`, `params.vad`, `totals`, `files[]`) reduce flat-record key sprawl from 55 to ~8 top-level keys.

**Atomicity**: append with `open(..., 'a', encoding='utf-8')` + single `f.write(json.dumps(record, separators=(',', ':')) + '\n')` + `f.flush()`. Single short writes are atomic on POSIX (POSIX guarantees no interleaving for writes ≤ `PIPE_BUF` to a regular file with `O_APPEND`; our records are <10 KB worst case, well under typical 4 KB block but writes to regular files don't actually have a strict atomicity guarantee — see Risk R-3).

### 4.2 Migration script

**Path**: `scripts/migrate_db_to_jsonl.py` (new, deleted in same PR after running once).

**Idempotency contract**: re-running against the same DB + output path emits identical JSONL (deterministic ordering by `runs.id ASC`).

**Algorithm**:
1. Open DuckDB read-only at `~/.local/share/stt-faster/transcribe_state.duckdb`.
2. For each row in `runs ORDER BY id ASC`:
   - Start from the wide `runs` row.
   - LEFT JOIN `run_configs` (newer narrow data) — when columns disagree, **`run_configs` wins** (it's the more recent source, intended successor).
   - LEFT JOIN `run_metrics` — same precedence rule.
   - `run_parameters` is treated as **dead data**, not exported. (5,429 rows are k/v duplicates of `runs` wide columns; no information lost.)
   - SELECT children from `file_metrics WHERE run_id = ?` → JSON-encode into `files[]`.
   - Map flat columns into the nested shape in §4.1.
   - Format `recorded_at` as ISO-8601 UTC with microseconds + `Z` suffix.
3. Write to `~/.local/share/stt-faster/runs.jsonl` (overwrite if exists).
4. Print summary: N runs exported, byte count, earliest/latest timestamp.

**Source-of-truth precedence when tables disagree**: `run_configs`/`run_metrics` (newer) > `runs` (older wide row).

**Skip empty `transcriptions` table** entirely — confirmed 0 rows.

**Run-ID continuity**: preserve historical IDs (1–465). New runs start at 466.

**Post-migration cleanup**: script does NOT auto-delete the DB. User runs `rm ~/.local/share/stt-faster/transcribe_state.duckdb*` manually after verifying the JSONL.

### 4.3 New write path in `backend/components.py`

Replace 3 state-store calls with 1 JSONL append. Sketch:

```python
# backend/run_log.py (NEW, ~60 LOC)
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from backend.config import get_default_run_log_path  # moved from database.schema.get_default_db_path

class JsonlRunLog:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else get_default_run_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> int:
        run_id = self._next_id()
        record_with_id = {"id": run_id, **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record_with_id, separators=(",", ":"), default=str) + "\n")
            f.flush()
        return run_id

    def _next_id(self) -> int:
        if not self.path.exists():
            return 1
        max_id = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec_id = json.loads(line).get("id", 0)
                    if isinstance(rec_id, int) and rec_id > max_id:
                        max_id = rec_id
                except json.JSONDecodeError:
                    continue
        return max_id + 1

    def tail(self, limit: int) -> list[dict[str, Any]]:
        ...  # read all, return last `limit` parsed records

    def get(self, run_id: int) -> dict[str, Any] | None:
        ...  # linear scan, return first matching record (2 MB file → ~1 ms)
```

`components.py` `_create_run_record` builds the nested dict and calls `run_log.append(record)`. The 3 prior write sites (`update_status` on per-file completion/failure, `record_run` at end, `record_file_metric` per file) collapse into one call after all files finish, building the `files[]` array in-memory.

### 4.4 New CLI readers

`backend/cli/db.py` shrinks from ~250 LOC to ~80 LOC:

- `stt-faster db recent [--limit N]` — `run_log.tail(N)`, pretty-print with same Rich formatting as today's `_print_runs_full`.
- `stt-faster db show <id>` — `run_log.get(id)`, pretty-print with same Rich formatting as today's `_print_run_details`.
- Drop `_resolve_run_timestamps`, `_print_timing_block` helpers — flatten the formatting now that there's no SQL row-vs-dict ambiguity.

Rich-handler setup (Stage F.1.b's `setup_logging()` call in `cli/main.py`) stays.

### 4.5 Delete list (exact files/lines)

**Whole files deleted:**
- `backend/database/__init__.py` + everything under `backend/database/` (1,439 LOC)
- `backend/database/migrations/` (640 LOC, 6 migrations + `__init__`)
- `backend/services/duckdb_state_store.py` (28 LOC)
- `scripts/db/check_orphaned_metrics.py` (93 LOC)
- `scripts/db/check_parameter_completeness.py` (594 LOC)
- `scripts/db/show_run.py` (182 LOC)
- `scripts/db/show_recent_runs.py` (195 LOC)
- `scripts/db/__init__.py` (1 LOC) — directory removed entirely
- `tests/integration/test_migration_006_roundtrip.py`
- `tests/integration/test_production_db_migration.py`
- `tests/unit/test_transcribe_database.py`

**Files touched:**
- `backend/components.py` — replace 3 state-store call sites (lines 155, 245, 253 update_status; 516 record_run; 606 record_file_metric) with 1 `run_log.append(...)` at end of `process_files`
- `backend/services/interfaces.py` — delete `StateStore` Protocol (lines 49-63)
- `backend/services/factory.py` — remove `TranscriptionDatabase` import + construction; inject `JsonlRunLog` instead
- `backend/cli/db.py` — rewrite as JSONL reader (~80 LOC)
- `backend/cli/main.py` — no change (the `db` typer app still registers)
- `scripts/transcribe_manager.py` — delete `status` subcommand + the `from backend.database import TranscriptionDatabase` import
- `pyproject.toml` — remove `duckdb` from `dependencies`
- `requirements*.lock` / `uv.lock` — regenerate via `make export-reqs`

**Tests touched (not deleted):**
- `tests/unit/test_transcribe_processor.py` — 4 `get_status(...)` assertions at lines 254, 307, 346, 549 → replace with assertions on the JSONL record (`run_log.tail(1)[0]`)
- `tests/e2e/test_real_transcription.py:86` — same swap
- `tests/integration/test_cli_commands.py` — adapt `db show`/`db recent` smoke tests to JSONL path
- `tests/unit/conftest.py`, `tests/integration/conftest.py`, `tests/e2e/conftest.py` — replace `temp_db` fixture with `temp_run_log` fixture (tmp_path / runs.jsonl)
- `tests/unit/test_architecture.py` — drop any rules that pin DB imports

**Docs touched:**
- `docs/Transcription_solution.md` — "Database tracking (SQLite-based history and metrics)" line → "Run log (JSONL append-only)"
- `CLAUDE.md` — no direct mention of `backend/database` in CRITICAL RULES, but verify Tooling Quickref + grep for any leftover refs
- `docs/AI_instructions.md` — grep for migration refs
- `docs/testing_approach.md` — drop migration-roundtrip test reference if present
- `AGENTS.md`, `CODEX_RULES.md`, `.cursor/rules/*.mdc` — grep

## 5. Risk register / pre-mortem

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|:----------:|:------:|------------|
| R-1 | Migration script silently drops fields from a run whose `run_configs` row is NULL (older runs) | Med | High | Test: export all 465 runs, assert no field is `null` for any run that had a non-null value in the source `runs` table. Walk `runs` columns explicitly when `run_configs` is missing. |
| R-2 | Run-ID gap if a CLI invocation fails between `_next_id()` and append | Low | Low | IDs may have small gaps. Accept — IDs are display-only, no FK references survive. |
| R-3 | Concurrent CLI runs racing on append (`_next_id` reads stale max while another append is in-flight) | Very low (single-user dev workflow) | Med | Acceptable: user runs CLI one at a time. Document in `run_log.py` docstring. Real fix would be `flock` or SQLite — out of scope. |
| R-4 | `runs.jsonl` grows unbounded | Low (5 mo → 465 runs ≈ 100 runs/mo; 10 yr → ~12K runs ≈ <50 MB) | Low | No mitigation needed. Add a `db archive --before <date>` flag if it ever matters. |
| R-5 | A test in `test_transcribe_processor.py` depends on `update_status("failed", error_msg)` ordering | Low | Med | When swapping fixtures, walk each `get_status` assertion site and replace with the equivalent assertion on the JSONL record's `files[N].status` + `files[N].error_message`. |
| R-6 | Migration script can't open the production DB (WAL still broken) | Already happened — WAL was moved aside this session | Med | The WAL backup at `transcribe_state.duckdb.wal.bak` was preserved. If a fresh WAL has appeared by implementation time, re-apply the move. Document in script header. |
| R-7 | `pyproject.toml` removes `duckdb` but a deferred import remains somewhere | Low | High (CI breaks) | `grep -rn '\bduckdb\b' backend/ scripts/ tests/` must return 0 hits before the PR lands. Pyright + ruff F401 sweep. |
| R-8 | Plan-hygiene violation: this doc grows stale if Stage G is split across sessions | Med | Low | Update §0 status line and §6 commit list as commits land. Archive to `docs/plans/archived/` once the PR merges. |

## 6. Order of operations (commits within the PR)

1. **G.a — Export tool + JSONL writer module** (no behavior change yet)
   - Add `backend/run_log.py` with `JsonlRunLog`
   - Add `scripts/migrate_db_to_jsonl.py`
   - Run the script locally, verify `runs.jsonl` has 465 records, eyeball sample
   - Tests: unit tests for `JsonlRunLog.append` / `tail` / `get` + a golden-file test for the migration script (small synthetic DB → expected JSONL)

2. **G.b — Swap write path in `components.py`** (DB still wired for reads)
   - Replace 3 state-store calls with 1 `run_log.append(...)` build-and-append at end of `process_files`
   - Update `factory.py` to inject `JsonlRunLog` alongside `DuckDBStateStore` (temporary dual-write disabled — JSONL only)
   - Update affected unit tests (`test_transcribe_processor.py` 4 sites)
   - Gate: `make unit` + `make integration` + smoke-test a real CLI run end-to-end, confirm new line in `runs.jsonl`

3. **G.c — Rewrite CLI readers**
   - Rewrite `backend/cli/db.py` against `JsonlRunLog`
   - Drop `transcribe_manager.py status` subcommand
   - Update `tests/integration/test_cli_commands.py`
   - Gate: `stt-faster db recent --limit 5` shows the new run from G.b + the 10 most recent migrated runs

4. **G.d — Delete DB code**
   - Remove `backend/database/`, `backend/services/duckdb_state_store.py`, `scripts/db/`
   - Remove `StateStore` Protocol from `interfaces.py`
   - Remove `TranscriptionDatabase` import + construction from `factory.py`
   - Delete migration tests
   - Gate: `grep -rn 'TranscriptionDatabase\|backend\.database\|DuckDBStateStore' backend/ scripts/ tests/` returns 0 hits

5. **G.e — Drop `duckdb` dep + docs**
   - `pyproject.toml`: remove `duckdb`
   - `make export-reqs` to regenerate locks
   - Update `docs/Transcription_solution.md` + CLAUDE.md + AGENTS.md + .cursor/rules sweep
   - Gate: `make pyright` 0/0/0, `make unit`, `make integration`, full `make pre-commit`

6. **G.f — Delete production DB files** (manual, after PR merges)
   - User runs `rm ~/.local/share/stt-faster/transcribe_state.duckdb*` and `transcribe_state.db` (the December 8 SQLite leftover) once Stage G lands

## 7. Verification gates

- **Migration correctness**: hand-spot-check 5 random runs across the date range — earliest, latest, et-large, turbo, et-subs. Confirm fields round-trip.
- **`make pyright`**: 0 errors, 0 warnings, 0 info.
- **`make unit`**: all 127+ pass (some test-count change expected from deleting `test_transcribe_database.py`).
- **`make integration`**: all pass (-2 expected from deleting the two migration tests).
- **`scripts/variant_checks/verify_all_variants.py`**: all 4 active variants PASSED end-to-end.
- **Real CLI run** against a small audio file: confirm new line appended to `runs.jsonl`, `db recent --limit 1` shows it.
- **Grep guards**: `grep -rn '\bduckdb\b\|TranscriptionDatabase\|backend\.database\|DuckDBStateStore' backend/ scripts/ tests/` → 0 hits.

## 8. Open decisions deferred to implementation

- **D-1**: Keep `StateStore` Protocol pointing at `JsonlRunLog`, or delete the Protocol and have `components.py` depend on `JsonlRunLog` directly? Recommendation: **delete the Protocol** — single implementation, single consumer, the seam was over-engineered.
- **D-2**: Move `get_default_run_log_path()` to `backend/config.py` (alongside other path helpers) or keep it next to `JsonlRunLog`? Recommendation: **`backend/config.py`** — matches the existing pattern.
- **D-3**: Should the migration script live in `scripts/` permanently or be deleted after running? Recommendation: **delete in the same PR** after the user confirms `runs.jsonl` is correct. Keeps the repo clean. Script body lives in git history if ever needed.
- **D-4**: Strip `default=str` from the JSON encoder and explicitly convert datetimes / Paths at the call site? Recommendation: **explicit conversion** — `default=str` would silently stringify any unexpected type, hiding bugs.

## 9. Out of scope

- Schema evolution (no JSONL schema version field) — if/when the shape changes, append additive keys; old records remain readable.
- Concurrent-writer safety (single-user dev workflow).
- Run-log compaction / archival.
- Restoring the broken `transcriptions` per-file resume feature — confirmed unused, dropped intentionally.
- Stage E.2 (`_run_single_filter` + `RunSummarizer` refactor) — independent, will ship separately.
