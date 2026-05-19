# Backend Simplification Rollout

**Source**: `docs/analysis/simplification_audit.md` (May 2026, pressure-tested 2026-05-19).
**Status**: design — no stages shipped.
**Scope**: `backend/` simplification + targeted cross-cutting cleanups surfaced by the pressure-test pass (Priority 6 in the audit).

## Goal

Reduce `backend/` by ~1200 LOC and drop two of the top-3 largest files below the 800-line threshold, while preserving the public import surface that `scripts/`, `tests/`, and the integration test suite depend on. Latent-bug fixes (CLI logging) ship alongside.

## Approach

Six sequenced stages, **one PR per stage**. Stages are gated on `make pyright && make unit` plus stage-specific integration checks. No stage assumes work from a later stage; later stages may depend on guardrails added in earlier stages (smoke test in C, migration round-trip in D).

Why six and not one big PR: the audit's two highest-risk items (P1.3 step/executor flatten, P2.1 database split) each need a guardrail that doesn't yet exist (every-variant smoke test; migration round-trip). Folding them into one PR would either delay the low-risk wins or force the guardrails to ship without a clean review boundary. Six PRs also let A and B land within a session each, giving early signal that the audit's import-surface assumptions hold under CI.

Why not multi-file plan: per project plan-hygiene, multi-file split only after ≥2 stages ship and a handoff is actually needed.

---

## Stage A — Pure deletes & lint-level cleanups (lowest risk)

**Risk**: low. No callers, no semantic change.
**Estimated delta**: ~−180 LOC.
**Audit items**: 1.1, 6.1, 1.4 (`backend/__init__.py` half), 1.5, unused-imports half of 1.4.

### Edits
- `git rm backend/transcribe.py.backup` (audit §1.1).
- `git rm backend/test_cuDNN.py` OR `git mv backend/test_cuDNN.py scripts/diagnose_cudnn.py` (audit §6.1) — decide based on whether the team still runs it. Default: `git rm`; the diagnostic script approach should be a separate decision.
- Reduce `backend/__init__.py` to a one-line docstring; drop the lazy `__getattr__` for `transcribe`, `transcribe_to_json`, `pick_model`. Pressure-test confirmed no callers via the package alias (audit §1.4).
- Replace the 22 `except Exception: pass` blocks in `backend/database.py` (L376, 430, 480, 548, 566, 571, 643, 647, 652, 802, 806, 810, 834, 842, 898, 907, 917, 1041, …) with `logger.debug("…: %s", exc)` (audit §1.5). Use the existing module logger; do not introduce new ones.
- Remove the specific unused imports listed in audit §1.4: `Segment`, `TranscriptionInfo` in `transcribe.py:107`; `Variant` in `services/factory.py:22`; `Variant` in `processor.py:20`; `TranscriptionProcessor` in `components.py:21`.

### Gate
- `make pyright && make unit && make integration-local` green.
- `grep -rn "from backend import" scripts/ tests/ frontend/` returns no hits for `transcribe`, `transcribe_to_json`, or `pick_model` (re-verify pressure-test finding).
- `grep -rn "transcribe.py.backup\|test_cuDNN" .` clean.

---

## Stage B — Script coordination + targeted re-export prune + CLI logging fix

**Risk**: low–medium. One script update + one re-export prune + one correctness fix.
**Estimated delta**: ~−30 LOC + bug fix.
**Audit items**: 1.4 (`backend/variants/__init__.py` rest), 6.2.

### Edits
- Update `scripts/compare_transcription_variants.py:18` from `from backend.variants import execute_variant, get_builtin_variants` to `from backend.variants.executor import execute_variant` + `from backend.variants.registry import get_builtin_variants`.
- Prune `backend/variants/__init__.py`: either reduce `__all__` to `["execute_variant", "get_builtin_variants"]` (the two names with confirmed external usage), or delete the re-export wall entirely. Recommend reducing `__all__` rather than deletion — keeps `Variant` and `PreprocessStep` re-exports for `tests/unit/test_variants.py` `isinstance` checks.
- Replace `cli/db.py:138–143` and `:197–202` traceback-by-argv pattern with `logger.exception("…", exc_info=exc)`. Drop the `import traceback` and `if "--verbose" in sys.argv` branches; the rich-tracebacks logger configured in `backend/config.py:39` renders the traceback when log level allows it (audit §6.2). **Caveat (surfaced post-Stage-B):** `setup_logging()` is not actually called from the `db` command entry path — only from `transcription_commands.py` and `scripts/transcribe_manager.py` — so `db show` / `db recent` route their `LOGGER.exception` calls through Python's `lastResort` stderr handler (plain text, no Rich formatting). Functionally equivalent (traceback prints), aesthetically not. See Stage F follow-up below.

### Gate
- Run `scripts/compare_transcription_variants.py --help` (or its dry-run mode if it has one) — exit 0, no `ImportError`.
- `make pyright && make unit` green.
- Manually trigger one CLI db command with a deliberately bad arg to confirm the new exception path renders correctly.

---

## Stage C — Step/executor flatten (highest LOC delta, behaviour-preserving)

**Risk**: HIGH. ~−700 LOC of structural change to `backend/variants/steps.py`. Behaviour parity must be proven, not assumed.
**Estimated delta**: ~−700 LOC.
**Audit items**: 1.2, 1.3, with the registry-coordination caveat from the pressure-test pass.

### Prerequisite (lands in the same PR, first commit)
- Add `tests/integration/test_every_step_smoke.py` (or extend `tests/integration/test_variant_systems_comparison.py`) that runs each step type at least once against a small fixture audio file and snapshots an output property (output sample count, RMS bucket, or first/last 1 KB MD5 — pick whichever is cheap and stable). This is the audit's named guardrail.
- The smoke test must pass with the current implementation BEFORE any flatten edit lands.

### Edits
1. Update `backend/variants/registry.py` (and any variant definition fixtures) so that variants currently referencing the 9 empty `*StepConfig` dataclasses by class reference a sentinel/None instead. Pressure-test established this is the load-bearing path — collapsing the dataclasses is not safe without this step.
2. Delete `class StepConfig(ABC)`, `class BaseStepConfig`, and the 9 empty configs: `ResampleStepConfig`, `FFmpegStepConfig`, `DenoiseStepConfig`, `DynaudnormStepConfig`, `HighlowAformLoudnormStepConfig`, `HighlowNosamplLoudnormStepConfig`, `Loudnorm2passLinearStepConfig`, `LimiterOnlyStepConfig`, `CompressorLimiterStepConfig`, `DynaudnormConservativeStepConfig`. Keep the 8 typed configs that carry real fields.
3. Replace `FFmpegExecutor`, `SoxExecutor`, `PreprocessStepsExecutor`, `PythonExecutor`, and the `StepExecutor` Protocol with module-level functions (`run_resample`, `run_loudnorm_only`, `run_denoise`, `run_ffmpeg_pipeline`, …) and a single `STEP_HANDLERS: dict[str, Callable]` dispatch table. Move `Step.execute` to a thin dispatch site.
4. Delete `class Step(ABC)` and the per-step glue subclasses if they become no-ops after step 3.
5. Convert the three lazy imports in `backend/variants/steps.py:491,511,532` (`from backend.variants import preprocess_steps`) to direct submodule imports (e.g. `from backend.variants import preprocess_steps as _preprocess_steps` or `import backend.variants.preprocess_steps as _preprocess_steps`). Stage B's `__init__.py` prune left these working only via Python's submodule fallback; under a major restructure of `steps.py` the implicit fallback is fragile. Surfaced by Stage B code review (commit `a79522c`).

### Gate
- Smoke test added in the prerequisite passes against the refactored code with **byte-identical or property-identical** output per the snapshot strategy chosen.
- `make pyright && make unit && make integration` green.
- `python scripts/variant_checks/verify_all_variants.py` runs to completion (audit's named seam; pressure-test confirmed it imports from `backend.variants.executor` + `.registry`, so the refactor must preserve those submodules' public names).

---

## Stage D — Database split + transactional dedup (highest structural complexity)

**Risk**: HIGH for the migration body; MED for the split itself.
**Estimated delta**: net 0 (moves); `database.py` drops from 1947 to ~300 LOC. ~−90 LOC from dedup.
**Audit items**: 2.1, 2.2, 6.3.

### Prerequisite (lands in the same PR, first commit)
- Add `tests/integration/test_migration_006_roundtrip.py`: snapshot a copy of `transcribe_state.db` (or a synthetic old-shape fixture), run `_init_db` against it, assert the post-migration schema and a row of data round-trip correctly. Audit's existing P2.1 guardrail.
- The round-trip test must pass against the current code BEFORE any split edit lands.

### Edits
1. Create `backend/database/` directory. Move `_migration_006_normalize_runs_schema` **untouched** into `backend/database/migrations/migration_006.py`. Body-level refactor is **out of scope** for this PR per the existing audit guardrail.
2. Move other migrations (`_migration_003_*`, `_migration_004_*`, `_migration_005_*`, etc.) to `backend/database/migrations/migration_NNN.py`. Extract the column-loop ladder into `_add_columns_if_missing(conn, table, columns)` in `backend/database/_columns.py` (audit §2.2).
3. Move `_get_columns`, `_table_exists`, DDL strings into `backend/database/schema.py`.
4. Move `record_run`, `get_run_by_id`, `get_run_history` into `backend/database/runs.py`.
5. Move `record_file_metric` and adjacent methods into `backend/database/file_metrics.py`.
6. Move file-tracking (queued/in-progress/done) methods into `backend/database/state.py`.
7. Extract `_begin_transaction_or_continue(conn)` and `_commit_or_rollback(conn, started: bool)` into `backend/database/_txn.py`. Replace call sites at `database.py:546–550`, `:1543–1547`, and the four COMMIT/ROLLBACK duplicates at L568, 649, 904, 914 (audit §6.3 + §2.2).
8. Reshape `TranscriptionDatabase` into a thin coordinator in `backend/database/__init__.py` (or `backend/database/coordinator.py` re-exported via `__init__.py`). **`from backend.database import TranscriptionDatabase` MUST keep working** — verified post-split via the 5 script callers in audit §2.1.

### Gate
- Migration round-trip test (prerequisite) still passes against the split code.
- `python -c "from backend.database import TranscriptionDatabase; TranscriptionDatabase()"` exits 0.
- All 5 scripts referenced in audit §2.1 importable: `python -c "import scripts.transcribe_manager; import scripts.db.show_run; import scripts.db.show_recent_runs; import scripts.db.check_orphaned_metrics; import scripts.db.check_parameter_completeness"`.
- `make pyright && make unit && make integration` green.

---

## Stage E — Function-level extractions (parallel-safe, mechanical)

**Risk**: MED. Mechanical refactors with named extraction targets, but several touch 8–13-param functions (audit §3).
**Estimated delta**: ~−230 LOC.
**Audit items**: 2.3, 2.4, 2.5, 4, and §3 (`VariantRunContext`) ride-along.

### Edits
- **2.3**: Break `transcribe.transcribe()` (231 LOC) into `_parse_suppress_tokens`, `_apply_language_default`, `_collect_segments(segments, threshold)`, `_build_metrics_payload(…)`. Reduces the body to ~70 LOC.
- **2.4**: Extract `_run_transcription(model_kwargs: dict, …)` helper from `transcribe_with_baseline_params`, `transcribe_with_minimal_params`, `execute_variant` in `backend/variants/executor.py`. The baseline/minimal distinction collapses to "which kwargs we pass". Pressure-test note: `verify_all_variants.py` imports `transcribe_with_minimal_params` and `create_variant_transcribe_config` from `backend.variants.executor` — those names must survive.
- **2.5**: Factor `_run_single_filter(input_path, output_path, filter_chain, step_name) -> StepMetrics` from `dynaudnorm_only`, `limiter_only`, and the fixed-preset loudnorm wrappers in `backend/variants/preprocess_steps.py`. **Scope limit (audit §6.4 informational)**: stay inside `variants/preprocess_steps.py`. Do not absorb `backend/preprocess/orchestrator.py` calls — those are intentional coexistence.
- **4**: Rename `RunSummarizer` (in `backend/components.py:261`) to a module of free functions; the four feature-envy methods don't share state. Also extract the ~30-line metrics-payload block out of `process_file` (`components.py:141`).
- **§3 ride-along**: introduce `VariantRunContext` dataclass (or extend the existing `EffectiveRunConfig` in `backend/run_config.py:17`) to absorb the `(variant_number, variant_description, base_name, datetime_suffix, output_dir, copy_intermediate, original_filename)` tuple threaded through `_create_run_record`, `execute_variant`, `transcribe`, `transcribe_and_save`, and `process` (CLI). Per audit §3, this rides on the first extraction that introduces a multi-param signature — it should NOT be its own PR.

### Gate
- `make pyright && make unit && make integration` green.
- `python scripts/variant_checks/verify_all_variants.py` succeeds (still gating; same submodule names as Stage C).
- `grep -rn "transcribe_with_minimal_params\|create_variant_transcribe_config\|PreprocessResult\|PreprocessMetrics" scripts/` shows the audit-named smoke seam imports unchanged.

---

## Stage F — Lint-level cleanups (deferrable)

**Risk**: low. Mechanical, no semantic change.
**Estimated delta**: small.
**Audit items**: §5.

### Edits
- Split `cli/db.py:show` (L59, cognitive=39) `--filter` branches into helpers.
- Split `cli/db.py:recent` (L148, cognitive=33).
- Narrow `cli/db.py:show` and `:recent` exception handlers: `except Exception as e:` currently swallows `typer.Exit` (subclass of `RuntimeError` via `click.exceptions.Exit`), so legitimate "not found" / "no runs" exits produce a spurious `Error: 1` line plus a traceback. Either re-raise `typer.Exit` before falling into the `LOGGER.exception` branch, or restrict the except to the specific DB error types. Pre-existing bug; Stage B made it slightly louder by emitting the traceback unconditionally. Surfaced by Stage B code review (commit `a79522c`).
- Call `setup_logging()` from `backend/cli/main.py` (once, before subcommand dispatch) so `db show` / `db recent` `LOGGER.exception` tracebacks render through the RichHandler in `backend/config.py:39` rather than Python's plain `lastResort` stderr handler. One-line fix; has visibility implications because pure `db`-command runs currently emit zero pre-exception logging. Surfaced by Stage B code review.
- Normalise `LOGGER.exception("…", exc_info=e)` in `cli/db.py` (Stage B added these) to the idiomatic `LOGGER.exception("…")` — inside an active `except` block, `exc_info=e` is redundant with `Logger.exception`'s built-in `exc_info=True`. Touch alongside the `show`/`recent` split since those handlers move anyway.
- Rewrite `cli/transcription_commands.py:_is_test_run` (L40) as `any(s in path for s in SENTINELS)`.
- Extract per-variant inner loop from `cli/transcription_commands.py:_process_multi_variant` (L355, 115 LOC).
- Replace `_NoOpTqdm` in `transcribe.py:43` with `huggingface_hub.utils.disable_progress_bars()` if available, else move to `backend/utils/quiet_tqdm.py`.
- `services/duckdb_state_store.py:DuckDBStateStore`: keep the layer (Protocol seam); add a comment documenting why it exists, then drop the thin-wrapper warning from the audit.

### Gate
- `make pyright && make unit && make integration` green.
- Stage B's `logger.exception` shape in `cli/db.py` is preserved (do NOT re-introduce traceback-by-argv when splitting `show` / `recent`).
- `stt-faster db show 99999999` (or any not-found ID) exits with the typer-expected status WITHOUT printing `Error: 1` + traceback (confirms the `typer.Exit` swallow is fixed).

---

## Decisions log (re-work audit per plan-hygiene)

This plan lands a long ordered step list. Per `~/.claude/skills/mybrain/SKILL.md` step 7.5 and project plan-hygiene "≥10 steps even single-PR rollout" trigger, I walked each stage's edits and tagged the files touched:

| Stage | Files touched |
|---|---|
| A | `backend/transcribe.py.backup` (del), `backend/test_cuDNN.py` (del), `backend/__init__.py` (gut), `backend/database.py` (22 lines), `backend/transcribe.py`, `backend/services/factory.py`, `backend/processor.py`, `backend/components.py` (imports) |
| B | `backend/variants/__init__.py`, `scripts/compare_transcription_variants.py`, `backend/cli/db.py` (2 sites) |
| C | `backend/variants/steps.py` (massive), `backend/variants/registry.py`, new test under `tests/integration/` |
| D | `backend/database.py` (split into ~7 files under `backend/database/`), new test under `tests/integration/` |
| E | `backend/transcribe.py`, `backend/variants/executor.py`, `backend/variants/preprocess_steps.py`, `backend/components.py`, `backend/run_config.py` (or new `VariantRunContext` location) |
| F | `backend/cli/db.py`, `backend/cli/transcription_commands.py`, `backend/transcribe.py`, possibly new `backend/utils/quiet_tqdm.py` |

**Cross-stage re-write risks identified and resolved:**

1. **Stage A modifies `backend/database.py` (22 logging lines); Stage D splits the file.** Acceptable: the logging changes move with their content under `git mv`-then-edit. Not a rewrite.
2. **Stage B edits `backend/cli/db.py` (logging fix); Stage F splits `show` / `recent` in the same file.** Risk: F's extraction could re-introduce the traceback-by-argv shape. Mitigated by Stage F's gate: "B's `logger.exception` shape preserved." Decision: this is a gate, not a re-merge.
3. **Stage A edits `backend/transcribe.py` imports; Stage E edits its body.** Different lines, no rewrite.
4. **Stage A edits `backend/components.py` imports; Stage E edits `RunSummarizer` + `process_file` in the same file.** Different sections, no rewrite.
5. **Stage C touches `backend/variants/steps.py`; Stage E touches `backend/variants/executor.py` and `backend/variants/preprocess_steps.py`.** Different files in the same package. `executor.py` calls into `steps.py`, so C must precede E so E's `_run_transcription` extraction doesn't rebuild around classes that C deletes. **Sequencing decision: C → E (mandatory order).** D is parallel-safe with C/E (different subsystem).
6. **`_migration_006_normalize_runs_schema` body**: explicitly moved untouched in D. Body refactor is out of scope per audit guardrail.

**Outcome**: no targets are written-then-rewritten across stages. Sequencing constraints: A and B can ship in either order or in parallel; C must precede E; D is parallel-safe with everything; F must come after B.

---

## Out of scope (do not absorb into this rollout)

- **FFmpeg cross-pipeline unification** between `backend/preprocess/orchestrator.py` and `backend/variants/preprocess_steps.py`. Pressure-test established these are intentional coexistence (audit §6.4); P2.5 stays inside the variants system only.
- **Config schema consolidation** across `backend/model_config.py`, `backend/preprocess/config.py`, `backend/run_config.py`. Layered, not duplicated; pressure-test for boundary reasons before any consolidation (audit §6.5).
- **`_migration_006_normalize_runs_schema` body refactor**. Move untouched in Stage D; refactor only after an integration test round-trips an old-shape DB.
- **frontend changes**. None of the proposed backend refactors change the API surface that the frontend consumes; the frontend is unaffected.
- **New features**. This is a debt-reduction rollout.
- **`scripts/db/check_parameter_completeness.py:26` private `_get_all_variants` import.** Flagged in audit §1.4 as a fragility, but no rename is proposed in this rollout. Leave the import as-is; revisit if a future refactor renames registry internals.

---

## Acceptance criteria (whole rollout)

- `backend/database.py` LOC < 800 (drops from 1947 to ~300 under Stage D).
- `backend/variants/steps.py` LOC < 800 (drops from 1531 under Stage C).
- `make pyright && make unit && make integration` green at every stage boundary.
- All 5 `TranscriptionDatabase` script callers (audit §2.1) work without modification at the end of D.
- `scripts/compare_transcription_variants.py` works after B's update.
- `scripts/variant_checks/verify_all_variants.py` works at end of C, E, and F.
- New integration tests added in C (every-step smoke) and D (migration round-trip) remain green in subsequent stages.

---

## Status tracking

Mark stages here as they ship. Do NOT pre-mark; do NOT update from a planning session. Only update when a PR merges.

- [ ] Stage A — pure deletes & lint-level cleanups
- [ ] Stage B — script coordination + re-export prune + CLI logging fix
- [ ] Stage C — Step/executor flatten
- [ ] Stage D — database split + transactional dedup
- [ ] Stage E — function-level extractions + `VariantRunContext`
- [ ] Stage F — lint-level cleanups
