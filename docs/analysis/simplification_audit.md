# Simplification Audit — Backend (May 2026)

**Source**: `python-simplifier` skill (analyze_complexity, find_code_smells, find_overengineering, find_dead_code, find_duplicates, find_coupling_issues, find_unpythonic) run against `backend/` on 2026-05-19.

**Scope**: `backend/` only. `tests/`, `scripts/`, `frontend/`, vendored `.cache/` excluded. False-positives for `from __future__ import annotations` filtered out (always reported as unused; mandatory for PEP 604 union syntax on 3.10).

**Pressure-test pass (2026-05-19)**: a follow-up scan extended the audit to `tests/`, `scripts/`, `frontend/`, and `docs/` and ran inversion against the two highest-LOC recommendations (FFmpeg pipeline unification and database split). Results inline as "Pressure-test caveat / found" callouts under §1.2, §1.4, §2.1, plus a new Priority 6 section for cross-cutting findings the per-file analyzer missed. Two original audit claims were amended: (a) `TranscriptionDatabase` cannot be retired (5 script callers); (b) the FFmpeg-pipeline duplication is intentional coexistence (do not unify).

**Top-of-file size table** (largest):

| LOC | File |
|---:|------|
| 1947 | `backend/database.py` |
| 1531 | `backend/variants/steps.py` |
| 1017 | `backend/variants/executor.py` |
| 847 | `backend/transcribe.py` |
| 842 | `backend/variants/preprocess_steps.py` |
| 656 | `backend/components.py` |
| 590 | `backend/cli/transcription_commands.py` |
| 418 | `backend/preprocess/config.py` |
| 375 | `backend/variants/registry.py` |

---

## Priority 1 — Quick wins (low-risk, high-signal)

### 1.1 Delete `backend/transcribe.py.backup` (131 lines, dead)
- Confirmed orphaned: no imports anywhere. Editor leftover from a stage cutover.
- **Action**: `git rm backend/transcribe.py.backup`. Verify `make pyright` + `make unit` green afterwards.

### 1.2 Remove confirmed-unused abstractions in `variants/steps.py`
Analyzer flags (HIGH/MED):
- `class Step(ABC)` at L845 — `unused_abstraction`: declared abstract but no concrete subclasses use the dispatcher meaningfully (each `Step` subclass is just `get_step_type` + `get_default_config` glue; the real polymorphism lives in the 100-line if/elif in `Step.execute`).
- `class StepConfig(ABC)` at L695 — single implementation (`BaseStepConfig`); merge.
- `BaseStepConfig` itself is empty (`pass`).
- 9 of the 15 `*StepConfig` dataclasses are empty (`pass`): `ResampleStepConfig`, `FFmpegStepConfig`, `DenoiseStepConfig`, `DynaudnormStepConfig`, `HighlowAformLoudnormStepConfig`, `HighlowNosamplLoudnormStepConfig`, `Loudnorm2passLinearStepConfig`, `LimiterOnlyStepConfig`, `CompressorLimiterStepConfig`, `DynaudnormConservativeStepConfig`.

**Action**: collapse empty configs into a sentinel; drop the `StepConfig`/`BaseStepConfig` layer; keep typed dataclasses only where they carry real fields (`DenoiseCustomStepConfig`, `LoudnormOnlyStepConfig`, `AresamplLoudnormFixedStepConfig`, `AresamplLoudnormFixed2StepConfig`, `VolumeLimiterStepConfig`, `PeakNormalize2passStepConfig`, `SoxPeakNormalizeStepConfig`, `LoudnormHighpassStepConfig`).

> **Pressure-test caveat (2026-05-19)**: the 9 empty `*StepConfig` dataclasses are not test-imported, but they are **runtime-constructed by `get_builtin_variants()` in `backend/variants/registry.py`** when variant definitions reference them by class. Collapsing to a sentinel requires coordinated registry updates (rewrite the variant definitions to use the sentinel) — not a pure deletion. Verify by grepping `registry.py` and any variant fixture files for each class name before removing.

### 1.3 Drop NotImplementedError "executor" classes
`variants/steps.py` defines `FFmpegExecutor`, `SoxExecutor`, `PreprocessStepsExecutor`, `PythonExecutor`, each declared against the `StepExecutor` Protocol. Most methods in each are stubs that raise `NotImplementedError("...should use OtherExecutor")` — analyzer reports LCOM=21 (very low cohesion) for all four.

The "executor" indirection is fake polymorphism: `Step.execute` (L882, cognitive_complexity=76, nesting_depth=9, function_length=103) hard-codes which executor each step type uses via if/elif, so the Protocol's Liskov-substitutable shape never gets exercised.

**Action**: replace 4 executor classes + 1 Protocol with a small set of module-level functions (`run_resample`, `run_loudnorm_only`, `run_denoise`, `run_ffmpeg_pipeline`, ...) and a single `STEP_HANDLERS: dict[str, Callable]` dispatch table. Removes ~700 lines from `steps.py`.

### 1.4 Prune dead re-exports / unused imports

**`backend/variants/__init__.py` re-exports** (`Variant`, `PreprocessStep`, `execute_variant`, `get_all_variants`, `get_builtin_variants`, `get_variant_by_name`, `get_variant_by_number`, `create_variant_preprocess_runner`, `create_variant_transcribe_config`):
- Pressure-test (2026-05-19) found **one external caller**: `scripts/compare_transcription_variants.py:18` uses `from backend.variants import execute_variant, get_builtin_variants`.
- The audit's named P1.3 smoke seam `scripts/variant_checks/verify_all_variants.py` imports from submodules (`backend.variants.registry`, `backend.variants.executor`) **not** through the re-export wall — pruning the wall does **not** break that script.
- **Action**: either update `scripts/compare_transcription_variants.py` to import from submodules first, then delete the re-exports; or keep `__all__` containing just `execute_variant` + `get_builtin_variants` and drop the rest. Do not delete the wall without coordinating the script update.

**`backend/__init__.py` lazy `__getattr__` re-exports** (`transcribe`, `transcribe_to_json`, `pick_model`):
- Pressure-test confirmed **zero external callers** via the package alias; all consumers use `from backend.transcribe import ...` directly.
- **Action**: safe to drop the `__getattr__` body and reduce `backend/__init__.py` to a docstring.

**Private-symbol fragility (separate finding)**: `scripts/db/check_parameter_completeness.py:26` imports `_get_all_variants` (leading underscore) directly from `backend.variants.registry`. Not in scope for P1.4 itself, but flag for any future rename of registry internals.

**Specific unused imports**: `Segment`, `TranscriptionInfo` in `transcribe.py:107`; `Variant` in `services/factory.py:22`; `Variant` in `processor.py:20`; `TranscriptionProcessor` in `components.py:21`.

### 1.5 Replace `try: ... except: pass` stubs in `database.py`
22 `except Exception: pass` blocks reported (lines 376, 430, 480, 548, 566, 571, 643, 647, 652, 802, 806, 810, 834, 842, 898, 907, 917, 1041, …). Most guard "migration probe" reads (e.g. checking for normalized-table existence). Add `logger.debug("...skipping: %s", exc)` per project Rule #3 (no `print`, structured logging only) so the migrations don't silently swallow real schema errors.

---

## Priority 2 — Structural splits (medium-risk; behaviour preservation matters)

### 2.1 Split `backend/database.py` (1947 lines, 972-line class)
`TranscriptionDatabase` is a god class. Severity-tagged hotspots:

| Symbol | Issue |
|--------|-------|
| `TranscriptionDatabase` (L975) | class_size=972 (threshold 300) |
| `_init_db` (L994) | function_length=165 |
| `record_file_metric` (L1401) | function_length=128 |
| `record_run` (L1532) | function_length=147, cognitive_complexity=37 |
| `get_run_by_id` (L1711) | cognitive_complexity=40, nesting_depth=7 |
| `get_run_history` (L1808) | cyclomatic=22, cognitive=72, nesting_depth=9 |
| `_migration_006_normalize_runs_schema` (L464) | function_length=375, cyclomatic=55, cognitive=166 |

**Suggested layout** (keep public class API as a façade — tests + `cli/db.py` consumers don't move):
- `database/schema.py` — `_get_columns`, `_table_exists`, DDL strings, all `_migration_00X` migrations (one per file under `database/migrations/`).
- `database/runs.py` — `record_run`, `get_run_by_id`, `get_run_history` (read/write of `runs` + normalized `run_configs`/`run_metrics`/`run_parameters`).
- `database/file_metrics.py` — `record_file_metric`, etc.
- `database/state.py` — file-tracking (queued/in-progress/done) used by `services/duckdb_state_store.py`.

**Façade is mandatory, not optional.** Pressure-test (2026-05-19) confirmed 5 script callers import `TranscriptionDatabase` directly: `scripts/transcribe_manager.py:26`, `scripts/db/show_run.py:20`, `scripts/db/show_recent_runs.py:20`, `scripts/db/check_orphaned_metrics.py:19`, `scripts/db/check_parameter_completeness.py:24`. The original wording ("or callers import them directly and the class is retired") would silently break all five. Keep `from backend.database import TranscriptionDatabase` working — either by promoting `TranscriptionDatabase` to a thin coordinator in `backend/database/__init__.py`, or by re-exporting it from a renamed implementation module. **Do not retire the class.**

### 2.2 Collapse migration 003/004/005 duplication
Migrations 003, 004, 005 each loop over a `{column_name: column_type}` dict and run the same `if column_type == "DOUBLE": ... elif "VARCHAR" ... elif "INTEGER" ... elif "BOOLEAN"` ladder (3 occurrences × ~12 lines each, flagged 3× by find_duplicates). The commit-on-COMMIT-or-ROLLBACK pattern in `database.py` lines 568/649/904/914 is also a 4-way duplicate.

**Action**: extract `_add_columns_if_missing(conn, table, columns)` and `_commit_or_rollback(conn, started: bool)` helpers. Saves ~80 lines.

### 2.3 Break up `transcribe.transcribe()` (L430, 231 lines)
Function length 231, cyclomatic 30, cognitive 66. Three identifiable phases that already exist informally as comment blocks:
1. **Setup**: language inference (L464–474), VAD param merge (L478–480), suppress_tokens parse (L482–497).
2. **Run**: `model.transcribe(...)` call + per-segment loop building `segment_payloads`, `audio_processed`, `no_speech_skip_windows` (L499–574).
3. **Aggregate**: timing + `TranscriptionMetrics` assembly (L576–end).

Extracting `_parse_suppress_tokens`, `_apply_language_default`, `_collect_segments(segments, threshold)`, and `_build_metrics_payload(...)` reduces the body to ~70 lines and lets the metrics builder be tested directly.

### 2.4 Break up `variants/executor.py` long functions
- `transcribe_with_baseline_params` (L318): 230 lines, cyclomatic 29.
- `transcribe_with_minimal_params` (L634): 284 lines, cyclomatic 41, cognitive 56.
- `execute_variant` (L149): 112 lines.

These three share ~60% of their bodies (model load → preprocess → transcribe → assemble metrics). Extract a `_run_transcription(model_kwargs: dict, ...)` helper; the baseline/minimal/full distinction becomes only "which kwargs we pass".

### 2.5 Deduplicate `variants/preprocess_steps.py` ffmpeg-only wrappers
`dynaudnorm_only` (L291) and `limiter_only` (L551) are 32-line near-duplicates with the same try/except + `ffmpeg.input(...).output(...)` shape. The same `ffmpeg.input/.output/.run` boilerplate at L301 and L561 is flagged as a 19-line block-level duplicate.

**Action**: factor an `_run_single_filter(input_path, output_path, filter_chain, step_name) -> StepMetrics` helper. Several "fixed-preset" loudnorm wrappers (`loudnorm_only`, `aresampl_loudnorm_fixed`, `aresampl_loudnorm_fixed2`) collapse into this too.

---

## Priority 3 — Parameter-list bloat (LOW-MED, do alongside 2.x)

49 functions exceed the 5-parameter threshold. The hot ones:

| Function | Params | File:L |
|----------|---:|-------|
| `_create_run_record` | 13 | `components.py:352` |
| `TranscriptionProcessor.__init__` | 12 | `processor.py:28` |
| `_execute_step_with_registry` | 12 | `variants/preprocess_steps.py:26` |
| `_run_pipeline` | 11 | `preprocess/orchestrator.py:154` |
| `preprocess_audio` | 9 | `preprocess/orchestrator.py:57` |
| `_copy_intermediate_file` | 9 | `variants/preprocess_steps.py:808` |
| `execute_variant` | 8 | `variants/executor.py:149` |
| `process` (CLI) | 8 | `cli/transcription_commands.py:563` |
| `transcribe` | 8 | `transcribe.py:430` |
| `transcribe_and_save` | 8 | `transcribe.py:767` |

Common theme: all these threads pass the same `(variant_number, variant_description, base_name, datetime_suffix, output_dir, copy_intermediate, original_filename)` tuple end-to-end. **Action**: introduce a `VariantRunContext` dataclass (or extend the existing `EffectiveRunConfig` in `backend/run_config.py:17`) and pass it once. Per `~/.claude/rules/python-refactors.md` "Tightening optional → required params", verify the orchestrator is the single fallback site before/while moving.

---

## Priority 4 — `components.py` cohesion fixes (MED)

`RunSummarizer` (L261) has LCOM=11 and four feature-envy methods:
- `_extract_run_config_from_metrics`: touches `sample_metric` 42× vs `self` 0×.
- `_persist_run_data`: touches `m` (file metrics) 59× vs `self` 2×.
- `summarize_run`: touches `metric` 6× vs `self` 3×.

Two options:
- **(a, minimal)**: rename `RunSummarizer` to a module of free functions (`summarize_run`, `_persist_run_data`, …) — they don't share state, so they shouldn't be methods.
- **(b, more invasive)**: move `_extract_run_config_from_metrics` and `_extract_run_config_from_config` onto `RunConfig` / `EffectiveRunConfig` in `run_config.py`.

Recommend **(a)** — pure dead-weight class with no instance state worth preserving.

Also: `process_file` (L141) is 80 lines, cyclomatic 16, cognitive 45 — extract the metrics-payload block (~30 lines) into a helper.

---

## Priority 5 — Small lint-level cleanups (deferrable)

- `cli/db.py:show` (L59) cognitive=39, cyclomatic=16 — split `--filter` branches into helpers.
- `cli/db.py:recent` (L148) cognitive=33.
- `cli/transcription_commands.py:_is_test_run` (L40) cognitive=29 — list-of-substring check; replace with `any(s in path for s in SENTINELS)`.
- `cli/transcription_commands.py:_process_multi_variant` (L355) — 115 lines, extract per-variant inner loop.
- `services/duckdb_state_store.py:DuckDBStateStore` (L6) — flagged thin_wrapper; either inline or document why the layer exists (probably for the `StateStore` Protocol — keep + add `# noqa` style note in the audit log only, no source change needed).
- `_NoOpTqdm` class in `transcribe.py:43` — LCOM=55, 7 methods, 0 attributes. Replace with `huggingface_hub.utils.disable_progress_bars()` if available, else move to a `utils/quiet_tqdm.py` module. Single-purpose, doesn't belong in transcribe.

---

## Priority 6 — Cross-cutting findings the per-file analyzer missed (added 2026-05-19)

These came from a pressure-test pass that grepped `tests/`, `scripts/`, `frontend/`, and `docs/` against the audit's proposals, plus an independent cross-module scan.

### 6.1 Delete `backend/test_cuDNN.py` (vestigial GPU smoke test)
- Confirmed no imports; not registered in any test runner; not in `.gitignore`. Same class as `transcribe.py.backup` (P1.1).
- **Action**: `git rm backend/test_cuDNN.py`, or `git mv` to `scripts/` if it's still occasionally useful as a manual smoke. Decide based on whether the team runs it.

### 6.2 CLI traceback-by-argv bypasses logging framework (correctness, not LOC)
`backend/cli/db.py:138–143` and `:197–202` use:
```
except Exception as exc:
    if "--verbose" in sys.argv:
        import traceback
        traceback.print_exc()
    ...
```
This bypasses the rich-tracebacks logger already configured at `backend/cli/config.py:42` (`rich_tracebacks=True`), and the string-in-argv flag check is brittle. Per project Rule #3 (no `print`, structured logging only), both handlers should use `logger.exception(...)` and let the configured handler render the traceback when the log level allows it. Fixes a latent bug, not LOC reduction.

### 6.3 Cross-module `BEGIN TRANSACTION` duplicate inside `database.py`
Separate from the migration column-loop dedup in P2.2. Lines 546–550 and 1543–1547 both wrap `conn.execute("BEGIN TRANSACTION")` in identical try/except-pass guards (DuckDB auto-commit fallback). Extract `_begin_transaction_or_continue(conn)` alongside `_commit_or_rollback` in P2.2; both helpers live in `database/_txn.py` or similar.

### 6.4 (informational, do **not** unify) Two FFmpeg pipelines are intentional coexistence
Pressure-test (2026-05-19) traced the relationship between `backend/preprocess/orchestrator.py` (calls `run_ffmpeg_pipeline` from `preprocess/steps/ffmpeg_pipeline.py`) and `backend/variants/preprocess_steps.py` + `backend/variants/steps.py` (21 direct `ffmpeg.run()` sites). They are **not** duplicates:
- `orchestrator` is the production single-path pipeline (fixed chain, monolithic `PreprocessConfig`, called from `transcribe.transcribe()` and `services/factory.py`).
- `variants/preprocess_steps` is the research/experimentation system (19+ composable steps, per-step `*StepConfig`, used to compare filter chains).
- Git history: orchestrator first (`e9fb300` "preprocessing iter 1"), variants added later (`3e34b01` "preprocessing variants added"), with `tests/integration/test_variant_systems_comparison.py` explicitly labeling them as two systems ("legacy system remains untouched and functional").
- **Action**: none on the cross-pipeline axis. P2.5 (intra-variants `_run_single_filter` extraction) remains valid; do not extend it to absorb the orchestrator path.

### 6.5 (informational, defer) Config schemas across 3 files are layered, not duplicated
`backend/model_config.py` (model presets), `backend/preprocess/config.py` (env-var ingest), and `backend/run_config.py` (composite `RunConfig`) compose rather than duplicate. The "add a field, touch 3 places" friction is real but pressure-testing for boundary reasons should come **before** any consolidation pass — same trap as 6.4. Not actionable in this audit.

### 6.6 Stale archived plan
`docs/plans/archived/archit_audit_problem1_solutions.md` embeds `TranscriptionDatabase(db_path)` example code. With P2.1 patched to keep the class as a façade (see above), the staleness risk is downgraded — the imports remain correct. Leave as-is unless the façade approach changes.

---

## Sequencing & guardrails

1. **P1 first** (1.1 → 1.5) — all of P1 should land as one or two small PRs. Touches few callers; reduces noise for P2 review.
2. **P2.1 (database split)** before **P2.2 (migration dedup)** — easier to add the helpers inside the new modules than to refactor in the 1947-line file then split.
3. **P2.3 + P2.4** can run in parallel; they touch different files.
4. **P3** rides on whichever P2 work first introduces `VariantRunContext`. Don't do P3 in isolation — without P2 refactors, dataclass-ification just renames the problem.

**For every change**:
- `make pyright && make unit` after each commit (project Rule #5).
- After moving symbols across modules, grep `tests/` **and `scripts/`** for `\bOldName\b` — pyright won't catch test-only or script-only `from x import OldName` until they run (per `~/.claude/rules/python-refactors.md` "Removing exported constants"). The pressure-test pass (2026-05-19) found 5 script callers of `TranscriptionDatabase` and 1 of the `backend.variants` re-export wall that the original audit missed; scripts/ is part of the import surface.
- For migrations (P2.2), exercise with a copy of `transcribe_state.db` and confirm `_init_db` is idempotent on a fresh DuckDB file.
- `_migration_006_normalize_runs_schema` carries 375 lines of irreversible schema reshape — do not refactor its body until there is an integration test that round-trips an old-shape DB through it. Move it into a module untouched first; refactor later.
- **P1.3 (flattening `Step.execute`) has the same risk class.** The 100-line if/elif at `variants/steps.py:882` is the only place step-type → executor wiring is encoded; getting one branch wrong silently swaps step behaviour. Before replacing with a dispatch dict, add an integration test that runs each step type at least once (a "every-variant smoke" pass through `scripts/variant_checks/verify_all_variants.py` or equivalent), and verify the new dispatch produces byte-identical output on a representative input per step type. `verify_all_variants.py` imports from `backend.variants.executor` (`transcribe_with_minimal_params`, `create_variant_transcribe_config`), `backend.variants.registry` (`get_builtin_variants`, `get_variant_by_number`), `backend.preprocess.orchestrator` (`PreprocessResult`), and `backend.preprocess.metrics` (`PreprocessMetrics`) — those submodule names are load-bearing for the smoke seam; do not rename them under P1/P2 without coordinated script updates.
- **P2.1 (database split) must preserve the `from backend.database import TranscriptionDatabase` surface** — 5 scripts depend on it (see §2.1). Either keep the class as a façade in `backend/database/__init__.py`, or re-export it from the renamed implementation module. Verify with `python -c "from backend.database import TranscriptionDatabase"` after the split.
- **P1.4 prune of `backend/variants/__init__.py` re-exports must precede or coincide with updating `scripts/compare_transcription_variants.py:18`** to import from submodules. The `backend/__init__.py` `__getattr__` drop is independent and has no script callers.
- **P1.2 collapse of empty `*StepConfig` dataclasses** must also touch `backend/variants/registry.py` and any variant definition files — the empty configs are runtime-constructed by `get_builtin_variants()`, not just declared (see §1.2 caveat).

## Expected outcome (rough)

| Change | LOC delta (est.) |
|--------|---:|
| 1.1 Delete `.backup` | −131 |
| 1.2 + 1.3 `variants/steps.py` flatten | −700 |
| 1.4 Re-export prune | −30 |
| 1.5 `except: pass` → `logger.debug` | +20 (net) |
| 2.1 `database.py` split into 4 modules | net 0 (mostly moves) but `database.py` itself shrinks to ~300 lines |
| 2.2 Migration dedup | −80 |
| 2.3 `transcribe.transcribe()` extract | net 0 |
| 2.4 `variants/executor.py` extract | −150 |
| 2.5 `preprocess_steps.py` dedup | −60 |
| 4. RunSummarizer → functions | −20 |
| 6.1 Delete `test_cuDNN.py` | small (~50) |
| 6.2 CLI logging fix | net 0 (correctness) |
| 6.3 `BEGIN TRANSACTION` extract | −10 |

**Total**: ~−1200 backend lines, two of the three top-3 largest files drop below the 800-line large-file threshold. Note: 6.2 is a latent-bug fix that ships as part of this audit but is not counted as LOC reduction.
