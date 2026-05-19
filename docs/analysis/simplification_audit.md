# Simplification Audit — Backend (May 2026)

**Source**: `python-simplifier` skill (analyze_complexity, find_code_smells, find_overengineering, find_dead_code, find_duplicates, find_coupling_issues, find_unpythonic) run against `backend/` on 2026-05-19.

**Scope**: `backend/` only. `tests/`, `scripts/`, `frontend/`, vendored `.cache/` excluded. False-positives for `from __future__ import annotations` filtered out (always reported as unused; mandatory for PEP 604 union syntax on 3.10).

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

### 1.3 Drop NotImplementedError "executor" classes
`variants/steps.py` defines `FFmpegExecutor`, `SoxExecutor`, `PreprocessStepsExecutor`, `PythonExecutor`, each declared against the `StepExecutor` Protocol. Most methods in each are stubs that raise `NotImplementedError("...should use OtherExecutor")` — analyzer reports LCOM=21 (very low cohesion) for all four.

The "executor" indirection is fake polymorphism: `Step.execute` (L882, cognitive_complexity=76, nesting_depth=9, function_length=103) hard-codes which executor each step type uses via if/elif, so the Protocol's Liskov-substitutable shape never gets exercised.

**Action**: replace 4 executor classes + 1 Protocol with a small set of module-level functions (`run_resample`, `run_loudnorm_only`, `run_denoise`, `run_ffmpeg_pipeline`, ...) and a single `STEP_HANDLERS: dict[str, Callable]` dispatch table. Removes ~700 lines from `steps.py`.

### 1.4 Prune dead re-exports / unused imports
- `backend/variants/__init__.py` re-exports 8 symbols (`execute_variant`, `get_all_variants`, …); analyzer reports none of them imported via the package alias outside of two callsites that already do `from backend.variants import …`. Either keep `__all__` minimal (only the names actually imported through `backend.variants`) or delete the re-export wall.
- `backend/__init__.py` lazy `__getattr__` re-exports `transcribe`, `transcribe_to_json`, `pick_model` — confirm callers; analyzer didn't find call sites. If unused, drop the file body to a one-line module docstring.
- Specific unused imports: `Segment`, `TranscriptionInfo` in `transcribe.py:107`; `Variant` in `services/factory.py:22`; `Variant` in `processor.py:20`; `TranscriptionProcessor` in `components.py:21`.

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

Then `TranscriptionDatabase` becomes a thin coordinator delegating to the modules, or callers import them directly and the class is retired.

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

## Sequencing & guardrails

1. **P1 first** (1.1 → 1.5) — all of P1 should land as one or two small PRs. Touches few callers; reduces noise for P2 review.
2. **P2.1 (database split)** before **P2.2 (migration dedup)** — easier to add the helpers inside the new modules than to refactor in the 1947-line file then split.
3. **P2.3 + P2.4** can run in parallel; they touch different files.
4. **P3** rides on whichever P2 work first introduces `VariantRunContext`. Don't do P3 in isolation — without P2 refactors, dataclass-ification just renames the problem.

**For every change**:
- `make pyright && make unit` after each commit (project Rule #5).
- After moving symbols across modules, `grep -rn '\bOldName\b' tests/` — pyright won't catch test-only `from x import OldName` until the test runs (per `~/.claude/rules/python-refactors.md` "Removing exported constants").
- For migrations (P2.2), exercise with a copy of `transcribe_state.db` and confirm `_init_db` is idempotent on a fresh DuckDB file.
- `_migration_006_normalize_runs_schema` carries 375 lines of irreversible schema reshape — do not refactor its body until there is an integration test that round-trips an old-shape DB through it. Move it into a module untouched first; refactor later.
- **P1.3 (flattening `Step.execute`) has the same risk class.** The 100-line if/elif at `variants/steps.py:882` is the only place step-type → executor wiring is encoded; getting one branch wrong silently swaps step behaviour. Before replacing with a dispatch dict, add an integration test that runs each step type at least once (a "every-variant smoke" pass through `scripts/variant_checks/verify_all_variants.py` or equivalent), and verify the new dispatch produces byte-identical output on a representative input per step type.

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

**Total**: ~−1150 backend lines, two of the three top-3 largest files drop below the 800-line large-file threshold.
