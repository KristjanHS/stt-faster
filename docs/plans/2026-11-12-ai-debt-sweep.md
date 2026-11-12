# stt-faster — AI-debt sweep 2026-11-12

Status: OPEN — rulings below are PROPOSED, not yet ratified (see "Open rulings").
Census: `docs/plans/2026-11-12-ai-debt-sweep.census.json` @ `65611a2`; prior: none.
Baseline gate at census time: `make unit` 442 passed / 1 skipped, `ruff check .` clean, `make pyright` 0 errors.
`<skill>` = `~/.claude/skills/ai-debt-sweep`. Gates ("autofix gate", "refactor gate") are defined in
`<skill>/references/plan-template.md`; guard shapes in `<skill>/references/guards.md`.

## Recommended refactors — the short list

Ranked by payoff. Each maps to a stage below (or to "Judged hints" when it is a one-off fix).

| # | Refactor | Why | Size | Where |
|---|---|---|---|---|
| 1 | Merge the two segment collectors (`executor._collect_executor_segments` → `transcribe._collect_segments`) and share the metrics field extraction | ~50 duplicated lines whose docstring says "Mirrors"; also the two highest cognitive-complexity functions in `backend/` (39 and 30) | M | Stage 8.1 |
| 2 | Port the hand-rolled ffmpeg step bodies in `backend/variants/preprocess_steps.py` onto the existing `_run_single_filter` | 5+ copies of input→filter→output→run→wrap-errors→metrics | M | Stage 8.2 |
| 3 | Give `TranscriptionProcessor` a public seam and hand `FileProcessor` a frozen settings object instead of `processor_ref` | removes private-attribute writes from the CLI, six proxy properties, and the `components ↔ processor` cycle | M | Stages 5.4, 6.1, 6.3 |
| 4 | Break the three remaining import cycles (`diarize`, `variants.steps ↔ preprocess_steps`, `executor ↔ run_config`) | two are hidden behind `TYPE_CHECKING` / function-local imports today | S each | Stage 5.1–5.3 |
| 5 | Replace the vacuous tests in `tests/unit/test_variants.py` | 7 tests cannot fail (two are literally `assert expected_keys` on a constant) while claiming to cover `execute_variant` output | S | Stages 2, 4 |
| 6 | Drop 11 freeze-era transitive pins from `pyproject.toml` base deps | nothing imports them; `uv.lock` already pins them through their real parents | S | Stage 9 |
| 7 | Remove dead `# noqa` comments and turn on `RUF100` | ~100 suppressions name rules the project never selects | S | Stage 3 |
| 8 | Modernise typing (`UP`) and turn the rule on | 97 safe mechanical fixes on Python 3.12 | S | Stages 1, 11 |
| 9 | Fix the 12-times-repeated walrus-lambda idiom in `backend/variants/registry.py` | accidental duplication inside a deliberate table | S | Stage 8.3 |
| 10 | One-off correctness fixes found while judging (silent env-var fallback, wrong `faster_whisper` import, naive timestamp, leaked temp secrets file, a diagnostic that reports "Hub OK" offline) | real defects, but their classes were too noisy to slice | S each | Judged hints H1–H7 |

Deferred (too large or too design-heavy for a debt slice; each needs its own plan): unifying `transcribe()` with
`executor._run_transcription` onto one core; collapsing the `transcribe_to_text` / `transcribe_to_json` /
`transcribe_and_save` wrappers (no callers outside `backend/transcribe.py` and its tests; `output_format="both"`
transcribes twice); collapsing the `_handle_*` adapters in `backend/variants/steps.py`.

## Open rulings

No ask round was run in session 1. Before executing a stage, ratify or change the "Ruling" column for its row.
Behaviour-changing rows that most need a decision: R3 (rewriting tests), R6 (public seam shape), R8 (which clone
families), R9 (dependency pins), R10 (clone guard needs Node in the dev loop).

## Rulings

| # | Class | Default policy | Ruling (PROPOSED) | Scope / exemptions |
|---|---|---|---|---|
| R1 | deprecated-api | judge+ask | fix all, promote `UP017` + `UP035` | all linted dirs; 15/15 judged safe on 3.12 |
| R2 | misplaced-defensiveness | judge+ask | fix the 5 sites | `tests/unit/test_variants.py:365-381` only; exempt `tests/unit/test_setup_gui.py:1296` (wrong annotation, see H6) |
| R3 | weak-oracle | judge+ask | rewrite or delete the 7 tests | `tests/unit/test_variants.py`; rewriting tests is the point of the slice, so the "tests read-only" gate step is waived for Stage 4 only |
| R4 | check-weakening | judge+ask | remove dead `# noqa`, promote `RUF100` | exempt live `# noqa: T201`; keep every `# nosec` and `# type: ignore` |
| R5 | import-cycle | engine | cut all 4 | cuts named per slice |
| R6 | protected-member-access | judge+ask | narrow scope | `backend/` (12 sites) + `tests/unit/test_run_log.py` (4); exempt `tests/unit/test_setup_gui.py` and `tests/unit/test_gui.py` (unbound Tk callbacks driven on fakes, behaviour asserted) |
| R7 | speculative-generality | engine | collapse `StepRegistry` | 1 site |
| R8 | copy-paste-clones | engine | narrow scope: judged families A, B1, C, D, F + `run_log` loop; family E (tests) included | exempt docstring `Args` blocks, signature lists, test arrange boilerplate, the deliberate `backend/gui.py` ↔ `installer/setup_gui.py` mirror |
| R9 | unused-deps | autofix | remove the 11 judged-unused pins | keep `click`, `hf-xet`, `onnxruntime`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `torchaudio` (both extras) |
| R10 | copy-paste-clones guard | tool gate | `make clones` (jscpd via `npx`) wired into pre-push only | no CI step; needs Node on the dev machine (present: v22) |
| R11 | debug-leftovers | autofix | fix, promote `ERA001` | 1 site |
| R12 | legacy-idioms | autofix | fix all, promote `UP` | 15/15 judged safe |
| R13 | swallowed-exception, clock-tz-handling, hallucinated-internal-symbol, inefficient-logic, stdlib-reinvention, unused-exports-files, unused-locals-params, cognitive-complexity, doc-drift, test-smells | various | report-only (precision < 70%) | their real sites are listed under "Judged hints" |

## Coverage

| Adapter / class | Status | Reason |
|---|---|---|
| ruff, complexipy, vulture, basedpyright, deptry, grimp, astgrep-py, shellcheck, scc, jscpd, gitleaks, agnix, pytestsmells, simplifier | ok | — |
| pyprobe (`py_abstraction_probe.py`) | skipped | crashed: `RecursionError` in `ast` unparse → `unused-knobs` and probe-based `needless-indirection` / `speculative-generality` / `class-as-namespace` are UNVERIFIED, not clean |
| bats_smells | skipped | no bats files |
| shfmt | skipped | no `.editorconfig` formatter config → style classes skipped |
| jscpd-semantic | skipped | changed-scope only → `repo-helper-reimplementation` unverified |
| sh_fanin | skipped | TRIAL adapter, never in the default census |
| rls_static, osv-scanner, dep_probe, instr_probe, residue-twin-file, astgrep-satd-agent, fakegreen-diff | skipped | not built in the census |
| layering-violation | subtracted | `tests/unit/test_architecture.py` owns it |
| test-smells: monkeypatch / `mock.patch` | subtracted | pre-commit hook `no-monkeypatch` |
| hardcoded-secret | 770 rows, all unranked | every hit is in a gitignored path (`.cache/`, `.pre-commit-cache/`, `.uv-cache*/`, `.env`); 0 in tracked files → no hand-off |
| unused-locals in `backend/`, `installer/` | already gated | pyright strict + ruff `F` |
| untracked files | not censused as project code | `scripts/windows/transcribe_mixed_{distil,large}_Desk.bat` were untracked at census time |

Census tooling defects seen while judging (they inflate counts; fix in the skill, not here): simplifier
`unused_import` flags `from __future__ import annotations`; `data_class` ignores dunder/private methods;
`py.coupling` fires on pytest grouping classes; ruff adapter runs `--ignore-noqa`, so already-suppressed `ARG` rows
return; `flag-call` counts `getattr(obj, name, default)`; shellcheck `SC2312` rows fail closed and are not swallowed
exceptions.

## Ranked findings

Hits = tracked files only. Precision = judged TP / n on a seeded sample (`random.seed(20261112)`, stratified by tool).

| Rank | Class | Tier | Hits | Δ since | Precision (n, seed) | Policy |
|---|---|---|---|---|---|---|
| 1 | deprecated-api | 2 | 35 | — | 100% (15, 20261112) | autofix → Stage 1 |
| 2 | misplaced-defensiveness | 2 | 6 | — | 83% (6) | judge+ask → Stage 2 |
| 3 | swallowed-exception | 2 | 54 | — | 7% (15) | report-only; hint H1 |
| 4 | inefficient-logic | 2 | 8 | — | 38% (8) | report-only; hints H5, H8, H12 |
| 5 | hallucinated-internal-symbol | 2 | 5 | — | 40% (5) | report-only; hint H2 |
| 6 | clock-tz-handling | 2 | 3 | — | 33% (3) | report-only; hint H3 |
| 7 | dead-branches | 2 | 1 | — | 0% (1) | report-only; hint H6 |
| 8 | check-weakening | 3 | 130 | — | 80% (15) | autofix → Stage 3 |
| 9 | weak-oracle | 3 | 7 | — | 100% (7) | judge+ask → Stage 4 |
| 10 | test-smells | 3 | 134 | — | 0% (15) | report-only (11/15 pure style) |
| 11 | type-escape | 4 | 615 escapes in 50 files | — | not sampled (count class) | report-only count |
| 12 | import-cycle | 5 | 4 | — | 100% (4) | engine → Stage 5 |
| 13 | protected-member-access | 5 | 38 | — | 53% (15); 8/8 in the R6 scope | narrowed → Stage 6 |
| 14 | cognitive-complexity | 5 | 38 | — | 47% (15) | report-only; top `backend/` functions fall to Stage 8.1 |
| 15 | speculative-generality | 5 | 1 | — | 100% (1) | engine → Stage 7 |
| 16 | long-function-file | 5 | 5 | — | not sampled | report-only |
| 17 | needless-indirection | 5 | 2 | — | 0% (2) | report-only |
| 18 | param-sprawl | 5 | 1 | — | 0% (1) | report-only |
| 19 | copy-paste-clones | 6 | 123 | — | 67% (15) | narrowed to judged families → Stage 8 |
| 20 | stdlib-reinvention | 6 | 36 | — | 47% (15) | report-only; hint H13 |
| 21 | magic-literals | 6 | 106 | — | 7% (15) | report-only |
| 22 | unused-deps | 7 | 18 | — | 61% (18 = full population) | per-name verdicts → Stage 9 |
| 23 | unused-exports-files | 7 | 185 | — | 13% (15) | report-only; hints H9, H10 |
| 24 | unused-locals-params | 7 | 97 | — | 13% (15) | report-only; hint H11 |
| 25 | debug-leftovers | 7 | 1 | — | 100% (1) | autofix → Stage 10 |
| 26 | legacy-idioms | 8 | 62 | — | 100% (15) | autofix → Stage 11 |
| 27 | py.complexity | 8 | 211 | — | 60% (15) | report-only feed |
| 28 | py.unpythonic | 8 | 25 | — | 33% (15) | report-only; hint H14 |
| 29 | py.code-smells | 8 | 700 | — | 7% (15) | report-only feed |
| 30 | py.coupling | 8 | 99 | — | 0% (15) | report-only feed |
| 31 | shadowed-variable | 8 | 3 | — | 0% (3) | report-only (Protocol / stdlib-signature names) |
| 32 | overlogging | 8 | 50 sites in 8 files | — | not sampled (count class) | report-only |
| 33 | doc-drift | 9 | 103 | — | 2% (103 = full population) | report-only; hints H15, H16 |
| 34 | comment-saturation.ratio | 9 | 5 | — | not sampled | report-only |
| 35 | comment-drift | 9 | 1 | — | 0% (1) | report-only; hint H17 |

Report-only counts worth knowing:

- type-escape, top files (`Any` + `cast` + `type: ignore` per file): `backend/variants/preprocess_steps.py` 97,
  `tests/unit/test_setup_gui.py` 60, `tests/unit/test_transcribe.py` 54, `backend/transcribe.py` 48,
  `scripts/generate_variant_report.py` 41, `backend/variants/executor.py` 37, `installer/setup_gui.py` 35.
- long-function-file (code lines > repo p90 of 641): `installer/setup_gui.py` 1494, `backend/components.py` 774,
  `scripts/generate_variant_report.py` 752, `backend/transcribe.py` 731, `backend/variants/executor.py` 666.
- cognitive-complexity > 25: `scripts/generate_variant_report.py::main` 86 and `::generate_variant_report` 78,
  `scripts/compare_transcription_variants.py::main` 68, `backend/variants/executor.py::_collect_executor_segments` 39,
  `scripts/new_project_git_setup.py::main` 38, `installer/setup_gui.py::run_process` 34,
  `scripts/diagnose_model_vram.py::main` 34, `backend/transcribe.py::_collect_segments` 30 and `::transcribe` 29,
  `backend/variants/executor.py::_run_transcription` 26.

## Judged hints (not sliced — class precision too low; each is a real site)

| # | Site | What is wrong | Fix |
|---|---|---|---|
| H1 | `backend/preprocess/config.py:96`, `:108` (`_parse_positive_int`, `_parse_float`; ~10 call sites `:239-:402`) | a typo in an env var silently falls back to the default | log a warning naming the variable and the rejected value, or raise a config error |
| H2 | `backend/transcribe.py:58` | `Segment`, `TranscriptionInfo` imported from top-level `faster_whisper`, which does not export them (checked against the installed package: `__all__` lacks both) | `from faster_whisper.transcribe import Segment, TranscriptionInfo` |
| H3 | `backend/cli/transcription_commands.py:422` | `run_meta.json` gets a naive local timestamp while the JSONL run log writes UTC | `datetime.now(UTC).isoformat()` for the JSON; keep the local folder name |
| H4 | `scripts/validate_estonian_models.py:27-32` | logs "Hub access: OK" after only constructing `HfApi()` (no network call) | make one cheap real call inside the `try` |
| H5 | `tests/e2e/test_production_container.py:166-179` | `NamedTemporaryFile(delete=False)` holding secrets is not in a `with`; a failed write leaves it in `/tmp`, then the fallback puts secrets on docker argv | `with` block, unlink in `except`, fail instead of the `-e` fallback |
| H6 | `tests/unit/test_setup_gui.py:1294` | parameter annotated `float \| None` but `...` is passed as a sentinel, so pyright calls a live branch unreachable | annotate `EllipsisType` or use a named sentinel |
| H7 | `backend/variants/executor.py:676` | UNVERIFIED: the executor diarize block has no `num_speakers < 2` guard; `backend/transcribe.py:825` has one and calls itself the single point of truth (`backend/run_config.py:88` also checks) | confirm whether the executor path can be reached with `num_speakers < 2`; resolves itself if the two cores are unified |
| H8 | `backend/run_config.py:71` | mutable default worked around with `# type: ignore[assignment]` | `field(default_factory=dict)` |
| H9 | `backend/variants/preprocess_steps.py:670` | `create_preprocess_runner(base_config=…)` parameter is never read (one caller, `executor.py:86-88`); `:715` hand-rolled `step_index` counter | delete the parameter and its argument; `enumerate` |
| H10 | `tests/integration/conftest.py:10,14-15` | empty `if TYPE_CHECKING: pass` + its import | delete |
| H11 | `tests/unit/test_installer_reuse.py:70` | `calls` unpacked and never asserted — looks like a missing assertion on the v1.1.0 download | assert it, or rename `_calls` |
| H12 | `scripts/compare_transcription_variants.py:138` | chained `startswith` where `:144` already uses the tuple form | tuple `startswith` |
| H13 | `backend/cli/db.py:42`, `:55`; `backend/transcribe.py:702`, `:707`; `backend/services/json_output_writer.py:25`; `tests/unit/test_architecture.py:72`; `scripts/new_project_git_setup.py:51` | stdlib one-liners reimplemented (`fromisoformat` accepts `Z` on 3.12, `x or default`, `Path.write_text` / `read_text`, `Path.cwd()`) | rewrite each; behaviour-identical |
| H14 | `backend/variants/steps.py:576`, `backend/model_config.py:100`, `backend/run_config.py:155`, `scripts/generate_variant_diffs.py:205` | `.keys()` where iterating the dict suffices | drop `.keys()` |
| H15 | `docker/README.md:158`, `:252` | cite `.cursor/plans/archived/architecture_refactoring_plan.md`, which moved to `docs/plans/archived/` | fix the path or delete the stale Ollama pointer |
| H16 | `docs/analysis/simplification_audit.md` | dated audit with 17 dead path cites, outside `archived/` | confirm it is fully consumed, then `git mv` into `docs/plans/archived/` |
| H17 | `backend/transcribe.py:272` | docstring `Args` omits `resolver`, `loader`, `tokenizer_loader` | add them |
| H18 | `scripts/variant_checks/` | excluded from ruff, carries 40 dead `# noqa`, and `verify_all_variants.py` uses `mock.patch` (banned in tests) | decide: delete the directory if unused, or bring it under ruff |
| H19 | `tests/conftest.py:101`, `:119` | unexplained HF-token minimum length `30`, duplicated | name it `_MIN_HF_TOKEN_LEN` |
| H20 | `scripts/generate_variant_report.py` | `main` (86), `generate_variant_report` (78, ~265 lines) | split only if the script is still in use |

## Hand-offs

- `security-reviewer`: none. All 770 gitleaks rows are in gitignored cache directories or `.env`.

## Stage 1 — deprecated-api (tier 2)

### Stage 1.1 — deprecated-api in `backend/`

- Sites: 27 rows (`UP017` `timezone.utc` → `UTC`, e.g. `backend/components.py:239,241,249,251,781,917,933`;
  `UP035` `typing.Callable/Mapping/List/…` imports in `backend/preprocess/`, `backend/services/`,
  `backend/variants/variant.py`, `backend/diarize/pipeline.py`, `backend/model_loader.py`, `backend/run_config.py`,
  `backend/cli/transcription_commands.py`, `backend/transcribe.py`)
- Op: autofix rule `UP017`, `UP035`
- Fix: `.venv/bin/ruff check backend --select UP017,UP035 --fix`
- Guard: none yet (lands in 1.2)
- Gate: autofix gate
- Done when: 0 `UP017`/`UP035` hits in `backend/`, pyright clean, suite green

### Stage 1.2 — deprecated-api in `scripts/` + `tests/`, then promote

- Sites: 8 rows (`scripts/generate_variant_report.py` ×3, `tests/` ×5)
- Op: autofix rule `UP017`, `UP035`
- Fix: `.venv/bin/ruff check scripts tests --select UP017,UP035 --fix`
- Guard: lint-rule promotion — add `"UP017", "UP035"` to `[tool.ruff.lint] select` in `pyproject.toml`; must-fail
  demo in the commit message
- Gate: autofix gate
- Done when: `ruff check .` clean with the rules selected, guard fails on a seeded `timezone.utc`, suite green

## Stage 2 — misplaced-defensiveness (tier 2)

### Stage 2.1 — misplaced-defensiveness in `tests/unit/`

- Sites: `tests/unit/test_variants.py:365`, `:369`, `:373`, `:378`, `:381` (`hasattr` / `is not None` guards on typed
  dataclass fields wrapping `isinstance` tautologies)
- Op: `rewrite` — delete the always-true guards; assert the value constraints unconditionally
- Fix: LLM text edit (guard inversion), one test function
- Guard: lint-rule promotion — `reportUnnecessaryComparison` and `reportUnnecessaryIsInstance` at `"error"` in
  `pyrightconfig.json` (0 hits in the included dirs today). Known gap: `tests/` is outside pyright's `include`, so
  this guard protects `backend/`, `scripts/`, `installer/` only
- Gate: refactor gate
- Done when: basedpyright reports 0 `reportUnnecessaryComparison` in the file, guard fails on a seeded always-true
  comparison in `backend/`, suite green

## Stage 3 — check-weakening (tier 3)

### Stage 3.1 — dead `# noqa` in `backend/` + `installer/`

- Sites: 30 census rows (`backend/variants/executor.py` 6, `backend/gui.py` 5, `backend/cli/transcription_commands.py`
  3, `installer/setup_gui.py` 9, …). Codes never selected by the project: `ANN001`, `ANN401`, `ARG001`, `ARG005`,
  `S603`, `S606`, `PLC0415`. Fewer rows will change than the census counts: `# noqa: T201` is live and stays
- Op: autofix rule `RUF100`
- Fix: `.venv/bin/ruff check backend installer --extend-select RUF100 --fix` (project config, so live codes survive)
- Guard: none yet (lands in 3.2)
- Gate: autofix gate; also confirm `make bandit` unchanged (every `# nosec` must survive)
- Done when: 0 `RUF100` under the project config in both dirs, suite green

### Stage 3.2 — dead `# noqa` in `scripts/` + `tests/`, then promote

- Sites: `scripts/generate_variant_report.py`, `scripts/debug_nospeech_prob.py`, `tests/unit/test_transcribe.py`,
  `tests/unit/test_denoise_light_step.py`, `tests/unit/test_setup_gui.py`, `tests/unit/test_preprocess.py`,
  `tests/unit/test_variants.py`, `tests/unit/test_diarize_pyannote_runner.py`
- Op: autofix rule `RUF100`
- Fix: `.venv/bin/ruff check scripts tests --extend-select RUF100 --fix`
- Guard: lint-rule promotion — add `"RUF100"` to `[tool.ruff.lint] select`; must-fail demo in the commit message
- Gate: autofix gate
- Done when: `ruff check .` clean with `RUF100` selected, guard fails on a seeded `# noqa: ANN001`, suite green
- Out of scope: `scripts/variant_checks/` (40 rows) is in ruff `exclude`; see H18

## Stage 4 — weak-oracle (tier 3)

### Stage 4.1 — weak-oracle in `tests/unit/test_variants.py`

- Sites: `:27`, `:81`, `:161`, `:187`, `:313`, `:333`, `:469`
- Op: `rewrite` — `:187` and `:469` (`assert expected_keys` on a constant) and `:161` (never calls
  `execute_variant`) become real calls with an injected fake, following the existing test at `:202`; `:27`, `:81`,
  `:313`, `:333` get value-pinning assertions or are deleted where another test already covers them (`:81`
  duplicates `:150`)
- Fix: LLM text edit; no production code changes
- Guard: count ratchet — copy `<skill>/scripts/py_test_smells.py` to `tests/guards/py_test_smells.py`, add
  `tests/unit/test_debt_ratchet.py` reading `tests/guards/debt-baseline.json` (keys `weak-oracle::<file>::<test>`);
  must-fail demo in the commit message
- Gate: refactor gate with step 8 (tests read-only) waived per R3; every rewritten test must be shown to fail once
  against a deliberately broken `execute_variant` result
- Done when: `py_test_smells` reports 0 `sole-weak-assert` in the file, the ratchet fails on a seeded
  `assert x is not None`-only test, suite green

## Stage 5 — import-cycle (tier 5)

Guard for the whole stage: extend `tests/unit/test_architecture.py` with an acyclicity test over `backend/` built
from every `ast.Import` / `ast.ImportFrom` node (including function-local and `TYPE_CHECKING` ones, which is what
the census counted) with an allowlist of the cycles still open. 5.1 adds the test with three allowlisted cycles;
each later slice deletes one entry.

### Stage 5.1 — import-cycle in `backend/diarize/`

- Sites: `backend/diarize/pyannote_runner.py:38` ↔ `backend/diarize/pipeline.py`
- Op: `move` — `SpeakerTurn` into new `backend/diarize/types.py`; keep the lazy torch import in `pipeline.annotate`
- Fix: `rope` move (`<skill>/references/tools-python.md` §Refactor engines)
- Guard: acyclicity test added here (allowlist: the other three cycles); must-fail demo in the commit message
- Gate: refactor gate
- Done when: grimp reports no cycle among `backend.diarize` children, guard fails on a seeded back-import, suite green

### Stage 5.2 — import-cycle `backend.variants.executor` ↔ `backend.run_config`

- Sites: `backend/variants/__init__.py:3` (eager `execute_variant` re-export), `backend/variants/executor.py:35` plus
  its two function-local `run_config` imports
- Op: `delete-knob` — drop the re-export, then hoist the local imports and delete the `TYPE_CHECKING` workaround
- Fix: grep `from backend.variants import` across `backend scripts tests` first (single-line grep found only
  direct `backend.variants.executor` importers); LLM text edit
- Guard: remove this cycle from the allowlist
- Gate: refactor gate
- Done when: cycle gone in grimp, allowlist shrunk, suite green

### Stage 5.3 — import-cycle `backend.variants.steps` ↔ `backend.variants.preprocess_steps`

- Sites: `backend/variants/steps.py:29`, `backend/variants/preprocess_steps.py:20`
- Op: `move` — `_execute_step_with_registry`, `create_preprocess_runner`, `_copy_intermediate_file` into new
  `backend/variants/preprocess_runner.py`; update the `executor.py` import. No re-export shim
- Fix: `rope` move
- Guard: remove this cycle from the allowlist
- Gate: refactor gate
- Done when: cycle gone in grimp, `preprocess_steps.py` no longer imports `steps`, suite green

### Stage 5.4 — import-cycle `backend.components` ↔ `backend.processor`

- Sites: `backend/components.py:22` (`TYPE_CHECKING` import that exists only for `processor_ref`)
- Op: `rewrite` — `FileProcessor` takes an immutable settings value object instead of the processor; the six proxy
  properties (incl. `components.py:573`, `:585`) go away
- Fix: LLM text edit; ratify the settings object's fields under R6 before starting
- Guard: remove the last allowlist entry (allowlist is now empty and deleted)
- Gate: refactor gate
- Done when: `backend/components.py` has no import of `backend.processor`, guard has no allowlist, suite green

## Stage 6 — protected-member-access (tier 5)

### Stage 6.1 — protected-member-access in `backend/cli/`

- Sites: `backend/cli/transcription_commands.py:349-351` (writes `_output_base_dir`, `_variant_number`, and a third
  private field on the processor); tests `tests/unit/test_transcribe_processor.py:635`, `:709` do the same
- Op: `extract` — public `TranscriptionProcessor.set_variant_output(run_folder, number, name)` (or constructor fields)
- Fix: LLM text edit
- Guard: none yet (lands in 6.4)
- Gate: refactor gate; surface check expects exactly one new public method
- Done when: no `._` write on a processor outside `backend/processor.py`, suite green

### Stage 6.2 — protected-member-access in `backend/variants/`

- Sites: `backend/variants/executor.py:113-153`, `:943` (`hasattr(cfg, "_explicit_fields")` reach-ins)
- Op: `extract` — public `TranscriptionConfig.explicit_fields` (frozenset property); drop the `hasattr` probes, the
  parameter is already typed
- Fix: LLM text edit
- Guard: none yet
- Gate: refactor gate
- Done when: 0 `SLF001` in `backend/variants/`, suite green

### Stage 6.3 — protected-member-access in `backend/components.py`

- Sites: remaining `SLF001` rows in `backend/components.py` after 5.4 (`_file_mover` reach-ins inside
  `FileProcessor.process_file`, `:587`)
- Op: `extract` — `FileMoverPolicy.move_to_processed(audio, created_files)`; guard clause in `process_file`
  (this is also its cognitive-complexity seam, 19 today)
- Fix: LLM text edit
- Guard: none yet
- Gate: refactor gate
- Done when: 0 `SLF001` in `backend/`, `process_file` complexity lower, suite green

### Stage 6.4 — protected-member-access in `tests/unit/test_run_log.py`, then promote

- Sites: `tests/unit/test_run_log.py:26`, `:30`, `:34`, `:113` (assert on private `_next_id`; the comment at
  `:108-112` has rotted)
- Op: `rewrite` — assert through the ids `append()` returns
- Fix: LLM text edit (test file; allowed, it is the slice's subject)
- Guard: lint-rule promotion — `"SLF001"` in `[tool.ruff.lint] select` with
  `[tool.ruff.lint.per-file-ignores]` for `tests/unit/test_setup_gui.py` and `tests/unit/test_gui.py` (R6
  exemption); must-fail demo in the commit message
- Gate: refactor gate
- Done when: `ruff check .` clean with `SLF001` selected, guard fails on a seeded `obj._x` in `backend/`, suite green

## Stage 7 — speculative-generality (tier 5)

### Stage 7.1 — speculative-generality in `backend/variants/steps.py`

- Sites: `backend/variants/steps.py:571` (`StepRegistry`: classmethod namespace over a dict, "kept for compat"; one
  production caller, one test — `tests/integration/test_every_step_smoke.py`)
- Op: `collapse` — module-level `create_step()` and `registered_step_types()`
- Fix: LLM text edit after 5.3 (the caller has moved to `preprocess_runner.py` by then)
- Guard: add to `tests/unit/test_architecture.py`: no class under `backend/` whose body is only
  `@classmethod` / `@staticmethod` members and class attributes; must-fail demo in the commit message
- Gate: refactor gate
- Done when: `StepRegistry` is gone, guard fails on a seeded static-only class, suite green

## Stage 8 — copy-paste-clones (tier 6)

Guard for the whole stage (R10): `make clones` = `npx --yes jscpd@5.4.2 backend scripts --min-tokens 50
--baseline-from-ref <stage base> --fail-on-new-clones 0` with the census ignore list (flags per
`<skill>/references/tools-multi.md` §jscpd guard — confirm them against `jscpd --help` at that pin), called from
`scripts/git-hooks/pre-push`. Added in 8.1.

### Stage 8.1 — family B1: segment collectors in `backend/transcribe.py` ↔ `backend/variants/executor.py`

- Sites: `backend/variants/executor.py:363` (`_collect_executor_segments`, cognitive 39) ≡
  `backend/transcribe.py:516` (`_collect_segments`, cognitive 30); the preprocess-step / loudnorm / denoise field
  extraction shared by `_build_metrics_payload` and `_build_executor_metrics`
- Op: `extract` — `_collect_segments` gains optional thresholds, the executor copy is deleted; one shared
  `_preprocess_step_fields(result)`
- Fix: LLM text edit; differential per refactor-gate step 11 over the repo's fixtures
- Guard: `make clones` + pre-push wiring; must-fail demo in the commit message
- Gate: refactor gate
- Done when: jscpd shows the family gone, complexipy for the surviving function ≤ 30, suite green

### Stage 8.2 — family A: ffmpeg step bodies in `backend/variants/preprocess_steps.py`

- Sites: `:125`, `:196`, `:362` and siblings (`volume_with_limiter`, `loudnorm_with_highpass`,
  `highlow_aform_loudnorm`, `highlow_nosampl_loudnorm`, `compressor_with_limiter`) duplicating `_run_single_filter`
- Op: `inline` onto `_run_single_filter(filter_chain=…)`; keep `resolve_loudnorm_params` inside the error-wrapping
  path so failures still raise `StepExecutionError`
- Fix: LLM text edit, one step function per commit if the diff exceeds ~300 lines
- Guard: `make clones` baseline moves down
- Gate: refactor gate; characterize each step's built ffmpeg argv before the edit
- Done when: jscpd shows the family gone, argv snapshots unchanged, suite green

### Stage 8.3 — family C: `backend/variants/registry.py`

- Sites: 12 × `(lambda: (config := create_baseline_config(), config.set(…), config)[N])()` (e.g. `:201`)
- Op: `extract` — `_baseline_with(**overrides) -> TranscriptionConfig`
- Fix: LLM text edit
- Guard: `make clones` baseline moves down
- Gate: refactor gate; characterize every variant's resolved config before the edit
- Done when: 0 walrus-lambda blocks, resolved configs identical, suite green

### Stage 8.4 — family D: `_configure_logging`

- Sites: `scripts/transcribe_manager.py:27` ≡ `backend/cli/transcription_commands.py:135`
- Op: `inline` — make the backend one public, import it in the script; first check whether the script's call is
  redundant (`cmd_process` already calls it at `:499`)
- Fix: LLM text edit
- Guard: `make clones` baseline moves down
- Gate: refactor gate
- Done when: one definition, suite green

### Stage 8.5 — family F + `run_log` loop

- Sites: `scripts/check_gpu.py:39` (`check_cuda_libraries` ≡ `check_cudnn_libraries`); `backend/run_log.py`
  (`_next_id`, `tail`, `get` repeat the read / strip / parse / skip loop)
- Op: `extract` — `_check_shared_lib(soname, label)`; `JsonlRunLog._iter_records()` generator
- Fix: LLM text edit, two commits (one per file)
- Guard: `make clones` baseline moves down
- Gate: refactor gate
- Done when: both families gone, suite green

### Stage 8.6 — family E: `tests/unit/test_preprocess_steps.py`

- Sites: one class per step with identical arrange / act / assert (e.g. `:239`)
- Op: `rewrite` — one `pytest.mark.parametrize` table of `(fn, filter_name, metric_name)`; run after 8.2
- Fix: LLM text edit (test file is the slice's subject)
- Guard: none new (jscpd guard ignores `tests/`)
- Gate: refactor gate with the test-read-only step waived; collected test count for the file must not drop
- Done when: same number of test cases collected, suite green

## Stage 9 — unused-deps (tier 7)

### Stage 9.1 — unused-deps in `pyproject.toml`

- Sites: base-dependency pins `anyio`, `certifi`, `filelock`, `fsspec`, `httpcore`, `httpx`, `idna`, `protobuf`,
  `pyyaml`, `shellingham`, `sympy` (all arrived in the first commit as a requirements freeze; nothing imports them;
  the `sympy` coexistence comment near `pyproject.toml:34` is stale since torch moved to the extras)
- Op: autofix (manifest + lockfile)
- Fix: delete the 11 lines, `uv lock`, `make export-reqs`; diff `uv.lock` and stop if any resolved version moves
  (the pins should already be satisfied transitively)
- Guard: tool gate — a `make deptry` target (`uvx deptry==0.25.1 .`) with `DEP002` per-rule ignores for the seven
  deliberate runtime pins (R9), called from `make pre-commit`. Repair `[tool.deptry]` in the same commit: it still
  reads `requirements_files = ["requirements.txt"]` and lists `frontend` as first-party; must-fail demo in the
  commit message
- Gate: autofix gate + `tests/unit/test_lean_deps.py`, `tests/unit/test_no_transitive_cuda.py`,
  `make verify-variants`; check the installer's `HF_TOOL_PINS` (`installer/setup_gui.py:47`) is unaffected
- Done when: deptry reports 0 `DEP002`, no resolved version changed in `uv.lock`, guard fails on a seeded unused
  dependency, suite green

## Stage 10 — debug-leftovers (tier 7)

### Stage 10.1 — debug-leftovers in `backend/variants/registry.py`

- Sites: `backend/variants/registry.py:338` (commented-out old `active_variant_numbers` set)
- Op: autofix rule `ERA001` (manual delete; the fixer is not safe-labelled)
- Fix: delete the line
- Guard: lint-rule promotion — `"ERA001"` in `[tool.ruff.lint] select`; must-fail demo in the commit message
- Gate: autofix gate
- Done when: `ruff check .` clean with `ERA001` selected, suite green

## Stage 11 — legacy-idioms (tier 8)

### Stage 11.1 — legacy-idioms in `backend/`

- Sites: 31 rows (`backend/transcribe.py` 19, `backend/preprocess/io.py` 3, `backend/preprocess/config.py` 2,
  `backend/preprocess/orchestrator.py` 2, `backend/run_config.py` 2, `backend/diarize/pyannote_runner.py`,
  `backend/model_loader.py`, `backend/variants/steps.py`)
- Op: autofix rule `UP006`, `UP007`, `UP037`, `UP045`
- Fix: `.venv/bin/ruff check backend --select UP006,UP007,UP037,UP045 --fix`
- Guard: none yet (lands in 11.2)
- Gate: autofix gate
- Done when: 0 hits in `backend/`, pyright clean, suite green

### Stage 11.2 — legacy-idioms in `scripts/` + `tests/`, then promote

- Sites: 31 rows (`scripts/generate_variant_report.py` 26, `tests/` 5)
- Op: autofix rule `UP006`, `UP007`, `UP037`, `UP045`
- Fix: `.venv/bin/ruff check scripts tests --select UP006,UP007,UP037,UP045 --fix`
- Guard: lint-rule promotion — replace the Stage 1 entries with `"UP"` in `[tool.ruff.lint] select` (the census ran
  all of `UP`: every hit is in Stage 1 or Stage 11, so it lands clean); must-fail demo in the commit message
- Gate: autofix gate
- Done when: `ruff check .` clean with `UP` selected, suite green

## Stage 12 — Verify + review

1. `python3 <skill>/scripts/census.py . --since docs/plans/2026-11-12-ai-debt-sweep.census.json` → each fixed class
   shows count down, no new keys. (The census took ~20 minutes here; the `simplifier` adapter is 8 of them.)
2. `pytest tests/unit tests/integration -q --no-cov`, `make pyright`, `ruff check .`, `make pre-commit`.
3. Dispatch `code-reviewer` over `65611a2..HEAD`.
