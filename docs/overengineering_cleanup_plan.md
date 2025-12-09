# overengineering_cleanup_plan.md

This repo is doing three jobs at once: **STT tool**, **generic template**, and **AI-agent playground**. For a one-person transcription project, that’s more moving parts than you need. Below are the **10 highest-value simplifications**, ordered roughly by “impact / effort”.

---

## 1. Strip out the “new project bootstrap” template system

You’re carrying a full “turn this into a generic template” layer inside an STT project:
`scripts/new_project_bootstrap.sh`, `scripts/new_project_cleanup.sh`, `scripts/new_project_git_setup.py`, `.bootstrap_slug`, plus matching docs.

**Symptoms**

* Shell + Python automation to **delete app-specific files** and rebrand the repo for a different project.
* Docs (`docs/new_project_guide.md`) that explain how to use this repo as a template instead of as an STT app. 

**Why it’s overkill**

* You rarely re-template this repo; it clutters day-to-day work.
* It confuses the repo’s identity for future you and any collaborator: “Is this an STT project or a generic template engine?”

**Refactor steps**

1. Create a new repo (or Gist) called e.g. `repo-bootstrap-tools` and move:

   * `scripts/new_project_bootstrap.sh`
   * `scripts/new_project_cleanup.sh`
   * `scripts/new_project_git_setup.py`
   * `docs/new_project_guide.md`
   * Any `.bootstrap_*` markers.
2. Remove references to these scripts from README and any docs here.
3. Add a short note in `README.md`:

   * “This repo used to contain generic bootstrap tooling; that now lives in <link>. This repository is purely an STT tool.”

**Effort / Risk**

* Effort: **S**
* Risk: **Low** (you’re just moving rarely-used helpers out)

**Impact**

* Clears identity confusion.
* Shrinks cognitive surface area so “what matters” is obvious: preprocessing + transcription.

---

## 2. Consolidate AI-agent configs and rules into one minimal source of truth

You have **multiple overlapping AI-assistant configs**: `.cursor/rules/*`, `AGENTS.md`, `docs/AI_instructions.md`, `docs/CODEX_RULES.md`, `.gemini/*`, `.agent/*`.

**Symptoms**

* Different tools (Cursor, Gemini, Agent, Codex) all have their own rule sets.
* Rules partially duplicate and partially contradict each other (e.g., which tools to prefer, how to use Docker, how to run tests).

**Why it’s overkill**

* You’re the only regular contributor; you don’t need a full “multi-agent governance framework”.
* Every new agent has to wade through multiple docs and configs before it can safely change anything.

**Refactor steps**

1. Pick **one canonical doc** for AI instructions (e.g. `docs/AI_instructions.md`) and:

   * Merge in the non-contradictory parts of `AGENTS.md` and `docs/CODEX_RULES.md`.
2. In `AGENTS.md`, replace the current content with a 5–10 line pointer:

   * “For all agents, read `docs/AI_instructions.md` first.”
3. Slim down `.cursor/rules/`:

   * Keep only rules that genuinely add value (e.g. project context, testing commands).
   * Archive niche rules into `docs/archived/AI_rules_legacy.md`. 
4. Delete `.gemini/*` and `.agent/*` **if you’re not actively using them**; otherwise, point them at the single canonical doc instead of embedding rules. 

**Effort / Risk**

* Effort: **S–M**
* Risk: **Low** (worst case, an agent needs minor retraining)

**Impact**

* AI agents stop fighting conflicting rule sets.
* Future you won’t waste working memory on “which AI instructions are the real ones?”

---

## 3. Simplify GitHub Actions: keep 1–2 workflows, archive the rest

Your `.github/workflows` folder is running a **mini-platform**: CodeQL (x2), Semgrep (x2), meta-linters, trivy/pip-audit, local/act-specific variants, auto-PR flow, etc.

**Symptoms**

* 8–9 workflows, many with `if: github.actor == 'nektos/act'` branches.
* Overlap between local and CI versions of the same tool (e.g., `semgrep.yml` vs `semgrep_local.yml`).

**Why it’s overkill**

* For a small, single-user project, every extra workflow is another failure mode and another thing to debug when CI breaks.
* The complexity mostly serves experiments with `act`, not delivering better transcriptions.

**Refactor steps**

1. Decide **CI MVP**:

   * One `python-lint-test.yml` (ruff + pytest + maybe pyright).
   * Optionally **one** security job (either CodeQL **or** Semgrep, not both).
2. Move non-essential workflows to `docs/archived/ci_examples/` as YAML examples (not active CI).
3. Remove `*_local.yml` variants that are only for `act` experiments.
4. Update `README` and `docs/testing_approach.md` to mention only the surviving workflows.

**Effort / Risk**

* Effort: **M**
* Risk: **Low** (CI will actually become *easier* to reason about)

**Impact**

* CI becomes predictable and debuggable.
* You stop burning time on “why did Semgrep_local under act fail *this* time?”

---

## 4. Right-size local hooks & security tooling

Scripts under `scripts/git-hooks/` plus `detect-secrets`, `.secrets.baseline`, Semgrep configs, Trivy/pip-audit, yamlfmt, etc., make your **local dev workflow** heavier than many production teams’.

**Symptoms**

* Pre-push hook orchestrates `act`, Docker, UV cache fiddling, optional security scans, and test selection. 
* A full `detect-secrets` baseline, yamlfmt config, Semgrep ignore file, Trivy/pip-audit workflow for deps.

**Why it’s overkill**

* For a private project, you gain little from industrial-strength sec-ops at every push.
* When these tools break (network, cache, Docker), they block actual work.

**Refactor steps**

1. Simplify `scripts/git-hooks/pre-commit`:

   * Keep **ruff + basic formatting + maybe detect-secrets**.
   * Remove anything that depends on network or Docker (no Trivy, no `act` here).
2. Simplify `scripts/git-hooks/pre-push`:

   * Option A: only run `pytest` (fast subset).
   * Option B: make it a **no-op** by default and document an opt-in “heavy local CI” script for rare use.
3. Mark heavy security tooling as **“manual/occasional”**:

   * Add `make security-scan` that runs Semgrep/Trivy on demand instead of on every push.
4. Keep `.secrets.baseline` but consider reducing its scope to just real risk areas (scripts, backend). 

**Effort / Risk**

* Effort: **M**
* Risk: **Low** (you can always re-add tools if you miss them)

**Impact**

* You get your git workflow back to “run tests, commit, push” without wrestling infra every time.
* Security scanning becomes deliberate, not constant friction.

---

## 5. Simplify Docker stack & container tests

You’re operating **two Docker builds** (`Dockerfile` + `docker/app.Dockerfile`), a dev docker-compose stack, production container docs, plus detailed e2e tests for “cloud-native” compliance.

**Symptoms**

* `docker/app.Dockerfile` + `docker/docker-compose.yml` used purely for test containers.
* `PRODUCTION_CONTAINER.md` + `scripts/verify_production_image.sh` + `tests/e2e/test_production_container.py` assert labels, image size < 2GB, non-root UID, venv in PATH, etc.

**Why it’s overkill**

* You’re not shipping this image to Kubernetes customers today.
* Maintaining two Dockerfiles and detailed OCI-label tests for a personal STT tool is pure overhead.

**Refactor steps**

1. Standardize on **one Dockerfile** at repo root:

   * Confirm it can run `python -m backend.main` and `scripts/transcribe_manager.py` reliably.
2. Delete `docker/app.Dockerfile` and `docker/docker-compose.yml` **unless** you’re actively running CI inside Docker.
3. Replace `tests/e2e/test_production_container.py` with a **single smoke test**:

   * Build image.
   * Run a short command inside: `--help` or one tiny transcription.
4. Demote `PRODUCTION_CONTAINER.md` to “notes” or move to `docs/archived/` if you don’t have real cloud deployment plans. 

**Effort / Risk**

* Effort: **M**
* Risk: **Low–M** (if you rely heavily on docker-compose in CI, double-check before deletion)

**Impact**

* Containers become boring and predictable: one image, one basic smoke test.
* Future changes to pipeline don’t require synchronizing two Docker setups and multiple e2e suites.

---

## 6. Consolidate user-facing entrypoints (one obvious way to run it)

Right now, you have multiple ways to run the tool:

* `scripts/transcribe_manager.py`
* Windows batch files (`scripts/windows/transcribe_*.bat`)
* Docker wrapper (`scripts/transcribe-docker`)
* Legacy “helpfiles” under `docs/Transcription_helpfiles_win/`.

**Symptoms**

* Different entrypoints hint at different “canonical” paths.
* Some Windows scripts hard-code paths and variant numbers (e.g., variant 16) which may drift from the main CLI defaults.

**Why it’s overkill**

* For your own use, you only need **one** clearly documented path (plus maybe **one** helper `.bat`).
* Extra entrypoints increase support surface and confusion.

**Refactor steps**

1. Declare **single canonical CLI**: `scripts/transcribe_manager.py process ...` (and document it in README).
2. Keep **one** Windows `.bat` wrapper (e.g. `transcribe_estonian.bat`) but:

   * Make it a thin wrapper around the canonical CLI (no extra logic).
   * Point users to that file from docs. 
3. Move old `Transcription_helpfiles_win` and any unmaintained `.bat` files to `docs/archived/`.
4. Update `scripts/transcribe-docker` to simply call the canonical CLI inside the container (and keep only if you genuinely use docker). 

**Effort / Risk**

* Effort: **S–M**
* Risk: **Low**

**Impact**

* There is a single, memorable way to run the tool.
* You reduce the “which command was the supported one again?” mental tax.

---

## 7. Finalize migration to the new variants system and delete legacy paths

The new `backend/variants/` system is now the real source of truth for preprocessing + transcription presets. You still carry legacy/compatibility code and extra scripts built around the old approach.

**Symptoms**

* `backend/variants/*.py` defines variants, presets, and an executor, but some scripts/tests still treat variants as an “experiment layer”.
* Legacy behaviors and flags in CLI/tests to support both old and new paths (e.g., variant selection and manual param sets).

**Why it’s overkill**

* Running **two conceptual systems** (legacy + new) just to be safe is valid during migration, not long-term.
* Every feature change risks being applied to one path but not the other.

**Refactor steps**

1. Make `backend/variants` the **only** way variants are defined:

   * Remove any legacy variant enums or configuration scattered in scripts. 
2. Ensure `scripts/transcribe_manager.py` and Windows `.bat` use **only** the new variant executor.
3. Remove flags/env vars that toggle between old and new behavior.
4. Update tests to:

   * Exercise variants via `backend/variants/executor.py` only.
   * Delete or archive tests that were purely for validating the migration.

**Effort / Risk**

* Effort: **M**
* Risk: **M** (do it once, then run full test suite + one manual run on real audio)

**Impact**

* Variants become stable, not a perpetual experiment.
* Simplifies mental model: “variant X = this exact pipeline, defined in one place.”

---

## 8. Move “variants comparison” tooling into an `experiments/` area

Scripts like `scripts/compare_transcription_variants.py` and `scripts/generate_variant_diffs.py` are excellent **research tools**, but they’re not needed for everyday transcription.

**Symptoms**

* Complex logic for diffing texts, generating Markdown reports, searching variant files by base name. 
* Special-cased in architecture tests (allowed to be huge because they’re “utility scripts”). 

**Why it’s overkill**

* Mixing “pipeline lab” tools with production scripts inflates the repo’s perceived complexity.
* Future you might be scared to touch them even if you’re not running variant comparisons anymore.

**Refactor steps**

1. Create `experiments/variants/` directory.
2. Move:

   * `scripts/compare_transcription_variants.py`
   * `scripts/generate_variant_diffs.py`
   * Any test data strictly used by those experiments (JSON comparison outputs, etc.).
3. Adjust imports/tests so they still run if you explicitly call them, but they’re clearly **“experimental”**.
4. Update `docs/variants_refactoring_plan.md` to reference `experiments/variants` as the place for further tinkering. 

**Effort / Risk**

* Effort: **S–M**
* Risk: **Low** (mostly moving files and fixing imports)

**Impact**

* Clear separation between **production pipeline** and **lab tooling**.
* You keep the value of your experiments without them bloating the “core” mental model.

---

## 9. Narrow configuration surface to STT-only and centralize it

`backend/config.py`, `.env` expectations, and docs still carry config for now-removed RAG/LLM features (embedding models, Ollama, etc.).

**Symptoms**

* Config/table entries for many transcription and model params, some of which are no longer used. 
* Old docs mention LangChain / Ollama / embeddings that no longer exist in `pyproject`. 

**Why it’s overkill**

* Dead configuration is a classic overengineering smell: “we might want this later”.
* New contributors (or future you) have to scan through unused options and wonder if they matter.

**Refactor steps**

1. In `backend/config.py`, remove constants and env-var wiring for:

   * Any RAG/LLM/Ollama/embedding bits not supported by the current code.
2. Introduce a small `AppConfig` dataclass for **only** what you genuinely use:

   * Model preset, device, compute type.
   * Preprocess defaults (rnnoise mix, loudnorm preset, etc.).
   * Database path, log level.
3. Keep a single `.env.example` that documents exactly those fields.
4. Update docs (`docs/Transcription_solution.md`, `README`) to match the **current** feature set only.

**Effort / Risk**

* Effort: **M**
* Risk: **Low–M** (do a project-wide search to ensure removals are actually unused)

**Impact**

* Config becomes understandable at a glance.
* You stop carrying mental debt for features that don’t exist.

---

## 10. Right-size architecture tests and documentation

You have architecture tests that assume multi-layer system boundaries (frontend vs backend, script thinness, etc.), plus detailed architecture docs and diagrams in `.cursor/plans/`.

**Symptoms**

* `tests/unit/test_architecture.py` enforces constraints for a frontend layer that’s basically empty.
* `.cursor/plans/architecture_*` plus `REFACTORING_SUMMARY.md` describe a much larger, multi-layer architecture than you’re actually running.

**Why it’s overkill**

* Architecture tests for non-existent layers become noise, not safety.
* Your docs describe an idealized system; they no longer match the actual scope and can mislead.

**Refactor steps**

1. Trim `tests/unit/test_architecture.py` to invariants that *actually protect you*:

   * “scripts contain minimal logic (thin wrappers)”.
   * “backend has no circular imports”. 
2. Remove or comment out tests for a future frontend until it exists.
3. Move the older big-bang architecture plans into `docs/archived/architecture_v1/`.
4. Add a short, **current** architecture sketch to `README` (one diagram, one paragraph).

**Effort / Risk**

* Effort: **S**
* Risk: **Low**

**Impact**

* Tests focus on real, present risks instead of hypothetical layers.
* Architecture docs stop overselling what the project is.

---

## Suggested execution order

To keep this manageable with limited working memory, tackle it in three passes:

1. **Meta/ceremony first (low risk, big clarity)**

   * Refactor 1 (template), 2 (AI agents), 3 (CI), 4 (hooks & security), 10 (architecture tests/docs).

2. **Runtime plumbing next**

   * Refactor 5 (Docker simplification), 6 (finalize variants), 8 (move experiments).

3. **Config & UX**

   * Refactor 7 (config cleanup) and 6/8 where they touch defaults.
   * Refactor 6 ties into entrypoints; do 6 and 7 close together.

Each of these is also a **self-contained branch/PR**, so you can stop after any one and still have a strictly better, simpler project.
