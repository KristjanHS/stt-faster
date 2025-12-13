# Architectural audit: Top 10 problems to solve (ranked)

I want to keep the pipeline working after each step!

- [x] **1. No real service boundaries (high-level code depends on low-level details)**

   * **Impact:** Hard to swap/benchmark engines, hard to test cleanly, changes ripple everywhere.
   * **Evidence:** `backend/processor.py` imports and directly calls `backend.transcribe` + variant executor functions.
   * **Fix direction:** Define interfaces like `TranscriptionService`, `PreprocessService`, `StateStore`, `FileMover`; inject them into the processor/orchestrator.
   * **Status:** ✅ IMPLEMENTED - Protocol-based service interfaces with dependency injection. All tests pass.

- [x] **2. `TranscriptionProcessor` is a "god object"**

   * **Impact:** One class owns scanning, processing, output writing, file moving, DB updates, and run aggregation → high bug risk, low reuse.
   * **Evidence:** Processor tests validate move + status update behavior tightly coupled to processor behavior.
   * **Fix direction:** Split into (a) folder scanner, (b) per-file worker, (c) output writer, (d) run summarizer, (e) "move/archive" policy.

- [x] **3. Import-time side effects (logging config runs on import)**

   * **Impact:** Surprising behavior in tests/tools, hard to embed as a library, hard to control logging.
   * **Evidence:** `_setup_logging()` is executed at module import time in `backend/config.py`.
   * **Fix direction:** Make logging setup explicit in entrypoints only (CLI `main()`), never at import time.

- [x] **4. Configuration is scattered and weakly validated**

   * **Impact:** You're mixing env-driven config + default constructors + variant overrides; invalid combos slip through silently.
   * **Evidence:** Variants store arbitrary `config: dict[str, Any]`, plus separate `PreprocessConfig` + `TranscriptionConfig` usage paths.
   * **Fix direction:** Create a single validated "RunConfig" (typed, with defaults) and derive all per-file/per-step settings from it.

- [x] **5. Variant system is "stringly typed" with step-type explosion**

   * **Impact:** Adding/changing steps becomes brittle; step names drift; refactors break variants silently.
   * **Evidence:** `PreprocessStep.step_type` is a large literal union and `config` is untyped.
   * **Fix direction:** Replace `step_type: str` with a registry of step classes/functions (`StepSpec` → `StepRunner`) and typed step configs.

- [ ] **6. Preprocess implementation uses inconsistent execution styles**

   * **Impact:** Different steps likely behave differently (error handling, logging, performance, determinism).
   * **Evidence:** Some steps use `ffmpeg` python bindings + regex parsing (e.g. volumedetect), rather than one unified runner approach.
   * **Fix direction:** Standardize on one execution abstraction (e.g., `run_ffmpeg(cmd)`) and force every step to report metrics the same way.

- [ ] **7. Schema drift risk: migrations are reactive and column-based**

   * **Impact:** "Works on my machine" DB breakages; production DB needs frequent emergency patching.
   * **Evidence:** Tests explicitly simulate missing columns and rely on migrations to patch old DBs.
   * **Fix direction:** Add explicit schema versioning + ordered migrations (even a simple `schema_version` table + migration list is enough).

- [ ] **8. Metrics persistence is fragile (wide tables + huge INSERT surface area)**

   * **Impact:** Every new metric/parameter forces schema changes + updates to long SQL insert lists → easy to miss fields.
   * **Evidence:** `record_run()` inserts a very wide set of columns in one statement.
   * **Fix direction:** Keep a stable core schema + store "extras" as JSON (or separate normalized tables) to stop constant churn.

- [ ] **9. State DB naming/extension inconsistency and doc drift**

   * **Impact:** Confusing ops/debugging; accidental "new DB created" vs "existing DB migrated"; harder cross-platform support.
   * **Evidence:** Refactor log mentions `transcribe_state.db`, but integration uses `transcribe_state.duckdb`; tests also create temp files with `.db` suffix while using DuckDB.
   * **Fix direction:** Pick one canonical filename/extension; add a one-time migration/alias; update docs + CLI help consistently.

- [ ] **10. Script sprawl + sys.path hacks + circular-import pressure**

   * **Impact:** Hard to package/distribute; inconsistent CLI behavior; "works when run from repo root only"; encourages hidden coupling.
   * **Evidence:** Scripts modify `sys.path` to import backend; also processor code uses local imports to dodge dependency cycles.
   * **Fix direction:** Provide proper console entrypoints (e.g., `stt-faster ...`), move scripts into `backend/cli/`, and restructure modules to remove circulars instead of importing inside functions.
