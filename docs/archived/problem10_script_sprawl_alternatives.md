# Problem 10 Solution: Script Sprawl + sys.path Hacks + Circular-Import Pressure - 3 Implementation Alternatives

**Problem**: Script sprawl, sys.path hacks, and circular-import pressure make the codebase hard to package/distribute, with inconsistent CLI behavior and hidden coupling.

**Current State**:
- Scripts in `scripts/` directory modify `sys.path` to import backend (e.g., `scripts/db/show_run.py`)
- Circular imports handled with local imports inside functions (e.g., `backend/services/whisper_transcription_service.py`)
- No proper console entrypoints - scripts must be run with `.venv/bin/python scripts/...`
- Scripts scattered across multiple locations (`scripts/`, `scripts/db/`, `scripts/variant_checks/`)
- Works only when run from repo root

**Impact**: Hard to package/distribute; inconsistent CLI behavior; "works when run from repo root only"; encourages hidden coupling.

---

## Alternative 1: Minimal Fix - Console Entrypoints + Remove sys.path Hacks (Cleanest)

**Philosophy**: Fix the immediate problems with minimal changes - just enough to make scripts work as proper entrypoints.

### Implementation

1. **Add Console Entrypoints to pyproject.toml**
   ```toml
   [project.scripts]
   stt-faster = "backend.cli.main:main"
   stt-faster-process = "backend.cli.transcription_commands:cmd_process"
   stt-faster-status = "backend.cli.transcription_commands:cmd_status"
   stt-faster-db-show = "backend.cli.db_commands:cmd_show_run"
   stt-faster-db-recent = "backend.cli.db_commands:cmd_show_recent_runs"
   stt-faster-db-check-params = "backend.cli.db_commands:cmd_check_parameters"
   stt-faster-db-check-orphaned = "backend.cli.db_commands:cmd_check_orphaned"
   ```

2. **Move Scripts to backend/cli/ and Remove sys.path Hacks**
   ```python
   # backend/cli/db_commands.py (consolidate scripts/db/*.py)
   """Database inspection commands."""
   import argparse
   from backend.database import TranscriptionDatabase
   
   def cmd_show_run(run_id: int | None = None, timezone: str = "EET") -> int:
       """Show run information by ID or latest if no ID provided."""
       # ... existing logic from scripts/db/show_run.py ...
   
   def cmd_show_recent_runs(limit: int = 10) -> int:
       """Show recent runs."""
       # ... existing logic from scripts/db/show_recent_runs.py ...
   ```

3. **Create Unified CLI Entrypoint**
   ```python
   # backend/cli/main.py
   """Main CLI entrypoint for stt-faster."""
   import sys
   from typing import NoReturn
   import typer
   
   from backend.cli import transcription_commands, db_commands
   
   app = typer.Typer(name="stt-faster", help="Fast audio transcription tool")
   app.add_typer(transcription_commands.app, name="process")
   app.add_typer(db_commands.app, name="db")
   
   def main() -> NoReturn:
       """Main entrypoint."""
       app()
       raise SystemExit(0)
   
   if __name__ == "__main__":
       main()
   ```

4. **Keep Circular Import Workaround (No Change)**
   - Leave `from backend.transcribe import transcribe` inside function in `whisper_transcription_service.py`
   - Document why it's necessary (service depends on transcribe, transcribe depends on service factory)

### Pros
- ✅ Minimal code changes
- ✅ Scripts work as proper entrypoints after `pip install -e .`
- ✅ No sys.path hacks needed
- ✅ Can be installed as a package
- ✅ Backward compatible (old scripts still work during transition)

### Cons
- ⚠️ Circular import still exists (just documented)
- ⚠️ Scripts still scattered (just moved to backend/cli/)
- ⚠️ No unified CLI structure (multiple entrypoints)
- ⚠️ Doesn't solve root cause of circular imports

### Migration from Current Code
- Add `[project.scripts]` to pyproject.toml
- Move `scripts/db/*.py` → `backend/cli/db_commands.py` (consolidate into one module)
- Remove `sys.path.insert()` calls
- Update imports to use `from backend.xxx import yyy`
- Keep `scripts/transcribe_manager.py` as-is for now (or move to `backend/cli/transcription_commands.py`)

---

## Alternative 2: Structured CLI + Fix Circular Imports (Middle Ground)

**Philosophy**: Organize CLI properly and eliminate circular imports by restructuring dependencies.

### Implementation

1. **Unified CLI with Typer**
   ```python
   # backend/cli/main.py
   """Unified CLI entrypoint."""
   import typer
   from backend.cli import db, transcription, utils
   
   app = typer.Typer(name="stt-faster", help="Fast audio transcription tool")
   app.add_typer(transcription.app, name="transcribe")
   app.add_typer(db.app, name="db")
   app.add_typer(utils.app, name="utils")
   
   def main() -> None:
       app()
   ```

2. **Fix Circular Import by Inverting Dependency**
   ```python
   # backend/services/whisper_transcription_service.py
   """Whisper transcription service implementation."""
   from backend.services.interfaces import TranscriptionService, TranscriptionRequest, TranscriptionResult
   from backend.transcribe.core import transcribe_audio  # New core function
   
   class WhisperTranscriptionService:
       def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
           """Transcribe audio - no circular import needed."""
           # Use transcribe_audio directly (moved from transcribe.py)
           payload = transcribe_audio(...)
           return TranscriptionResult(...)
   ```

3. **Refactor transcribe.py to Split Core from Orchestration**
   ```python
   # backend/transcribe/__init__.py
   """Public API - uses services."""
   from backend.services.factory import create_transcription_service
   
   def transcribe(...):
       """High-level transcription function (backward compatible)."""
       service = create_transcription_service(...)
       return service.transcribe(...)
   
   # backend/transcribe/core.py
   """Core transcription logic (no service dependencies)."""
   def transcribe_audio(...):
       """Low-level transcription function."""
       # Pure transcription logic, no service layer
   ```

4. **Organize CLI Commands by Domain**
   ```
   backend/cli/
   ├── __init__.py
   ├── main.py              # Unified entrypoint
   ├── transcription.py     # Process, status commands
   ├── db.py                # Database inspection commands
   └── utils.py             # Utility commands (check-gpu, etc.)
   ```

5. **Console Entrypoints**
   ```toml
   [project.scripts]
   stt-faster = "backend.cli.main:main"
   ```

### Pros
- ✅ Eliminates circular imports
- ✅ Clean dependency structure (services → core, not services ↔ transcribe)
- ✅ Unified CLI entrypoint
- ✅ Better organized (commands grouped by domain)
- ✅ Proper package structure

### Cons
- ⚠️ Requires refactoring `transcribe.py` (split into core + orchestration)
- ⚠️ More code changes than Alternative 1
- ⚠️ Need to maintain backward compatibility for existing `transcribe()` function

### Migration from Current Code
- Split `backend/transcribe.py` into:
  - `backend/transcribe/core.py` (pure transcription logic)
  - `backend/transcribe/__init__.py` (orchestration using services)
- Move `from backend.transcribe import transcribe` to top-level import in `whisper_transcription_service.py`
- Consolidate all scripts into `backend/cli/` organized by domain
- Remove all `sys.path` hacks
- Add single `stt-faster` console entrypoint

---

## Alternative 3: Full Package Restructure + Dependency Injection (Best Practices)

**Philosophy**: Complete architectural cleanup - proper package structure, eliminate all circular dependencies, dependency injection throughout.

### Implementation

1. **Restructure Package Layout**
   ```
   backend/
   ├── __init__.py
   ├── cli/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── commands/
   │   │   ├── transcription.py
   │   │   ├── database.py
   │   │   └── utils.py
   │   └── utils.py          # CLI helpers
   ├── core/                  # Pure business logic (no dependencies on services)
   │   ├── transcription.py  # Core transcription engine
   │   └── preprocessing.py  # Core preprocessing logic
   ├── services/              # Service layer (depends on core, not vice versa)
   │   ├── interfaces.py
   │   ├── transcription.py
   │   └── factory.py
   ├── infrastructure/        # Infrastructure concerns
   │   ├── database.py
   │   ├── file_io.py
   │   └── config.py
   └── orchestration/         # High-level orchestration (depends on services)
       ├── processor.py
       └── transcribe.py      # Public API that uses services
   ```

2. **Eliminate Circular Dependencies with Dependency Injection**
   ```python
   # backend/core/transcription.py
   """Pure transcription core - no service dependencies."""
   def transcribe_audio_core(
       audio_path: str,
       model: Model,
       config: TranscriptionConfig,
   ) -> dict[str, Any]:
       """Core transcription logic."""
       # Pure function, no dependencies on services
   
   # backend/services/transcription.py
   """Transcription service - depends on core, not orchestration."""
   from backend.core.transcription import transcribe_audio_core
   
   class TranscriptionService:
       def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
           # Use core directly
           payload = transcribe_audio_core(...)
           return TranscriptionResult(...)
   
   # backend/orchestration/transcribe.py
   """Public API - uses services via DI."""
   from backend.services.interfaces import TranscriptionService
   
   def transcribe(
       *,
       transcription_service: TranscriptionService,  # Injected
       ...
   ) -> dict[str, Any]:
       """High-level transcription function."""
       result = transcription_service.transcribe(...)
       return result.payload
   ```

3. **Unified CLI with Proper Structure**
   ```python
   # backend/cli/main.py
   """Main CLI entrypoint."""
   import typer
   from backend.cli.commands import transcription, database, utils
   from backend.services.factory import create_services
   
   app = typer.Typer()
   app.add_typer(transcription.app, name="transcribe")
   app.add_typer(database.app, name="db")
   app.add_typer(utils.app, name="utils")
   
   def main() -> None:
       """Main entrypoint with service initialization."""
       app()
   ```

4. **All Scripts Become CLI Commands**
   - `scripts/transcribe_manager.py` → `backend/cli/commands/transcription.py`
   - `scripts/db/*.py` → `backend/cli/commands/database.py`
   - `scripts/check_gpu.py` → `backend/cli/commands/utils.py`
   - All utility scripts consolidated into appropriate command modules

5. **Single Console Entrypoint**
   ```toml
   [project.scripts]
   stt-faster = "backend.cli.main:main"
   ```

6. **Proper Import Structure**
   - No `sys.path` hacks anywhere
   - All imports use absolute paths from `backend.`
   - Clear dependency direction: `cli` → `orchestration` → `services` → `core` → `infrastructure`

### Pros
- ✅ Complete elimination of circular dependencies
- ✅ Clear separation of concerns (core, services, orchestration, infrastructure)
- ✅ Proper dependency direction (no cycles)
- ✅ Easy to test (each layer can be tested independently)
- ✅ Professional package structure
- ✅ Can be easily packaged and distributed
- ✅ Works from any directory after installation

### Cons
- ⚠️ Significant refactoring required
- ⚠️ Need to carefully restructure existing code
- ⚠️ More complex migration path
- ⚠️ Risk of breaking existing functionality during transition
- ⚠️ Requires comprehensive testing after refactoring

### Migration from Current Code
1. **Phase 1: Extract Core Logic**
   - Create `backend/core/` directory
   - Move pure transcription logic from `backend/transcribe.py` to `backend/core/transcription.py`
   - Move pure preprocessing logic to `backend/core/preprocessing.py`
   - Ensure core has no dependencies on services or orchestration

2. **Phase 2: Restructure Services**
   - Update services to depend only on core
   - Remove any orchestration dependencies from services
   - Update `WhisperTranscriptionService` to use `core.transcription` directly

3. **Phase 3: Update Orchestration**
   - Move `backend/transcribe.py` to `backend/orchestration/transcribe.py`
   - Make it depend on services (injected via DI)
   - Update `backend/processor.py` to use services

4. **Phase 4: Consolidate CLI**
   - Move all scripts to `backend/cli/commands/`
   - Remove all `sys.path` hacks
   - Create unified `backend/cli/main.py` entrypoint
   - Add single console script entrypoint

5. **Phase 5: Update Tests**
   - Update all test imports
   - Ensure tests work with new structure
   - Add integration tests for CLI commands

---

## Comparison Matrix

| Aspect | Alternative 1 (Cleanest) | Alternative 2 (Middle) | Alternative 3 (Best Practices) |
|--------|-------------------------|------------------------|-------------------------------|
| **Code Changes** | Minimal | Moderate | Extensive |
| **Circular Imports** | Documented, not fixed | Eliminated | Eliminated |
| **sys.path Hacks** | Removed | Removed | Removed |
| **CLI Structure** | Multiple entrypoints | Unified entrypoint | Unified + structured |
| **Package Structure** | Basic | Improved | Professional |
| **Dependency Direction** | Unchanged | Improved | Clear hierarchy |
| **Migration Risk** | Low | Medium | High |
| **Time to Implement** | 1-2 days | 3-5 days | 1-2 weeks |
| **Maintainability** | Good | Better | Best |
| **Testability** | Good | Better | Best |

---

## Recommendation

**For immediate fix**: Use **Alternative 1** to quickly eliminate sys.path hacks and enable proper entrypoints.

**For long-term health**: Plan migration to **Alternative 3** in phases, starting with Alternative 2 as an intermediate step.

**Suggested Approach**:
1. Implement Alternative 1 first (quick win, low risk)
2. Plan Alternative 2 as next step (fix circular imports)
3. Eventually migrate to Alternative 3 (full restructure) when time permits

This allows the codebase to be immediately usable as a package while planning for architectural improvements.
