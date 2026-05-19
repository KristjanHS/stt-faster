# Architecture Critique & Refactoring Plan

**Project:** stt-faster  
**Date:** 2025-12-05  
**Status:** PROPOSAL

---

## Executive Summary

This document identifies critical architectural shortcomings in the `stt-faster` project and proposes a comprehensive refactoring plan. The project suffers from unclear module boundaries, duplicate code, unused dependencies, and mixed concerns that violate separation of concerns principles.

**Key Issues:**
1. **Identity Crisis**: Project unclear if it's a transcription tool or RAG/LLM system
2. **Ghost Architecture**: Empty frontend, unused Ollama/LangChain dependencies
3. **Module Boundary Violations**: Scripts importing from backend, unclear ownership
4. **Duplicate Exception Hierarchies**: Two separate exception modules for same domain
5. **Missing Abstraction Layers**: Direct coupling to filesystem, models, and infrastructure

---

## Critical Architecture Shortcomings

### 1. **Identity Crisis: What Is This Project?**

**Problem:**
- **Name**: "stt-faster" (speech-to-text)
- **Dependencies**: LangChain, Ollama, sentence-transformers (RAG/LLM stack)
- **Reality**: Audio transcription batch processor
- **Config**: Mentions embedding models, rerankers, Ollama models (unused)
- **Description in pyproject.toml**: "template for LLM experimentation"

```python
# From backend/config.py - COMPLETELY UNUSED
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL = "cas/mistral-7b-instruct-v0.3"
```

**Impact:**
- **Bloated dependencies**: 500MB+ of unused ML libraries
- **Confused contributors**: What features are in scope?
- **Maintenance burden**: Keeping track of RAG-related deps that serve no purpose
- **Security surface**: Unnecessary attack vectors from unused dependencies

**Evidence:**
```toml
dependencies = [
    # Actually used for transcription
    "ctranslate2>=4.6.0,<5",
    "faster-whisper>=1.0.0,<2.0.0",
    
    # RAG/LLM stack - COMPLETELY UNUSED
    "langchain>=0.3.27,<0.4.0",
    "langchain-community>=0.3.27,<0.4.0", 
    "sentence-transformers>=5.1.0,<6.0.0",
    "torch>=2.8.0,<3.0.0",  # Only for sentence-transformers, not Whisper
]
```

**Recommendation:**
→ **Decision point: Choose one path**
  - **Path A**: Pure transcription tool → Remove LangChain, Ollama, RAG deps
  - **Path B**: RAG/LLM system with transcription → Build the missing features

---

### 2. **Ghost Architecture: The Empty Frontend**

**Problem:**
```
frontend/
... no children found ...
```

Yet:
- `pyproject.toml` declares `frontend` as first-party code
- Test coverage configured for non-existent frontend
- Folder structure implies client-server architecture
- Documentation mentions "frontend" in AGENTS.md

**Impact:**
- **Misleading structure**: Suggests web UI that doesn't exist
- **False expectations**: New developers look for API endpoints
- **Test configuration waste**: Coverage tools scan empty directories

**Recommendation:**
→ **Remove the illusion**
  - Delete `frontend/` folder or document it as "future work"
  - Update pyproject.toml `known_first_party` to remove frontend
  - Clarify this is a CLI-only tool

---

### 3. **Module Boundary Chaos**

**Problem: Scripts are not scripts, they're application code**

Current structure violates the principle that `scripts/` should be thin wrappers:

```
scripts/
  ├── transcribe_manager.py  # 186 lines - CLI layer ✓
  └── transcription/         # Business logic ✗
      ├── processor.py       # 213 lines - Core domain logic
      ├── database.py        # 172 lines - Infrastructure
      └── exceptions.py      # Domain exceptions
```

**The violation:**
```python
# scripts/transcription/processor.py imports from backend
from backend.transcribe import transcribe_to_json
```

**Why this is bad:**
1. **Circular dependencies waiting to happen**: Scripts import backend, backend could import scripts
2. **Testing nightmare**: Where do you test `TranscriptionProcessor`? Unit or integration?
3. **Packaging confusion**: `scripts/` typically not included in wheel distributions
4. **Unclear ownership**: Is `processor.py` part of the application or a utility?

**Current coverage config:**
```toml
[tool.pytest.ini_options]
--cov=backend  # processor.py, database.py NOT COVERED despite being business logic!
```

**Recommendation:**
→ **Reorganize by architectural layer**
```
backend/
  ├── domain/              # Pure business logic
  │   ├── models.py
  │   ├── exceptions.py
  │   └── transcription.py # Core transcription logic
  ├── infrastructure/      # External dependencies
  │   ├── whisper_service.py
  │   ├── file_repository.py
  │   └── state_database.py
  ├── application/         # Use cases
  │   └── process_audio_batch.py
  └── config.py

scripts/
  ├── transcribe_manager.py  # CLI only - calls application layer
  └── check_gpu.py           # Diagnostic utilities only
```

---

### 4. **Duplicate Exception Hierarchies**

**Problem: Two exception modules for the same domain**

```python
# backend/exceptions.py
class TranscriptionError(Exception): ...
class ModelLoadError(TranscriptionError): ...
class ModelNotFoundError(TranscriptionError, FileNotFoundError): ...

# scripts/transcription/exceptions.py
class TranscriptionAutomationError(Exception): ...
class FileProcessingError(TranscriptionAutomationError): ...
class FileMoveError(FileProcessingError, OSError): ...
```

**Why this exists:**
- Developer created exceptions close to usage point (good instinct)
- But forgot exceptions are part of the **domain model**, not infrastructure
- Result: Two root exception types for same business domain

**Impact:**
- **Cannot catch all transcription errors**: Need two except blocks
- **Unclear hierarchy**: Which exception should `DatabaseError` inherit from?
- **Maintenance**: Changes to error handling require updates in two places

**Recommendation:**
→ **Unified exception hierarchy in domain layer**

```python
# backend/domain/exceptions.py
class TranscriptionError(Exception):
    """Root exception for all transcription domain errors."""

class ModelError(TranscriptionError):
    """Model-related errors."""
    
class ModelLoadError(ModelError): ...
class ModelNotFoundError(ModelError, FileNotFoundError): ...

class ProcessingError(TranscriptionError):
    """Audio processing errors."""
    
class FileProcessingError(ProcessingError): ...
class FileMoveError(ProcessingError, OSError): ...

class InfrastructureError(TranscriptionError):
    """Infrastructure failures."""
    
class DatabaseError(InfrastructureError, RuntimeError): ...
class StorageError(InfrastructureError, OSError): ...
```

---

### 5. **The God Function: `pick_model()`**

**Problem:**
```python
# backend/transcribe.py
def pick_model(preset: str = "et-large") -> WhisperModel:
    def _load(model_name: str, *, default_device: str, default_type: str) -> WhisperModel:
        try:
            return WhisperModel(model_name, device=default_device, compute_type=default_type)
        except Exception as error:  # fall back to CPU when GPU libs are missing
            LOGGER.error(...)
            try:
                return WhisperModel(model_name, device="cpu", compute_type="int8")
            except Exception as cpu_error:
                raise ModelLoadError(...)
    
    # ... 50 more lines of nested logic
```

**Issues:**
1. **Mixed concerns**: Model selection + downloading + loading + fallback logic
2. **Hard to test**: How do you test GPU fallback without a GPU?
3. **Hidden dependencies**: Calls `snapshot_download` (network I/O) without abstractions
4. **Configuration in code**: Model presets hardcoded as string switches
5. **Poor error recovery**: Broad `except Exception` masks real problems

**Recommendation:**
→ **Separate concerns with Strategy pattern**

```python
# backend/domain/model_config.py
@dataclass
class ModelConfig:
    model_id: str
    device: str
    compute_type: str
    requires_ct2_subfolder: bool = False

# backend/infrastructure/model_loader.py
class ModelLoader:
    def __init__(self, downloader: ModelDownloader, device_detector: DeviceDetector):
        self.downloader = downloader
        self.device_detector = device_detector
    
    def load(self, config: ModelConfig) -> WhisperModel:
        """Load model with automatic fallback."""
        path = self.downloader.get_or_download(config)
        return self._load_with_fallback(path, config)

# backend/application/model_presets.py
PRESETS = {
    "et-large": ModelConfig("TalTechNLP/whisper-large-v3-turbo-et-verbatim", ...),
    "turbo": ModelConfig("Systran/faster-whisper-large-v3-turbo", ...),
}
```

---

### 6. **State Management Anti-Pattern**

**Problem: SQLite database at repository root**

```bash
$ ls -la transcribe_state.db
-rw-r--r-- 1 user user 12288 Dec  5 10:30 transcribe_state.db
```

**Issues:**
1. **Pollutes repository**: State file committed to git (see .gitignore gap)
2. **Multi-user conflicts**: Two users can't process same folder
3. **No migration strategy**: Schema changes break existing databases
4. **Implicit coupling**: File location hardcoded as default parameter
5. **Testing pollution**: Tests create real database files

**Current code:**
```python
class TranscriptionDatabase:
    def __init__(self, db_path: str = "transcribe_state.db"):  # ← Root directory!
        self.conn = sqlite3.connect(db_path)
```

**Recommendation:**
→ **XDG Base Directory compliance**

```python
# backend/infrastructure/state_database.py
from pathlib import Path
import os

def get_default_db_path() -> Path:
    """Get user-specific state directory following XDG spec."""
    if os.name == 'nt':  # Windows
        base = Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    else:  # Linux/macOS
        base = Path(os.getenv('XDG_DATA_HOME', Path.home() / '.local/share'))
    
    db_dir = base / 'stt-faster'
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / 'transcribe_state.db'

class TranscriptionDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_default_db_path()
```

Also need:
- Alembic or simple migration system
- Schema versioning
- Separate test database fixture

---

### 7. **No Service Boundaries**

**Problem: Direct infrastructure coupling everywhere**

```python
# scripts/transcription/processor.py
from backend.transcribe import transcribe_to_json  # Direct function call

def process_file(self, file_path: str) -> bool:
    transcribe_to_json(str(file_path_obj), str(json_path), self.preset)
    self._move_to_processed(file_path_obj, json_path)
    self.db.update_status(file_path, "completed")
```

**What's wrong:**
- **Cannot swap implementations**: Stuck with faster-whisper forever
- **Cannot mock for testing**: Must use real WhisperModel in tests
- **Violates Dependency Inversion**: High-level code depends on low-level details

**Recommendation:**
→ **Define service interfaces**

```python
# backend/domain/services.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    confidence: float

class TranscriptionService(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file to text."""

# backend/infrastructure/whisper_service.py
class WhisperTranscriptionService(TranscriptionService):
    def __init__(self, model: WhisperModel):
        self.model = model
    
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        segments, info = self.model.transcribe(str(audio_path), ...)
        return self._convert_to_result(segments, info)

# backend/application/process_audio_batch.py
class ProcessAudioBatchUseCase:
    def __init__(
        self,
        transcription: TranscriptionService,
        storage: FileRepository,
        state: StateRepository,
    ):
        self.transcription = transcription
        self.storage = storage
        self.state = state
    
    def execute(self, folder: Path, preset: str) -> BatchResult:
        """Process all audio files in folder."""
```

---

### 8. **Configuration Confusion**

**Problem: Scattered configuration**

Config exists in 4 places:
1. `backend/config.py` - Unused LLM/RAG models
2. `pyproject.toml` - Project metadata
3. Environment variables - Some mentioned, some not
4. Function defaults - Model presets hardcoded

**Issues:**
- No single source of truth
- Can't easily change model presets
- No environment-specific configs (dev/prod)
- Logging config side-effects on import

```python
# backend/config.py - Executes on import!
_setup_logging()  # ← Side effect
```

**Recommendation:**
→ **Centralized config with validation**

```python
# backend/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

@dataclass
class AppConfig:
    # Transcription
    default_model_preset: str = "et-large"
    cache_dir: Path = Path(os.getenv("HF_HOME", "~/.cache/hf")).expanduser()
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path | None = None
    
    # Processing
    supported_formats: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac")
    batch_size: int = 1
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            default_model_preset=os.getenv("STT_DEFAULT_PRESET", "et-large"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            ...
        )

def setup_logging(config: AppConfig) -> None:
    """Initialize logging based on config (call explicitly)."""
```

---

### 9. **Testing Architecture Gaps**

**Problems:**

1. **Coverage blind spots:**
```toml
[tool.pytest.ini_options]
--cov=backend  # ← scripts/ business logic NOT covered!
```

2. **No test isolation:**
- Tests use real filesystem
- Tests create databases at root level
- GPU tests can't run in CI

3. **Missing test categories:**
- No contract tests for service interfaces
- No property-based tests for transcription
- No performance benchmarks

**Recommendation:**
→ **Test pyramid with proper isolation**

```
tests/
  ├── unit/
  │   ├── domain/          # Pure logic, no I/O
  │   ├── application/     # Use cases with mocked services
  │   └── infrastructure/  # Repository patterns with in-memory adapters
  ├── integration/
  │   ├── test_whisper_service.py     # Real model, small audio
  │   ├── test_file_repository.py     # tmp_path fixtures
  │   └── test_state_database.py      # SQLite in-memory
  ├── e2e/
  │   └── test_full_workflow.py       # End-to-end with test fixtures
  └── fixtures/
      ├── audio/           # Sample audio files
      └── factories.py     # Test data builders
```

Update coverage:
```toml
--cov=backend
--cov=scripts.transcription  # Include business logic in scripts
```

---

### 10. **Docker/Deployment Confusion**

**Problem:**
- Empty `docker/` directory
- README mentions Docker Compose
- No Dockerfile
- Backend main.py is a "keepalive" for Docker that never starts

```python
# backend/main.py
def main() -> NoReturn:
    """Block indefinitely until SIGINT/SIGTERM is received."""
```

**Issues:**
- **What is this for?**: No web server, no API, just sleeps
- **Ghost infrastructure**: Docker mentioned but not present
- **Incomplete features**: Suggests microservices architecture that doesn't exist

**Recommendation:**
→ **Remove Docker or implement it properly**

If keeping Docker:
```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY backend/ ./backend/
CMD ["uv", "run", "uvicorn", "backend.api:app", "--host", "0.0.0.0"]
```

If removing Docker:
- Delete `docker/` directory
- Remove Docker references from README
- Delete backend/main.py keepalive

---

## Proposed Target Architecture

### Layered Architecture (Hexagonal/Clean Architecture)

```
stt-faster/
├── backend/
│   ├── domain/                    # Business logic (no external dependencies)
│   │   ├── __init__.py
│   │   ├── models.py             # TranscriptionResult, AudioFile, etc.
│   │   ├── exceptions.py         # Domain exception hierarchy
│   │   └── services.py           # Abstract service interfaces
│   │
│   ├── application/               # Use cases (orchestration)
│   │   ├── __init__.py
│   │   ├── process_audio_batch.py
│   │   ├── get_processing_status.py
│   │   └── model_presets.py
│   │
│   ├── infrastructure/            # External dependencies
│   │   ├── __init__.py
│   │   ├── whisper/
│   │   │   ├── model_loader.py
│   │   │   ├── model_downloader.py
│   │   │   └── transcription_service.py
│   │   ├── persistence/
│   │   │   ├── state_database.py
│   │   │   └── file_repository.py
│   │   └── device/
│   │       └── gpu_detector.py
│   │
│   └── config.py                  # Configuration management
│
├── scripts/
│   ├── transcribe_manager.py     # CLI interface (thin wrapper)
│   └── check_gpu.py              # Diagnostic utilities
│
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── e2e/
│
└── pyproject.toml
```

### Dependency Flow (Hexagonal Architecture)

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Scripts                    │
│              (transcribe_manager.py)                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Application Layer                      │
│         (Use Cases / Orchestration)                 │
│    ┌─────────────────────────────────────┐          │
│    │  ProcessAudioBatchUseCase           │          │
│    │  - Coordinates domain + infra       │          │
│    └─────────────────────────────────────┘          │
└────────────┬────────────────────┬───────────────────┘
             │                    │
             ▼                    ▼
┌──────────────────────┐  ┌──────────────────────────┐
│   Domain Layer       │  │  Infrastructure Layer    │
│  (Business Logic)    │  │  (External Dependencies) │
│                      │  │                          │
│  - Models            │  │  - WhisperService        │
│  - Exceptions        │  │  - FileRepository        │
│  - Service Interfaces│  │  - StateDatabase         │
│                      │  │  - ModelLoader           │
│  No external deps!   │  │                          │
└──────────────────────┘  └──────────────────────────┘
```

---

## Refactoring Plan

### Phase 1: Clarify Identity & Remove Cruft (1-2 days)

**Goal:** Remove confusion about what this project is.

#### Tasks:

1. **Decide on project scope** (see Critical Issue #1)
   - [ ] **Option A**: Pure transcription → Remove LangChain, Ollama, RAG dependencies
   - [ ] **Option B**: Keep RAG → Implement the features or document roadmap

2. **Clean up unused code**
   - [ ] Remove `frontend/` directory or add README explaining "future work"
   - [ ] Delete unused config variables (EMBEDDING_MODEL, RERANKER_MODEL, OLLAMA_MODEL)
   - [ ] Remove or implement Docker infrastructure
   - [ ] Update pyproject.toml description to match reality

3. **Update documentation**
   - [ ] README: Clarify this is a CLI transcription tool
   - [ ] AGENTS.md: Remove references to non-existent features
   - [ ] Add ARCHITECTURE.md explaining actual structure

**Validation:**
```bash
# Dependencies are minimal
uv tree | wc -l  # Should be <30 if going pure transcription

# No references to unused concepts
rg "langchain|ollama|embedding" backend/ scripts/  # Should be empty

# Documentation is accurate
grep -i "frontend\|API\|web" README.md  # Should not claim web features
```

---

### Phase 2: Reorganize Module Boundaries (2-3 days)

**Goal:** Clear separation between domain, application, and infrastructure.

#### Tasks:

1. **Create new directory structure**
   ```bash
   mkdir -p backend/domain
   mkdir -p backend/application
   mkdir -p backend/infrastructure/whisper
   mkdir -p backend/infrastructure/persistence
   mkdir -p backend/infrastructure/device
   ```

2. **Move code by architectural layer**

   **2.1: Extract domain models**
   - [ ] Create `backend/domain/models.py`:
     ```python
     @dataclass
     class TranscriptionSegment:
         id: int
         start: float
         end: float
         text: str
         speaker: str | None = None
     
     @dataclass  
     class TranscriptionResult:
         audio_filename: str
         language: str
         language_probability: float
         duration: float
         segments: list[TranscriptionSegment]
     
     @dataclass
     class ModelConfig:
         model_id: str
         device: str
         compute_type: str
         requires_ct2_subfolder: bool = False
     ```

   **2.2: Unify exception hierarchy**
   - [ ] Move all exceptions to `backend/domain/exceptions.py`
   - [ ] Delete `scripts/transcription/exceptions.py`
   - [ ] Update all imports across codebase

   **2.3: Define service interfaces**
   - [ ] Create `backend/domain/services.py` with abstract interfaces:
     - `TranscriptionService`
     - `FileRepository`
     - `StateRepository`
     - `ModelLoader`

   **2.4: Move transcription logic**
   - [ ] Refactor `backend/transcribe.py`:
     - Extract model loading → `infrastructure/whisper/model_loader.py`
     - Extract model download → `infrastructure/whisper/model_downloader.py`
     - Create service impl → `infrastructure/whisper/transcription_service.py`
   - [ ] Create presets config → `application/model_presets.py`

   **2.5: Move file operations**
   - [ ] Move `scripts/transcription/processor.py` file ops → `infrastructure/persistence/file_repository.py`
   - [ ] Move `scripts/transcription/database.py` → `infrastructure/persistence/state_database.py`

   **2.6: Create use cases**
   - [ ] Create `application/process_audio_batch.py`:
     ```python
     class ProcessAudioBatchUseCase:
         def __init__(
             self,
             transcription_service: TranscriptionService,
             file_repository: FileRepository,
             state_repository: StateRepository,
         ):
             ...
         
         def execute(self, folder: Path, preset: str) -> BatchResult:
             """Orchestrates the batch processing workflow."""
     ```
   - [ ] Create `application/get_processing_status.py`

   **2.7: Update scripts to use application layer**
   - [ ] Refactor `scripts/transcribe_manager.py` to only:
     - Parse CLI arguments
     - Load configuration
     - Instantiate dependencies
     - Call use case
   - [ ] Delete `scripts/transcription/` (code moved to backend)

3. **Update imports and module paths**
   - [ ] Run automated import fixer:
     ```bash
     find backend scripts tests -name "*.py" -exec sed -i \
       's/from backend.transcribe import/from backend.infrastructure.whisper import/g' {} \;
     ```
   - [ ] Manual review and fixes

4. **Update test structure**
   - [ ] Move tests to match new structure:
     ```
     tests/unit/domain/
     tests/unit/application/
     tests/unit/infrastructure/
     ```
   - [ ] Update imports in all test files

**Validation:**
```bash
# Architecture boundaries respected
rg "from backend.infrastructure" backend/domain/  # Should be empty
rg "from backend.infrastructure" backend/application/  # Should be empty
rg "import backend" scripts/transcription/  # Should not exist (folder deleted)

# Test coverage includes all business logic
pytest --cov=backend --cov-report=term-missing | grep "backend/"
# Should show coverage for infrastructure, application, domain

# All tests pass
make unit && make integration
```

---

### Phase 3: Improve Configuration & State Management (1 day)

**Goal:** Proper configuration and user-friendly state storage.

#### Tasks:

1. **Centralize configuration**
   - [ ] Create `backend/config.py` with dataclass-based config
   - [ ] Implement `Config.from_env()` factory
   - [ ] Remove side effects (no `_setup_logging()` on import)
   - [ ] Move logging setup to explicit initialization

2. **Fix state database location**
   - [ ] Implement XDG Base Directory spec for database location
   - [ ] Add migration mechanism (Alembic or custom)
   - [ ] Add database schema versioning
   - [ ] Update README with new database location

3. **Environment-specific configs**
   - [ ] Create `.env.example` with all configurable options
   - [ ] Document each environment variable
   - [ ] Add config validation with clear error messages

**Validation:**
```bash
# Database not in repo root
ls -la transcribe_state.db  # Should not exist

# Config documented
grep -c "STT_" .env.example  # Should list all options

# No import side effects
python -c "import backend.config" && echo "No logging output during import"
```

---

### Phase 4: Add Service Boundaries & Testing (2-3 days)

**Goal:** Testable, swappable components.

#### Tasks:

1. **Implement service interfaces**
   - [ ] `WhisperTranscriptionService(TranscriptionService)`
   - [ ] `FileSystemFileRepository(FileRepository)`
   - [ ] `SQLiteStateRepository(StateRepository)`
   - [ ] `HuggingFaceModelLoader(ModelLoader)`

2. **Create test doubles**
   - [ ] `InMemoryFileRepository` for tests
   - [ ] `InMemoryStateRepository` for tests
   - [ ] `FakeTranscriptionService` for tests
   - [ ] `StubModelLoader` for tests

3. **Refactor tests**
   - [ ] Unit tests use only in-memory/fake implementations
   - [ ] Integration tests use real implementations with fixtures
   - [ ] Add contract tests for each interface
   - [ ] Update coverage configuration:
     ```toml
     [tool.pytest.ini_options]
     --cov=backend
     --cov-fail-under=80
     ```

4. **Add missing test categories**
   - [ ] Property-based tests for transcription output validation
   - [ ] Performance benchmarks for model loading
   - [ ] Contract tests ensuring all implementations satisfy interfaces

**Validation:**
```bash
# Unit tests run fast
time make unit  # Should be <5 seconds

# No network I/O in unit tests
pytest -v tests/unit --allow-hosts=none  # Should pass

# Coverage meets threshold
pytest --cov=backend --cov-fail-under=80
```

---

### Phase 5: Documentation & Developer Experience (1 day)

**Goal:** Clear documentation for new contributors.

#### Tasks:

1. **Architecture documentation**
   - [ ] Create `docs/ARCHITECTURE.md` explaining layers
   - [ ] Add architecture diagram (ASCII or mermaid)
   - [ ] Document design decisions (ADRs)

2. **Development guide**
   - [ ] Update `docs/DEVELOPMENT.md` with:
     - Setup instructions for new structure
     - How to add new model presets
     - How to implement new services
     - Testing guidelines

3. **Update README**
   - [ ] Accurate project description
   - [ ] Clear feature list
   - [ ] Simple quickstart
   - [ ] Link to architecture docs

4. **Code-level documentation**
   - [ ] Add docstrings to all public interfaces
   - [ ] Add inline comments for complex logic
   - [ ] Update type hints to be complete and accurate

**Validation:**
```bash
# Documentation completeness
rg "class.*Service" backend/domain/services.py | wc -l  # Count interfaces
rg '""".*"""' backend/domain/services.py | wc -l       # Count docstrings
# Should be equal (all interfaces documented)

# Type checking passes
make pyright  # Should have no errors

# README works for new users
docker run --rm -it python:3.12 bash
# Follow README instructions - should work first time
```

---

## Success Metrics

### Before Refactoring
- **LOC in correct layer**: ~30% (most logic in wrong places)
- **Test coverage**: ~60% (scripts/ business logic not covered)
- **Unused dependencies**: 6 (LangChain, Ollama, sentence-transformers, torch for embeddings)
- **Duplicate code**: 2 exception hierarchies, scattered config
- **Architecture violations**: scripts → backend imports (circular risk)
- **Time to onboard new dev**: 2-3 hours (confusing structure)

### After Refactoring
- **LOC in correct layer**: >90%
- **Test coverage**: >80% (all business logic covered)
- **Unused dependencies**: 0
- **Duplicate code**: 0
- **Architecture violations**: 0 (enforced by tests)
- **Time to onboard new dev**: <30 minutes (clear structure)

---

## Risk Mitigation

### Risks

1. **Breaking changes for existing users**
   - CLI interface changes
   - Database location changes
   - Import paths change

   **Mitigation:**
   - Keep CLI interface stable (`transcribe_manager.py` args unchanged)
   - Auto-migrate old database to new location on first run
   - Deprecation warnings for any breaking changes
   - Version bump to 0.2.0 signaling breaking changes

2. **Regression in functionality**
   - Transcription accuracy changes
   - Performance degradation
   - New bugs introduced

   **Mitigation:**
   - Comprehensive test suite before refactoring
   - Golden file tests (record expected output for sample audio)
   - Benchmark tests to catch performance regressions
   - Gradual rollout (Phase 1 → Phase 5, validate after each)

3. **Scope creep**
   - Refactoring takes longer than estimated
   - Adding new features during refactor
   - Over-engineering

   **Mitigation:**
   - Strict phase boundaries
   - No new features - only restructuring existing code
   - Regular check-ins (end of each phase)
   - Rollback plan (git branches per phase)

---

## Rollback Plan

Each phase is a separate git branch. If a phase fails validation:

1. **Review failure**:
   - Run full test suite
   - Check architecture boundaries
   - Validate against success metrics

2. **Decide**:
   - Fix issues and proceed
   - Revert to previous phase
   - Adjust plan

3. **Rollback procedure**:
   ```bash
   git checkout main
   git branch -D phase-N-failed
   # Start phase again with adjusted approach
   ```

---

## Conclusion

The current architecture suffers from unclear identity, mixed concerns, and organizational chaos. The proposed refactoring follows Clean/Hexagonal Architecture principles:

1. **Domain at the center**: Pure business logic, no dependencies
2. **Application layer**: Use cases orchestrating domain + infrastructure
3. **Infrastructure at edges**: All external dependencies isolated
4. **Dependency inversion**: High-level code doesn't depend on low-level details

**Estimated effort:** 7-10 days for full refactoring
**Priority:** High - current structure will cause maintenance problems as project grows

**Next step:** Review this plan with team, decide on Option A vs B for project identity, then begin Phase 1.

---

## Appendix: Quick Wins (Can be done immediately)

These require minimal effort but provide immediate value:

1. **Remove unused dependencies** (15 minutes)
   ```bash
   uv remove langchain langchain-community sentence-transformers
   # If torch only used for embeddings:
   uv remove torch
   ```

2. **Move database out of repo root** (30 minutes)
   ```python
   # Update default in database.py
   def get_default_db_path():
       return Path.home() / '.local/share/stt-faster/transcribe_state.db'
   ```

3. **Unify exception hierarchies** (1 hour)
   - Copy all exceptions to `backend/exceptions.py`
   - Update 3-4 import statements
   - Delete `scripts/transcription/exceptions.py`

4. **Add .gitignore entry** (2 minutes)
   ```
   # .gitignore
   transcribe_state.db
   *.db
   ```

5. **Fix coverage config** (5 minutes)
   ```toml
   [tool.pytest.ini_options]
   --cov=backend
   --cov=scripts.transcription
   ```

**Total time for quick wins: ~2 hours**
**Impact: Immediate reduction in technical debt**

