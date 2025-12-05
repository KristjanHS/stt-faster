# Architecture Diagrams - Current vs. Target

## Current Architecture (Problematic)

### Dependency Flow - Current State

```
┌─────────────────────────────────────────────────────────────┐
│                    User / CLI                               │
│              (transcribe_manager.py)                        │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│           scripts/transcription/                           │
│  ┌──────────────────────────────────────────────────┐      │
│  │ processor.py (213 lines)                         │      │
│  │  - Business logic ✗                              │      │
│  │  - File operations ✗                             │      │
│  │  - Database calls ✗                              │      │
│  │                                                   │      │
│  │  imports backend.transcribe ───────────┐         │      │
│  └──────────────────────────────────────────┼───────┘      │
│                                             │               │
│  ┌──────────────────────────────────────────┼───────┐      │
│  │ database.py (172 lines)                  │       │      │
│  │  - SQLite operations at ROOT ✗           │       │      │
│  │  - No migrations ✗                       │       │      │
│  └──────────────────────────────────────────┘       │      │
│                                                      │      │
│  ┌──────────────────────────────────────────────────┘      │
│  │ exceptions.py                                           │
│  │  - Duplicate hierarchy ✗                                │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  backend/                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │ transcribe.py (203 lines)                          │     │
│  │  ┌─────────────────────────────────────────────┐   │     │
│  │  │ pick_model() - GOD FUNCTION ✗               │   │     │
│  │  │  - Model selection                          │   │     │
│  │  │  - Model downloading (network I/O)          │   │     │
│  │  │  - Model loading                            │   │     │
│  │  │  - GPU/CPU fallback logic                   │   │     │
│  │  │  - Error handling                           │   │     │
│  │  └─────────────────────────────────────────────┘   │     │
│  │                                                     │     │
│  │  transcribe() - direct WhisperModel calls          │     │
│  │    No abstraction ✗                                │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ config.py                                          │     │
│  │  - UNUSED RAG/LLM config ✗                         │     │
│  │  - Side effects on import ✗                        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ exceptions.py                                      │     │
│  │  - Another exception hierarchy ✗                   │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              External Dependencies                           │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ WhisperModel   │  │ SQLite         │  │ Filesystem    │  │
│  │ (faster-whisper│  │ (at root dir!) │  │ (direct calls)│  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────┐      │
│  │ UNUSED: LangChain, Ollama, sentence-transformers │      │
│  │         torch (for embeddings) ✗                  │      │
│  └───────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ frontend/  (EMPTY!) ✗                  │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ transcribe_state.db (at root!) ✗       │
└────────────────────────────────────────┘
```

### Problems Highlighted:

1. **❌ Scripts contain business logic** - should be thin CLI wrappers
2. **❌ Circular dependency risk** - scripts import backend
3. **❌ No test coverage** - processor/database not in coverage config
4. **❌ God function** - pick_model() does everything
5. **❌ No abstractions** - direct coupling to WhisperModel
6. **❌ Duplicate exceptions** - two separate hierarchies
7. **❌ State at root** - database pollutes repo
8. **❌ Unused dependencies** - 500MB+ wasted
9. **❌ Empty frontend** - misleading structure

---

## Target Architecture (Clean/Hexagonal)

### Dependency Flow - Target State

```
                    ┌─────────────────────────────┐
                    │         User / CLI          │
                    │   (transcribe_manager.py)   │
                    │     - Parse args            │
                    │     - Load config           │
                    │     - Wire dependencies     │
                    │     - Call use case         │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                           │
│              (Use Cases - Orchestration)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ProcessAudioBatchUseCase                                 │  │
│  │                                                            │  │
│  │  __init__(                                                │  │
│  │      transcription: TranscriptionService,  ◄─── Interface │  │
│  │      storage: FileRepository,              ◄─── Interface │  │
│  │      state: StateRepository                ◄─── Interface │  │
│  │  )                                                        │  │
│  │                                                            │  │
│  │  execute(folder: Path, preset: str) -> BatchResult:       │  │
│  │      1. Get files from storage                            │  │
│  │      2. For each file:                                    │  │
│  │         - Call transcription service                      │  │
│  │         - Save result to storage                          │  │
│  │         - Update state                                    │  │
│  │      3. Return summary                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ModelPresets (Config as Code)                            │  │
│  │    PRESETS = {                                            │  │
│  │      "et-large": ModelConfig(...),                        │  │
│  │      "turbo": ModelConfig(...),                           │  │
│  │    }                                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────┬──────────────────────────┬─────────────────────┘
                 │                          │
     ┌───────────▼─────────┐    ┌───────────▼──────────────────┐
     │                     │    │                              │
     │   DOMAIN LAYER      │    │   INFRASTRUCTURE LAYER       │
     │  (Business Logic)   │    │  (External Dependencies)     │
     │                     │    │                              │
     │  ┌────────────────┐ │    │  ┌────────────────────────┐ │
     │  │ Models         │ │    │  │ Implementations        │ │
     │  │ ────────────── │ │    │  │ ──────────────────────│ │
     │  │ @dataclass     │ │    │  │ WhisperTranscription   │ │
     │  │ Transcription  │ │    │  │   Service              │ │
     │  │   Result       │ │    │  │   implements           │ │
     │  │                │ │    │  │   TranscriptionService │ │
     │  │ @dataclass     │ │    │  │                        │ │
     │  │ ModelConfig    │ │    │  │ FileSystemFile         │ │
     │  │                │ │    │  │   Repository           │ │
     │  │ @dataclass     │ │    │  │   implements           │ │
     │  │ Segment        │ │    │  │   FileRepository       │ │
     │  └────────────────┘ │    │  │                        │ │
     │                     │    │  │ SQLiteState            │ │
     │  ┌────────────────┐ │    │  │   Repository           │ │
     │  │ Exceptions     │ │    │  │   implements           │ │
     │  │ ────────────── │ │    │  │   StateRepository      │ │
     │  │ Transcription  │ │    │  │                        │ │
     │  │   Error        │ │    │  └────────────────────────┘ │
     │  │   ├─ ModelErr  │ │    │                              │
     │  │   ├─ Process   │ │    │  ┌────────────────────────┐ │
     │  │   └─ Infra     │ │    │  │ Model Loading          │ │
     │  └────────────────┘ │    │  │ ──────────────────────│ │
     │                     │    │  │ HuggingFaceModel       │ │
     │  ┌────────────────┐ │    │  │   Downloader           │ │
     │  │ Interfaces     │ │    │  │   implements           │ │
     │  │ ────────────── │ │    │  │   ModelDownloader      │ │
     │  │ @abstractmethod│ │    │  │                        │ │
     │  │                │ │    │  │ ModelLoader            │ │
     │  │ Transcription  │ │    │  │   - GPU detection      │ │
     │  │   Service      │ │    │  │   - Fallback logic     │ │
     │  │                │ │    │  │   - Load with config   │ │
     │  │ FileRepository │ │    │  └────────────────────────┘ │
     │  │                │ │    │                              │
     │  │ StateRepos     │ │    │  ┌────────────────────────┐ │
     │  │                │ │    │  │ Test Doubles           │ │
     │  │ ModelLoader    │ │    │  │ ──────────────────────│ │
     │  └────────────────┘ │    │  │ InMemoryFile           │ │
     │                     │    │  │   Repository           │ │
     │  NO EXTERNAL DEPS!  │    │  │ InMemoryState          │ │
     │  ✅ Pure Python     │    │  │   Repository           │ │
     │  ✅ Fast tests      │    │  │ FakeTranscription      │ │
     │  ✅ No I/O          │    │  │   Service              │ │
     │                     │    │  └────────────────────────┘ │
     └─────────────────────┘    └──────────────┬───────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────┐
                         │    External Systems                  │
                         │  ┌─────────────────────────────────┐ │
                         │  │ WhisperModel (faster-whisper)   │ │
                         │  │ SQLite (~/.local/share/stt-*)   │ │
                         │  │ Filesystem                       │ │
                         │  │ HuggingFace Hub                  │ │
                         │  └─────────────────────────────────┘ │
                         └──────────────────────────────────────┘

                         ┌──────────────────────────────────────┐
                         │    Test Layer                        │
                         │  ┌─────────────────────────────────┐ │
                         │  │ tests/unit/domain/              │ │
                         │  │   - Fast, no I/O                │ │
                         │  │ tests/unit/application/         │ │
                         │  │   - Mocked services             │ │
                         │  │ tests/unit/infrastructure/      │ │
                         │  │   - In-memory adapters          │ │
                         │  │ tests/integration/              │ │
                         │  │   - Real implementations        │ │
                         │  │ tests/e2e/                      │ │
                         │  │   - Full workflow               │ │
                         │  └─────────────────────────────────┘ │
                         └──────────────────────────────────────┘
```

### Dependency Rule (The Dependency Inversion Principle)

```
       Outer Layers                          Inner Layers
    (Details/Infra)                         (Business Logic)

         ┌────┐                                  ┌────┐
         │ UI │──────────────────────────────────│ UC │
         └────┘                                  └────┘
            │                                       │
            │  Depends on ─────────────────────►   │
            │     (imports, uses)                  │
            │                                       │
            │                                       │
         ┌──────┐                              ┌────────┐
         │ Infra│──────────────────────────────│ Domain │
         └──────┘                              └────────┘
     
     Key: Arrows point INWARD only
          Domain has ZERO outward dependencies
```

### Benefits of Target Architecture:

1. **✅ Clear separation of concerns**
   - Domain = pure business logic (no external deps)
   - Application = orchestration of use cases
   - Infrastructure = adapters to external systems

2. **✅ Testability**
   - Unit tests: Fast, use in-memory doubles
   - Integration tests: Real implementations
   - Can test domain without any I/O

3. **✅ Flexibility**
   - Swap WhisperModel for different service
   - Change from SQLite to Postgres
   - Add REST API without touching domain

4. **✅ Maintainability**
   - Changes localized to single layer
   - Dependencies flow inward only
   - Easy to understand and reason about

---

## Module Organization Comparison

### Current Structure
```
stt-faster/
├── backend/
│   ├── config.py              (Mixed: config + logging side effects)
│   ├── transcribe.py          (Mixed: domain + infra + app)
│   ├── exceptions.py          (Domain - OK placement)
│   └── main.py                (Unused Docker keepalive)
├── scripts/
│   ├── transcribe_manager.py (CLI - OK)
│   └── transcription/         ← WRONG! Business logic in scripts
│       ├── processor.py
│       ├── database.py
│       └── exceptions.py
├── frontend/                  ← EMPTY
└── transcribe_state.db        ← WRONG! State at repo root
```

### Target Structure
```
stt-faster/
├── backend/
│   ├── domain/                      # Pure business logic
│   │   ├── models.py               # TranscriptionResult, ModelConfig
│   │   ├── exceptions.py           # Unified hierarchy
│   │   └── services.py             # Abstract interfaces
│   ├── application/                 # Use cases
│   │   ├── process_audio_batch.py  # Main use case
│   │   ├── get_status.py           # Status queries
│   │   └── model_presets.py        # Preset configurations
│   ├── infrastructure/              # External dependencies
│   │   ├── whisper/
│   │   │   ├── model_loader.py     # GPU detection, loading
│   │   │   ├── model_downloader.py # HF download logic
│   │   │   └── transcription_service.py  # WhisperModel adapter
│   │   ├── persistence/
│   │   │   ├── file_repository.py  # File operations
│   │   │   └── state_database.py   # SQLite adapter
│   │   └── device/
│   │       └── gpu_detector.py     # CUDA/cuDNN detection
│   └── config.py                    # Centralized config
├── scripts/
│   ├── transcribe_manager.py       # CLI only (thin wrapper)
│   └── check_gpu.py                # Diagnostic utility
├── tests/
│   ├── unit/
│   │   ├── domain/                 # Fast, no I/O
│   │   ├── application/            # Mocked services
│   │   └── infrastructure/         # In-memory adapters
│   ├── integration/                # Real implementations
│   └── e2e/                        # Full workflow
└── docs/
    ├── ARCHITECTURE.md             # This document
    └── ADR/                        # Architecture Decision Records
```

---

## Test Architecture Comparison

### Current Test Strategy (Problematic)

```
pytest coverage:
  --cov=backend        ← Only covers backend/
                         scripts/transcription/ NOT covered! ✗

tests/
  ├── unit/
  │   ├── test_config.py         (backend/config.py)
  │   └── test_transcribe.py     (backend/transcribe.py)
  │                               ↑ Can't mock WhisperModel easily
  ├── integration/
  │   └── test_full_flow.py      (Uses real filesystem, real DB at root)
  └── e2e/

Problems:
  - Business logic in scripts/ not tracked
  - Can't easily mock external dependencies
  - Tests use real filesystem (slow, brittle)
  - No in-memory test doubles
```

### Target Test Strategy (Clean)

```
pytest coverage:
  --cov=backend        ← Covers ALL backend code
  --cov=scripts.transcription  ← Wait, this is deleted! ✅

tests/
  ├── unit/
  │   ├── domain/
  │   │   ├── test_models.py           (Pure dataclass tests)
  │   │   └── test_exceptions.py       (Exception hierarchy)
  │   ├── application/
  │   │   ├── test_process_batch.py    (Uses FakeTranscriptionService)
  │   │   └── test_get_status.py       (Uses InMemoryStateRepository)
  │   └── infrastructure/
  │       ├── test_whisper_service.py  (Unit tests with mock WhisperModel)
  │       ├── test_file_repository.py  (Tests with tmp_path fixture)
  │       └── test_state_database.py   (SQLite :memory: database)
  │
  ├── integration/
  │   ├── test_whisper_integration.py  (Real WhisperModel, small audio)
  │   ├── test_file_operations.py      (Real filesystem with cleanup)
  │   └── test_database_migrations.py  (Real SQLite with test DB)
  │
  └── e2e/
      └── test_full_workflow.py        (End-to-end with fixtures)

Benefits:
  - Unit tests: <5 seconds (no I/O)
  - Can test domain in isolation
  - Easy to mock/stub services
  - Clear test boundaries
  - No test pollution (in-memory/tmp)
```

---

## Configuration Management

### Current (Scattered)

```
┌─────────────────────────────────────────────────────────┐
│ Config Location 1: backend/config.py                   │
│   DEFAULT_EMBEDDING_MODEL = "..." ✗ UNUSED             │
│   OLLAMA_MODEL = "..." ✗ UNUSED                         │
│   _setup_logging() ✗ SIDE EFFECT ON IMPORT             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Config Location 2: backend/transcribe.py               │
│   def pick_model(preset: str = "et-large"):            │
│     if preset == "et-large": ... ✗ HARDCODED           │
│     if preset == "turbo": ... ✗ HARDCODED              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Config Location 3: scripts/transcription/processor.py  │
│   SUPPORTED_AUDIO_EXTENSIONS = {...} ✗ SCATTERED        │
│   PROCESSED_FOLDER_NAME = "processed" ✗ SCATTERED       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Config Location 4: Environment variables (undocumented)│
│   LOG_LEVEL, HF_HOME, APP_LOG_DIR ✗ NO .env.example    │
└─────────────────────────────────────────────────────────┘
```

### Target (Centralized)

```
┌─────────────────────────────────────────────────────────────────┐
│ backend/config.py (Single Source of Truth)                     │
│                                                                 │
│  @dataclass                                                     │
│  class AppConfig:                                               │
│      # Transcription                                            │
│      default_preset: str = "et-large"                           │
│      supported_formats: tuple[str, ...] = (".wav", ".mp3", ...) │
│      output_folder_name: str = "processed"                      │
│                                                                 │
│      # Storage                                                  │
│      cache_dir: Path = ...                                      │
│      state_db_path: Path | None = None  # XDG default if None   │
│                                                                 │
│      # Logging                                                  │
│      log_level: str = "INFO"                                    │
│      log_dir: Path | None = None                                │
│                                                                 │
│      @classmethod                                               │
│      def from_env(cls) -> "AppConfig":                          │
│          """Load from environment variables."""                 │
│          return cls(                                            │
│              default_preset=os.getenv("STT_DEFAULT_PRESET", ..),│
│              log_level=os.getenv("LOG_LEVEL", "INFO"),          │
│              ...                                                │
│          )                                                      │
│                                                                 │
│      def validate(self) -> None:                                │
│          """Validate configuration values."""                   │
│          if self.log_level not in ["DEBUG", "INFO", ...]:       │
│              raise ValueError(f"Invalid log level: ...")        │
│                                                                 │
│  # Logging setup - EXPLICIT, not on import                      │
│  def setup_logging(config: AppConfig) -> None:                  │
│      """Initialize logging (call explicitly, no side effects)."""│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ backend/application/model_presets.py                           │
│                                                                 │
│  PRESETS: dict[str, ModelConfig] = {                            │
│      "et-large": ModelConfig(                                   │
│          model_id="TalTechNLP/whisper-large-v3-turbo-et-...",   │
│          device="cuda",                                         │
│          compute_type="int8_float16",                           │
│          requires_ct2_subfolder=True,                           │
│      ),                                                         │
│      "turbo": ModelConfig(...),                                 │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ .env.example (Documentation)                                   │
│                                                                 │
│  # Transcription settings                                       │
│  STT_DEFAULT_PRESET=et-large                                    │
│                                                                 │
│  # Logging                                                      │
│  LOG_LEVEL=INFO                                                 │
│  LOG_DIR=/path/to/logs  # Optional                              │
│                                                                 │
│  # Model cache                                                  │
│  HF_HOME=/path/to/cache                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Key Architectural Shifts

| Aspect | Current | Target |
|--------|---------|--------|
| **Layer separation** | Mixed concerns | Clean domain/app/infra |
| **Dependency direction** | Bidirectional risk | Inward only |
| **Business logic location** | scripts/ and backend/ | backend/ only |
| **Test doubles** | None | In-memory repos, fakes |
| **Configuration** | 4 locations | 1 central config |
| **Exception hierarchy** | 2 separate | 1 unified |
| **Service boundaries** | Direct coupling | Abstract interfaces |
| **State storage** | Repo root | User directory (XDG) |
| **Test coverage** | ~60%, gaps | >80%, complete |
| **Unused code** | 500MB+ deps | Minimal |

---

For detailed implementation steps, see:
- `.cursor/plans/architecture_refactoring_plan.md` - Full analysis
- `.cursor/plans/architecture_quick_reference.md` - Quick wins and checklists

