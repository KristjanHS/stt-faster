# Architecture Refactoring - Quick Reference

## 🎯 The Core Problems (Top 10)

| # | Problem | Impact | Effort to Fix |
|---|---------|--------|---------------|
| 1 | **Identity Crisis** | Unused deps (500MB+), confused scope | 2h (remove deps) |
| 2 | **Ghost Frontend** | Misleading structure, false expectations | 15min (delete folder) |
| 3 | **Module Chaos** | Scripts contain business logic, wrong boundaries | 2-3 days (reorganize) |
| 4 | **Duplicate Exceptions** | Two hierarchies for same domain | 1h (unify) |
| 5 | **God Function** | `pick_model()` does everything | 1 day (extract services) |
| 6 | **State at Root** | `transcribe_state.db` pollutes repo | 30min (XDG paths) |
| 7 | **No Service Boundaries** | Direct coupling to infrastructure | 2-3 days (add interfaces) |
| 8 | **Config Scattered** | 4 different places, import side effects | 1 day (centralize) |
| 9 | **Coverage Gaps** | Business logic in scripts/ not covered | 1 day (reorganize tests) |
| 10 | **Docker Ghost** | Empty docker/, keepalive that does nothing | 30min (remove or implement) |

**Total estimated effort: 7-10 days**

---

## 📊 Current vs Target Architecture

### Current (Messy)
```
stt-faster/
├── backend/               ← Transcription logic
│   ├── config.py         ← Unused LLM/RAG config
│   └── transcribe.py     ← God function (300+ lines)
├── scripts/
│   └── transcription/    ← Business logic in scripts! ✗
│       ├── processor.py
│       ├── database.py
│       └── exceptions.py ← Duplicate hierarchy! ✗
├── frontend/             ← EMPTY! ✗
└── transcribe_state.db   ← State at repo root! ✗
```

**Problems:**
- ❌ Scripts importing backend (circular risk)
- ❌ Business logic not covered by tests
- ❌ Can't swap implementations (direct coupling)
- ❌ 6 unused dependencies

### Target (Clean Architecture)
```
stt-faster/
├── backend/
│   ├── domain/                # Pure business logic
│   │   ├── models.py         # TranscriptionResult, ModelConfig
│   │   ├── exceptions.py     # Unified hierarchy
│   │   └── services.py       # Abstract interfaces
│   ├── application/           # Use cases
│   │   ├── process_audio_batch.py
│   │   └── model_presets.py
│   └── infrastructure/        # External dependencies
│       ├── whisper/
│       │   ├── model_loader.py
│       │   └── transcription_service.py
│       └── persistence/
│           ├── state_database.py
│           └── file_repository.py
├── scripts/
│   └── transcribe_manager.py # CLI only (thin wrapper)
└── tests/
    ├── unit/domain/          # Fast, no I/O
    ├── unit/application/     # Mocked services
    └── integration/          # Real implementations
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Testable with mocks
- ✅ Swappable implementations
- ✅ No circular dependencies
- ✅ All business logic covered

---

## 🚀 Quick Wins (Do These First)

Can be completed in **~2 hours** with immediate impact:

### 1. Remove Unused Dependencies (15 min)
```bash
cd /home/kristjans/projects/stt-faster
uv remove langchain langchain-community sentence-transformers

# If torch only used for embeddings (not Whisper):
# uv remove torch
```

**Impact:** -500MB dependencies, faster installs, clearer project scope

### 2. Fix .gitignore (2 min)
```bash
echo "" >> .gitignore
echo "# State database" >> .gitignore
echo "transcribe_state.db" >> .gitignore
echo "*.db" >> .gitignore
```

**Impact:** Stop tracking state files

### 3. Move Database to User Directory (30 min)

Edit `scripts/transcription/database.py`:
```python
from pathlib import Path
import os

def get_default_db_path() -> Path:
    """Get user-specific state directory (XDG Base Directory spec)."""
    if os.name == 'nt':  # Windows
        base = Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    else:  # Linux/macOS
        base = Path(os.getenv('XDG_DATA_HOME', Path.home() / '.local/share'))
    
    db_dir = base / 'stt-faster'
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / 'transcribe_state.db'

class TranscriptionDatabase:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(get_default_db_path())
        self.db_path = db_path
        # ... rest of __init__
```

**Impact:** No more repo pollution, multi-user friendly

### 4. Unify Exception Hierarchies (1 hour)

```bash
# 1. Copy all exceptions to backend/exceptions.py
cat scripts/transcription/exceptions.py >> backend/exceptions.py

# 2. Update imports (3-4 files)
# In scripts/transcription/processor.py, database.py:
# Change: from .exceptions import ...
# To: from backend.exceptions import ...

# 3. Delete duplicate
rm scripts/transcription/exceptions.py

# 4. Run tests
.venv/bin/python -m pytest tests/unit -q
```

**Impact:** Single exception hierarchy, easier error handling

### 5. Update Coverage Config (5 min)

Edit `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = [
    # ... existing options ...
    "--cov=backend",
    "--cov=scripts.transcription",  # Add this line
    # ... rest ...
]
```

**Impact:** Track coverage for business logic in scripts/

### 6. Delete Ghost Frontend (15 min)
```bash
# Option A: Delete it
rm -rf frontend/

# Option B: Explain it
echo "# Frontend (Future Work)\n\nPlanned web UI - not yet implemented." > frontend/README.md

# Update pyproject.toml
# Remove 'frontend' from known_first_party list
```

**Impact:** No misleading structure

---

## 📋 5-Phase Refactoring Plan

### Phase 1: Clarify Identity (1-2 days)
- [ ] **Decide:** Pure transcription tool or RAG system?
- [ ] Remove unused dependencies
- [ ] Delete/document ghost frontend
- [ ] Update README and docs

**Validation:** No references to unused features

### Phase 2: Reorganize Modules (2-3 days)
- [ ] Create domain/application/infrastructure layers
- [ ] Move code to correct locations
- [ ] Delete `scripts/transcription/` (move to backend)
- [ ] Update all imports

**Validation:** Architecture boundaries respected

### Phase 3: Config & State (1 day)
- [ ] Centralize configuration
- [ ] Move database to XDG location
- [ ] Add schema versioning
- [ ] Create .env.example

**Validation:** No state files in repo

### Phase 4: Service Boundaries (2-3 days)
- [ ] Implement abstract interfaces
- [ ] Create test doubles
- [ ] Add contract tests
- [ ] Refactor use cases to use interfaces

**Validation:** Unit tests run <5s with no network

### Phase 5: Documentation (1 day)
- [ ] ARCHITECTURE.md with diagrams
- [ ] Update DEVELOPMENT.md
- [ ] Comprehensive docstrings
- [ ] README accuracy

**Validation:** New dev can onboard in <30min

---

## 🎯 Decision Point: Project Identity

**You MUST choose one path before starting refactoring:**

### Option A: Pure Transcription Tool ✅ RECOMMENDED
**Remove:**
- langchain
- langchain-community  
- sentence-transformers
- torch (if only used for embeddings)
- Ollama references
- RAG config variables

**Keep:**
- faster-whisper
- ctranslate2
- huggingface-hub
- hf-transfer

**Result:** Focused, lightweight CLI tool for audio transcription

### Option B: RAG/LLM System with Transcription
**Keep:**
- All current dependencies

**Must implement:**
- Vector store for transcriptions
- LangChain RAG pipeline
- Ollama integration
- Query interface
- Web API or UI

**Result:** Full-featured document intelligence system

---

## 🔍 Validation Checklist

After each phase, run these checks:

### Architecture Boundaries
```bash
# Domain should not import infrastructure
rg "from backend.infrastructure" backend/domain/
# Should be empty

# Application should not import infrastructure directly  
rg "from backend.infrastructure" backend/application/
# Should be empty

# Scripts should not import business logic modules
rg "from scripts.transcription" backend/
# Should be empty (folder deleted after Phase 2)
```

### Test Coverage
```bash
# All business logic covered
.venv/bin/python -m pytest --cov=backend --cov=scripts.transcription --cov-report=term-missing

# Unit tests are fast
time make unit
# Should complete in <5 seconds

# No network in unit tests
.venv/bin/python -m pytest tests/unit --allow-hosts=none
# Should pass
```

### Code Quality
```bash
# Linting passes
make ruff-check

# Type checking passes  
make pyright

# Pre-commit passes
uv run pre-commit run --all-files
```

### Functional
```bash
# CLI still works
.venv/bin/python scripts/transcribe_manager.py --help

# Sample transcription works
.venv/bin/python scripts/transcribe_manager.py process tests/test.mp3 --preset turbo
```

---

## 📈 Success Metrics

| Metric | Before | After | Goal |
|--------|--------|-------|------|
| **Architecture violations** | Many | 0 | 0 |
| **Test coverage** | ~60% | >80% | >80% |
| **Unused dependencies** | 6 | 0 | 0 |
| **Unit test speed** | Variable | <5s | <10s |
| **LOC in correct layer** | ~30% | >90% | >90% |
| **Duplicate code** | 2 hierarchies | 0 | 0 |
| **Time to onboard** | 2-3h | <30min | <1h |

---

## 🛟 Rollback Plan

Each phase = separate git branch:
```bash
# Start phase
git checkout -b phase-1-identity-cleanup

# Work...

# Validate
make test && make pyright && make ruff-check

# If validation fails:
git checkout main
git branch -D phase-1-identity-cleanup
# Adjust plan and restart
```

---

## 📞 Getting Help

1. **Review full plan:** `.cursor/plans/architecture_refactoring_plan.md` (25KB detailed analysis)
2. **Architecture questions:** See proposed layer structure in full plan
3. **Implementation details:** Each phase has detailed task breakdown
4. **Stuck on a phase:** Check validation criteria and rollback if needed

---

## 🎨 Architecture Diagram

```
┌──────────────────────────────────────────┐
│         CLI (transcribe_manager.py)      │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│        APPLICATION LAYER                 │
│  ┌────────────────────────────────┐      │
│  │ ProcessAudioBatchUseCase       │      │
│  │  execute(folder, preset)       │      │
│  └────────────────────────────────┘      │
└───┬──────────────────┬───────────────────┘
    │                  │
    ▼                  ▼
┌───────────────┐  ┌──────────────────────┐
│ DOMAIN        │  │ INFRASTRUCTURE       │
│               │  │                      │
│ Models        │  │ WhisperService       │
│ Exceptions    │  │ FileRepository       │
│ Interfaces    │  │ StateDatabase        │
│               │  │ ModelLoader          │
│ No external   │  │                      │
│ dependencies! │  │ Adapters for         │
│               │  │ external systems     │
└───────────────┘  └──────────────────────┘

Dependency Rule: 
  → Outer layers depend on inner layers
  → Inner layers NEVER depend on outer layers
  → Domain has ZERO external dependencies
```

---

**Next Step:** Review with team, choose Option A or B, run Quick Wins, then start Phase 1.

