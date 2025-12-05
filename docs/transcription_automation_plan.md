# Audio Transcription Automation - Implementation Plan

## Context
- **Source folder:** `C:\Users\PC\Downloads\transcribe\` (accessible via `/mnt/c/Users/PC/Downloads/transcribe/` in WSL2)
- **Existing capability:** `backend/transcribe.py` with `transcribe_to_json()` function
- **Requirements:** Manual trigger, move processed files to archive

---

## Alternative 1: Basic SQLite Tracker ⭐ SIMPLEST [IMPLEMENTING]
**Focus:** Clean structure with minimal features

### Description
Multi-file implementation with basic SQLite tracking:
- Database module with simple class interface
- Processing module with basic error handling
- CLI module with essential commands
- Simple schema (no retry tracking, no timestamps)
- Sequential processing only (no concurrency)
- Basic unit tests included

### Usage
```bash
# Process folder
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe

# Check status
.venv/bin/python scripts/transcribe_manager.py status
```

### File Structure
```
scripts/
├── transcribe_manager.py        # CLI entry point (~100 lines)
├── transcription/
│   ├── __init__.py
│   ├── database.py              # Database operations (~120 lines)
│   └── processor.py             # Processing logic (~120 lines)
transcribe_state.db              # SQLite database
tests/
└── unit/
    ├── test_transcribe_database.py
    └── test_transcribe_processor.py
```

### Database Schema
```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,               -- 'pending', 'completed', 'failed'
    error_message TEXT
);
```

### Key Features
✅ Persistent state tracking
✅ Files moved to `processed/` or `failed/` subfolders
✅ Status reporting command
✅ Clean separation of concerns
✅ Basic unit tests
✅ Simple to understand and modify

### Limitations
❌ No automatic retry logic (manual retry via re-run)
❌ No timestamps
❌ No concurrent processing
❌ No detailed history
❌ Single attempt per file

### Implementation Time
~2-3 hours

---

## Alternative 2: Structured SQLite Manager ⭐ RECOMMENDED
**Focus:** Production-ready with reasonable complexity

### Description
Enhanced implementation with:
- Enhanced schema with timestamps and retry tracking
- Automatic retry logic (3 attempts)
- Optional concurrent processing (--workers flag)
- Multiple CLI commands (process, status, retry, history)
- Comprehensive error tracking
- Logging to file + console

### Usage
```bash
# Process folder (sequential)
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe

# Process with 4 workers (concurrent)
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe --workers 4

# Check status
.venv/bin/python scripts/transcribe_manager.py status

# Retry failed files
.venv/bin/python scripts/transcribe_manager.py retry

# Show history for specific file
.venv/bin/python scripts/transcribe_manager.py history "audio_file.wav"
```

### File Structure
```
scripts/
├── transcribe_manager.py        # CLI entry point (~150 lines)
├── transcription/
│   ├── __init__.py
│   ├── database.py              # Database operations (~200 lines)
│   ├── processor.py             # Processing logic (~180 lines)
│   └── utils.py                 # Helper functions (~80 lines)
transcribe_state.db              # SQLite database
logs/
└── transcribe_manager.log       # Log file
tests/
└── unit/
    ├── test_transcribe_database.py
    ├── test_transcribe_processor.py
    └── test_transcribe_retry.py
```

### Enhanced Schema
```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,               -- 'pending', 'processing', 'completed', 'failed', 'quarantined'
    attempts INTEGER DEFAULT 0,
    error_message TEXT,
    json_output_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_updated (updated_at)
);
```

### Key Features (beyond Alternative 1)
✅ Automatic retry logic (3 attempts)
✅ Concurrent processing with worker pool
✅ Timestamps for tracking
✅ Enhanced error messages
✅ File history queries
✅ Quarantine for permanently failed files

### Implementation Time
~4-5 hours

---

## Alternative 3: Production-Grade with Best Practices ⭐ MOST ROBUST
**Focus:** Enterprise-ready with comprehensive error handling

### Description
Full production implementation with:
- State machine for status transitions
- Context managers for resource safety
- Configuration file support (YAML)
- Database migrations
- Transaction management with rollback
- Comprehensive testing suite
- Type hints throughout
- Pydantic validation
- Documentation strings

### Usage
```bash
# Initialize (one-time setup)
.venv/bin/python -m backend.transcribe_cli init --config config.yaml

# Process with all features
.venv/bin/python -m backend.transcribe_cli process \
    --input /mnt/c/Users/PC/Downloads/transcribe \
    --workers 4 \
    --preset distil \
    --dry-run  # Preview what would be processed

# Comprehensive status
.venv/bin/python -m backend.transcribe_cli status --format table
.venv/bin/python -m backend.transcribe_cli status --format json > status.json

# Advanced retry with filters
.venv/bin/python -m backend.transcribe_cli retry --max-age 24h --error-pattern "CUDA"

# Database maintenance
.venv/bin/python -m backend.transcribe_cli db vacuum
.venv/bin/python -m backend.transcribe_cli db migrate
```

### File Structure
```
backend/
├── transcribe_cli/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point (~100 lines)
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── process.py           # Process command (~150 lines)
│   │   ├── status.py            # Status command (~100 lines)
│   │   ├── retry.py             # Retry command (~120 lines)
│   │   └── db.py                # DB maintenance (~80 lines)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py          # Database with context managers (~300 lines)
│   │   ├── processor.py         # Processing with state machine (~250 lines)
│   │   ├── config.py            # Configuration management (~150 lines)
│   │   └── state_machine.py    # Status transition logic (~100 lines)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transcription.py    # Data models with validation (~100 lines)
│   │   └── schemas.py           # Pydantic schemas (~80 lines)
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Advanced logging setup (~100 lines)
│       ├── file_utils.py        # File operations (~80 lines)
│       └── validation.py        # Input validation (~60 lines)
├── migrations/
│   ├── 001_initial.sql
│   └── 002_add_metadata.sql
config.yaml                      # User configuration
transcribe_state.db              # SQLite database
logs/
├── transcribe_cli.log          # Main log (rotated)
├── transcribe_cli.2024-12-04.log
└── errors.log                   # Error-only log
tests/
└── unit/
    ├── test_transcribe_database.py
    ├── test_transcribe_processor.py
    ├── test_transcribe_state_machine.py
    └── test_transcribe_config.py
```

### Configuration File (config.yaml)
```yaml
transcription:
  input_folder: /mnt/c/Users/PC/Downloads/transcribe
  archive_folder: processed
  failed_folder: failed
  model_preset: distil
  max_attempts: 3
  workers: 4
  
processing:
  batch_size: 10
  timeout_seconds: 300
  retry_backoff: exponential  # linear, exponential
  
database:
  path: transcribe_state.db
  backup_enabled: true
  backup_interval_hours: 24
  
logging:
  level: INFO
  file: logs/transcribe_cli.log
  rotation: daily
  retention_days: 30
  format: "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
```

### Enhanced Schema
```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT,                     -- SHA256 for deduplication
    file_size INTEGER,
    status TEXT NOT NULL,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    error_traceback TEXT,
    json_output_path TEXT,
    audio_duration REAL,
    model_preset TEXT,
    processing_time REAL,               -- seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata TEXT,                      -- JSON blob for extensibility
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'quarantined', 'cancelled'))
);

CREATE TABLE processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id INTEGER NOT NULL,
    status_from TEXT,
    status_to TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcription_id) REFERENCES transcriptions(id)
);

CREATE INDEX idx_status ON transcriptions(status);
CREATE INDEX idx_updated ON transcriptions(updated_at);
CREATE INDEX idx_hash ON transcriptions(file_hash);
```

### Key Features (beyond Alternative 2)
✅ State machine prevents invalid transitions
✅ Transaction management (atomic operations)
✅ Configuration file (no hardcoded values)
✅ Database migrations
✅ File deduplication (hash checking)
✅ Processing history audit trail
✅ Type hints + Pydantic validation
✅ Comprehensive test suite
✅ Log rotation and retention
✅ Dry-run mode
✅ Graceful shutdown (SIGTERM handling)

### Implementation Time
~1-2 weeks

---

## Comparison Matrix

| Feature | Alt 1: Basic | Alt 2: Structured | Alt 3: Production |
|---------|--------------|-------------------|-------------------|
| **Implementation Time** | 2-3 hours | 4-5 hours | 1-2 weeks |
| **Total Lines of Code** | ~350 | ~600 | ~2000 |
| **Number of Files** | 4 | 5 | 15+ |
| **Database Tables** | 1 | 1 | 2 |
| **Retry Logic** | Manual | Automatic (3x) | Configurable |
| **Concurrent Processing** | No | Yes (optional) | Yes (advanced) |
| **Timestamps** | No | Yes | Yes + audit trail |
| **Configuration** | Hardcoded | Args only | Config file |
| **Error Details** | Basic | Good | Comprehensive |
| **Testing** | Basic tests | Unit testable | Full test suite |
| **State Machine** | No | Implicit | Explicit |
| **Transaction Safety** | No | Partial | Full |
| **Logging** | Console | File + console | Rotated files |
| **File Deduplication** | No | No | Yes (hash) |
| **Maintenance Effort** | Low | Medium | Higher |

---

## Recommendation

**Start with Alternative 1** for quick implementation (currently implementing).

Upgrade to Alternative 2 if you need:
- Automatic retry logic
- Concurrent processing
- Better error tracking
- Timestamps and history

Upgrade to Alternative 3 if you need:
- Production reliability
- Audit trails
- Configuration management
- Multiple users/deployments

---

## Implementation Progress

### Alternative 1 - ✅ COMPLETED

- [x] Create directory structure
- [x] Implement database.py (~172 lines)
- [x] Implement processor.py (~207 lines)
- [x] Implement transcribe_manager.py CLI (~186 lines)
- [x] Create unit tests (test_transcribe_database.py, test_transcribe_processor.py)
- [x] Test end-to-end workflow
- [x] Document usage
- [x] Test with real audio files

### Implementation Summary (December 5, 2024)

**Status:** ✅ **FULLY FUNCTIONAL**

All components have been implemented and tested successfully:

1. **Unit Tests**: All 22 tests passing (10 database tests, 12 processor tests)
2. **End-to-End Testing**: Successfully processed a 40-minute M4A audio file
   - File: `AI_Models_Learn_to_Cheat_and_Lie.m4a` (78MB)
   - Processing time: ~1 minute 20 seconds
   - Output: 96KB JSON file with timestamped segments
   - File correctly moved to `processed/` subfolder
3. **Database Tracking**: SQLite database correctly tracking file status
4. **Idempotency**: Re-running process command correctly skips already-processed files

### Key Features Verified

✅ Automatic file discovery (supports .wav, .mp3, .m4a, .flac, .ogg)  
✅ SQLite state persistence across runs  
✅ Files moved to `processed/` subfolder after successful transcription  
✅ Files moved to `failed/` subfolder if transcription fails  
✅ Status reporting via `status` command  
✅ Idempotent processing (won't re-process completed files)  
✅ Clean separation of concerns (database, processor, CLI)  
✅ Comprehensive logging with timestamps  
✅ Working with WSL2 Windows folder paths (`/mnt/c/...`)

### Usage Examples

```bash
# Process all audio files in the Windows Downloads folder
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe

# Check status of all tracked files
.venv/bin/python scripts/transcribe_manager.py status

# Use a different model preset (turbo, distil, large8gb)
.venv/bin/python scripts/transcribe_manager.py process /path/to/folder --preset turbo
```

### What's Next

Alternative 1 is fully functional and ready for production use. Consider upgrading to:
- **Alternative 2** if you need automatic retry logic or concurrent processing
- **Alternative 3** if you need production-grade features like audit trails and configuration management
