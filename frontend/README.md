# Frontend (Not Implemented)

Web UI for stt-faster transcription tool.

**Current status:** Placeholder only  
**Use instead:** `scripts/transcribe_manager.py` (CLI)

## Planned Features

When implemented, this could include:
- Upload audio files via browser
- View transcription history
- Manage model presets
- Real-time transcription status
- Batch processing dashboard

## Architecture

The frontend should:
- ✅ Call backend APIs (not import business logic directly)
- ✅ Use a proper web framework (Flask/FastAPI + React/Vue)
- ✅ Have its own tests in `tests/frontend/`
- ❌ Never import from `backend/` modules directly

## Getting Started (Future)

```bash
cd frontend
# Install dependencies (TBD)
# Run dev server (TBD)
```

For now, use the CLI:
```bash
.venv/bin/python scripts/transcribe_manager.py --help
```

