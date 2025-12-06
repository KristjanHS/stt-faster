# Docker Setup

This directory contains **development/test Docker configuration** for running integration tests in containers.

## 🎯 Two Docker Setups

### 1️⃣ **Production Docker** (for end users)

**Location:** `../Dockerfile` (project root)  
**Purpose:** Run stt-faster without installing Python/dependencies  
**Usage:** `./scripts/transcribe-docker process ./audio --preset turbo`

See [Main README](../README.md#docker-usage) for full usage guide.

---

### 2️⃣ **Dev/Test Docker** (this directory)

**Location:** `docker/docker-compose.yml` + `docker/app.Dockerfile`  
**Purpose:** Run tests in containers (used by CI and developers)  
**Usage:** `make docker-back && make docker-unit`

---

## 📦 Production Docker (End Users)

### Quick Start

```bash
# Build the image
docker build -t stt-faster:latest .

# Use the wrapper script (recommended)
./scripts/transcribe-docker process /path/to/audio --preset turbo

# Or run directly
docker run --rm -v $(pwd):/workspace stt-faster:latest process /workspace/audio.mp3
```

### Features

- ✅ No Python installation required
- ✅ No dependency management needed
- ✅ Minimal image size (production deps only)
- ✅ Models cached in `~/.cache/hf`
- ✅ State persisted in `~/.local/share/stt-faster`

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `$(pwd)` | `/workspace` | Audio files to process |
| `~/.cache/hf` | `/home/appuser/.cache/hf` | Whisper model cache |
| `~/.local/share/stt-faster` | `/home/appuser/.local/share/stt-faster` | Transcription state DB |

### Examples

```bash
# Process all files in directory
./scripts/transcribe-docker process ./audio --preset turbo

# List available models
./scripts/transcribe-docker list-models

# Show help
./scripts/transcribe-docker --help

# Custom image name
STT_DOCKER_IMAGE=my-stt:v1 ./scripts/transcribe-docker process ./audio
```

---

## 🧪 Dev/Test Docker (Developers)

### Purpose

- Run integration tests in isolated containers
- Test Dockerfile builds in CI
- Consistent test environment across machines

### Files

```
docker/
├── docker-compose.yml   # Dev/test services (app only, no Ollama)
├── app.Dockerfile       # Dev/test image (includes test deps)
└── README.md            # This file
```

### Quick Start

```bash
# Build and start dev container
make docker-back

# Run unit tests inside container
make docker-unit

# Stop containers
docker compose -f docker/docker-compose.yml down
```

### Manual Usage

```bash
# Build dev image
docker compose -f docker/docker-compose.yml build

# Start container in background
docker compose -f docker/docker-compose.yml up -d

# Run tests inside container
docker compose -f docker/docker-compose.yml exec -T app \
    python -m pytest tests/unit -v

# View logs
docker compose -f docker/docker-compose.yml logs -f app

# Stop and remove containers
docker compose -f docker/docker-compose.yml down
```

### Differences from Production

| Feature | Production (`../Dockerfile`) | Dev/Test (`docker-compose.yml`) |
|---------|------------------------------|--------------------------------|
| **Dependencies** | Production only | Includes test deps |
| **Purpose** | End users | CI and developers |
| **Entrypoint** | `transcribe_manager.py` | `backend.main` (keepalive) |
| **Volumes** | User audio files | Backend/tests/logs (live reload) |
| **Size** | Minimal | Larger (test tools) |

---

## 🔧 Architecture Notes

### Why Two Dockerfiles?

1. **Production (`../Dockerfile`):**
   - Optimized for end users
   - Minimal dependencies (no pytest, no dev tools)
   - Clear entrypoint for transcription
   - Small image size

2. **Dev/Test (`app.Dockerfile`):**
   - Includes test dependencies
   - Mounts source code for live reload
   - Used by CI/CD pipelines
   - Runs `backend.main` keepalive

### Removed: Ollama Service

**Previous setup** included an Ollama container for LLM/RAG features.  
**Current state:** Removed (not needed for pure transcription tool).

If you need LLM features:
- See `.cursor/plans/architecture_refactoring_plan.md`
- Consider Issue #1: Identity Crisis

---

## 🚀 CI/CD Integration

The dev/test Docker setup is used in CI workflows:

```yaml
# .github/workflows/test.yml
- name: Run tests in Docker
  run: |
    docker compose -f docker/docker-compose.yml up -d
    docker compose -f docker/docker-compose.yml exec -T app \
        python -m pytest tests/unit tests/integration -v
```

---

## 📝 Development Workflow

### Local Development (no Docker)

```bash
# Preferred for fast iteration
.venv/bin/python scripts/transcribe_manager.py process ./audio
```

### Test with Production Docker

```bash
# Build production image
docker build -t stt-faster:latest .

# Test as end user would
./scripts/transcribe-docker process ./test_audio --preset turbo
```

### Test with Dev Docker

```bash
# For integration tests that need isolation
make docker-back
make docker-unit
```

---

## 🆘 Troubleshooting

### Production Docker

**Image not found:**
```bash
docker build -t stt-faster:latest .
```

**Permission errors:**
```bash
# Use user's UID/GID
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/workspace stt-faster:latest
```

**Model download fails:**
```bash
# Ensure HF cache directory is writable
mkdir -p ~/.cache/hf
chmod 755 ~/.cache/hf
```

### Dev/Test Docker

**Container won't start:**
```bash
# Check logs
docker compose -f docker/docker-compose.yml logs app

# Rebuild
docker compose -f docker/docker-compose.yml build --no-cache
```

**Tests fail in container but pass locally:**
```bash
# Ensure volumes are mounted correctly
docker compose -f docker/docker-compose.yml config | grep volumes -A 10
```

---

## 📚 Further Reading

- [Main README](../README.md) - Project overview and setup
- [Testing Approach](../docs/testing_approach.md) - Test strategy
- [Architecture Plan](../.cursor/plans/architecture_refactoring_plan.md) - Refactoring details

