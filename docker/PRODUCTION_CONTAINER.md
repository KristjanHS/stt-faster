# Production Container - Cloud-Native Verification

This document describes the cloud-native characteristics of the production Docker container (`Dockerfile` at project root).

## Quick Verification

```bash
# Build the production image
docker build -t stt-faster:latest .

# Run verification script
./scripts/verify_production_image.sh

# Run comprehensive tests
.venv/bin/python -m pytest tests/e2e/test_production_container.py -v -m "not slow and not network"
```

## Cloud-Native Characteristics ✅

### 1. **Security & Isolation**
- ✅ Runs as non-root user (`appuser`, UID 1000)
- ✅ No `sudo` or privilege escalation tools
- ✅ Minimal package footprint (no build tools like gcc, make)
- ✅ No sensitive data in environment variables
- ✅ Proper file ownership and permissions

### 2. **Observability**
- ✅ Healthcheck configured and working
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
      CMD ["python", "-c", "import backend.transcribe; import backend.processor; print('OK')"]
  ```
- ✅ Unbuffered Python output (`PYTHONUNBUFFERED=1`) for immediate logs
- ✅ OCI-compliant image labels (title, description, etc.)

### 3. **Reproducibility**
- ✅ Uses Debian snapshot repositories for deterministic builds
- ✅ Digest-pinned base images (Python 3.12 slim)
- ✅ Locked dependency versions via `uv.lock`
- ✅ Multi-stage build for smaller final image (780MB)

### 4. **Stateless & Portable**
- ✅ All state externalized to volumes:
  - `/workspace` - Audio files to process
  - `/home/appuser/.cache/huggingface` - Model cache (persistent)
  - `/home/appuser/.local/share/stt-faster` - Application state DB
- ✅ No hardcoded paths or assumptions about host environment
- ✅ Works with user-provided UID/GID mapping

### 5. **Efficient Resource Usage**
- ✅ Virtual environment properly configured (`/opt/venv`)
- ✅ Minimal base image (Python 3.12 slim on Debian Bookworm)
- ✅ Build cache optimization via layer ordering
- ✅ No unnecessary dependencies (production only, no test/dev tools)

### 6. **12-Factor App Compliance**
- ✅ **Codebase**: Single codebase, multiple deployments via Docker
- ✅ **Dependencies**: Explicitly declared in `pyproject.toml` and `uv.lock`
- ✅ **Config**: Environment variables for configuration (LOG_LEVEL, HF_HOME)
- ✅ **Backing services**: Models treated as attached resources
- ✅ **Build, release, run**: Strict separation via multi-stage build
- ✅ **Processes**: Stateless (state in external volumes)
- ✅ **Port binding**: N/A (CLI tool, not a service)
- ✅ **Concurrency**: Scale via multiple containers
- ✅ **Disposability**: Fast startup, graceful shutdown
- ✅ **Dev/prod parity**: Same container for dev and production
- ✅ **Logs**: stdout/stderr (unbuffered)
- ✅ **Admin processes**: Via same container with different commands

## Usage Examples

### Basic Usage
```bash
# Show help
docker run --rm stt-faster:latest --help

# Process audio files
docker run --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  stt-faster:latest process /workspace/audio --preset turbo

# Check status
docker run --rm \
  -v $(pwd):/workspace \
  -v ~/.local/share/stt-faster:/home/appuser/.local/share/stt-faster \
  stt-faster:latest status --verbose
```

### Cloud Deployment (Kubernetes)
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: transcribe-job
spec:
  template:
    spec:
      containers:
      - name: transcribe
        image: stt-faster:latest
        command: ["python", "/app/scripts/transcribe_manager.py"]
        args: ["process", "/workspace/audio", "--preset", "turbo"]
        volumeMounts:
        - name: audio-files
          mountPath: /workspace
        - name: model-cache
          mountPath: /home/appuser/.cache/huggingface
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
      volumes:
      - name: audio-files
        persistentVolumeClaim:
          claimName: audio-pvc
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      restartPolicy: OnFailure
```

### Docker Compose
```yaml
version: '3.8'
services:
  transcribe:
    image: stt-faster:latest
    volumes:
      - ./audio:/workspace
      - model-cache:/home/appuser/.cache/huggingface
      - app-data:/home/appuser/.local/share/stt-faster
    command: ["process", "/workspace", "--preset", "turbo"]
    user: "1000:1000"
    
volumes:
  model-cache:
  app-data:
```

## Testing

### Unit Tests (24 tests covering):
- Image build and size optimization
- Cloud-native characteristics (user, healthcheck, env vars)
- Runtime behavior (entrypoint, Python version, modules)
- Volume mounts and permissions
- Security (no sudo, minimal packages)
- OCI labels

### Run Tests
```bash
# All non-slow, non-network tests
.venv/bin/python -m pytest tests/e2e/test_production_container.py -v -m "not slow and not network"

# Specific test class
.venv/bin/python -m pytest tests/e2e/test_production_container.py::TestProductionCloudNative -v

# Include slow/network tests (downloads models)
.venv/bin/python -m pytest tests/e2e/test_production_container.py -v
```

## Comparison: Production vs Dev Containers

| Feature | Production (`Dockerfile`) | Dev/Test (`docker/app.Dockerfile`) |
|---------|---------------------------|-----------------------------------|
| **Purpose** | End users, cloud deployment | Developers, CI/CD testing |
| **Dependencies** | Production only (no test deps) | Includes pytest, dev tools |
| **Entrypoint** | `transcribe_manager.py` | `backend.main` (keepalive) |
| **Volumes** | User audio files | Source code (live reload) |
| **Size** | 780MB (optimized) | Larger (includes test tools) |
| **Use Case** | Run transcription tasks | Run tests, develop features |

## Troubleshooting

### Healthcheck Failing
```bash
# Test healthcheck manually
docker run --rm --entrypoint="" stt-faster:latest \
  python -c "import backend.transcribe; import backend.processor; print('OK')"
```

### Permission Issues
```bash
# Run with host user UID/GID
docker run --rm -u "$(id -u):$(id -g)" \
  -v $(pwd):/workspace \
  stt-faster:latest process /workspace/audio
```

### Model Download Issues
```bash
# Pre-download models to cache
docker run --rm \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  stt-faster:latest process /workspace --preset turbo --help
```

## References

- [Dockerfile](../Dockerfile) - Production Dockerfile
- [Test Suite](../tests/e2e/test_production_container.py) - Comprehensive E2E tests
- [Verification Script](../scripts/verify_production_image.sh) - Quick validation
- [Dev Container](./README.md) - Dev/test Docker setup
- [Main README](../README.md#docker-usage) - User-facing documentation

## Continuous Integration

The production container is automatically tested in CI via:
- Build verification
- Cloud-native characteristic validation
- Security and compliance checks
- Functional testing

All tests must pass before merge to ensure cloud-ready deployments.

