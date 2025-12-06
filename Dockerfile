# syntax=docker/dockerfile:1.7
##########################################################################
# Production Dockerfile for end users
# 
# Purpose: Run stt-faster transcription tool without installing dependencies
# Usage: See README.md or run: docker run stt-faster --help
##########################################################################

# ARGs used in FROM statements must be defined before first FROM
# Override UV_IMAGE_REF to change base; default is digest‑pinned for reproducibility.
ARG UV_IMAGE_REF=ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:c6fad5e08092142ea53fc03cbebb6304c5d06a25dd53ab450114f6c5f03376a7
# Override PYTHON_RUNTIME_IMAGE to change base; default is digest‑pinned for reproducibility.
ARG PYTHON_RUNTIME_IMAGE=python:3.12-slim-bookworm@sha256:3ad2a947749a3eb74acd9e00636ffa0def5aae0bbbd9fa4fff6253e404e2fe15

############################
# Build stage: Python deps #
############################
# hadolint ignore=DL3006
FROM ${UV_IMAGE_REF} AS builder

ENV VENV_PATH=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app

# Install ONLY production dependencies (no test group)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-install-project --no-dev

# Install the project itself
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev

############################################
# Runtime stage: Minimal production image #
############################################
# hadolint ignore=DL3006
FROM ${PYTHON_RUNTIME_IMAGE} AS runtime

# Debian snapshot for reproducible builds
ARG SNAPSHOT=20250906T000000Z

# Configure Debian snapshot repository
RUN set -eux; \
    rm -f /etc/apt/sources.list; \
    printf '%s\n' \
    'Types: deb' \
    "URIs: https://snapshot.debian.org/archive/debian/${SNAPSHOT}/" \
    'Suites: bookworm bookworm-updates' \
    'Components: main contrib non-free non-free-firmware' \
    'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
    'Check-Valid-Until: no' \
    '' \
    'Types: deb' \
    "URIs: https://snapshot.debian.org/archive/debian-security/${SNAPSHOT}/" \
    'Suites: bookworm-security' \
    'Components: main contrib non-free non-free-firmware' \
    'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
    'Check-Valid-Until: no' \
    > /etc/apt/sources.list.d/debian.sources

# Install minimal runtime dependencies
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python environment setup
ENV VENV_PATH=/opt/venv
# hadolint ignore=SC2269
# Intentionally prepend venv bin to PATH while preserving base PATH
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Copy prebuilt virtualenv from builder
COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"

WORKDIR /app
ENV HOME=/home/appuser

# Create non-root user
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} appgroup && \
    useradd -l -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash appuser

# Hugging Face cache directory (for Whisper models)
ENV HF_HOME=/home/appuser/.cache/hf

# Force CPU-only for cloud portability (no cuDNN in slim image)
# Local runs can use GPU by not setting this or setting STT_DEVICE=cuda
ENV STT_DEVICE=cpu

# Create folders and set permissions for non-root user
RUN mkdir -p /workspace /home/appuser/.local/share/stt-faster "${HF_HOME}" /app && \
    chown -R appuser:appgroup /workspace /home/appuser /app

# Copy application code (from builder stage already has backend, add scripts)
COPY --chown=appuser:appgroup backend/ /app/backend/
COPY --chown=appuser:appgroup scripts/ /app/scripts/

USER appuser

# Default: show help (users mount their audio files to /workspace)
WORKDIR /workspace
ENTRYPOINT ["python", "/app/scripts/transcribe_manager.py"]
CMD ["--help"]

# Health check (verify Python environment is working)
# Note: This is a basic import check since transcribe_manager doesn't have a --healthcheck flag
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import backend.transcribe; import backend.processor; print('OK')"]

# Labels for metadata
LABEL org.opencontainers.image.title="stt-faster"
LABEL org.opencontainers.image.description="Fast audio transcription using Whisper models"
LABEL org.opencontainers.image.authors="stt-faster contributors"
LABEL org.opencontainers.image.source="https://github.com/yourusername/stt-faster"
LABEL org.opencontainers.image.documentation="https://github.com/yourusername/stt-faster/blob/main/README.md"

