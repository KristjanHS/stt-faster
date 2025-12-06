# syntax=docker/dockerfile:1.7
##########
# Two-stage build: builder (Python deps via uv) → runtime (OS deps + app)
##########

# ARGs used in FROM statements must be defined before first FROM
# Override UV_IMAGE_REF to change base; default is digest‑pinned for reproducibility.
# Got digest from `docker pull ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
ARG UV_IMAGE_REF=ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:c6fad5e08092142ea53fc03cbebb6304c5d06a25dd53ab450114f6c5f03376a7
# Override PYTHON_RUNTIME_IMAGE to change base; default is digest‑pinned for reproducibility.
# Got digest from `docker pull python:3.12-slim-bookworm`
ARG PYTHON_RUNTIME_IMAGE=python:3.12-slim-bookworm@sha256:3ad2a947749a3eb74acd9e00636ffa0def5aae0bbbd9fa4fff6253e404e2fe15

############################
# Build stage: wheels/venv #
############################
# hadolint ignore=DL3006
FROM ${UV_IMAGE_REF} AS builder

ENV VENV_PATH=/opt/venv \
  UV_PROJECT_ENVIRONMENT=/opt/venv \
  UV_LINK_MODE=copy

WORKDIR /app

# uv phase 1 (to optimize uv cache use): install ONLY deps (lockfile-driven) into /opt/venv
#    (Copy only files needed for dependency resolution)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv sync --locked --no-install-project --group test

# uv phase 2: copy project and install the project as package

# Copy only backend/ that will be packaged here.
# NB! frontend/ will be referenced as folder in runtime image, so it will be copied in runtime stage below
COPY backend/ /app/backend/

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv sync --locked --group test;

############################################
# Runtime stage: Debian + apt + your app  #
############################################
# hadolint ignore=DL3006
FROM ${PYTHON_RUNTIME_IMAGE} AS runtime

# --- Debian snapshot date (set to base image date in CI) ---
# Accepts YYYYMMDD or YYYYMMDDTHHMMSSZ. Example default is a placeholder.
ARG SNAPSHOT=20250906T000000Z  # Bookworm 12.12 point release date

# Replace sources with a single Deb822 file pointing to snapshot.debian.org.
# Ensure SNAPSHOT is expanded by the shell (avoid quoted heredoc preventing expansion).
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

# Install OS runtime deps (single RUN; clean lists)
# Snapshot URLs pin package set; explicit versions unnecessary
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Python tuning
ENV VENV_PATH=/opt/venv
# hadolint ignore=SC2269
# Intentionally prepend venv bin to PATH while preserving base PATH
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
# Unbuffered: Forces unbuffered stdout/stderr (same as python -u) so logs show up immediately in Docker.

# Bring in the prebuilt venv (same Python ABI as runtime base)
COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"

WORKDIR /app
ENV HOME=/home/appuser
# Create a real non-root user with home, into the conventional /home/<username>
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} appgroup && \
  useradd -l -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash appuser

# Huggingface models cache that lives inside the image or named volume
# TODO: make a named volume for persistance of HF cache
ENV HF_HOME=/hf_cache

# NB! Never rely on chown in the image for bind mounts. A bind mount hides whatever you COPY/chowned at
# build-time and writes go back to the host filesystem. Instead, pre-create/own the host directories.

# Create folders and set permission for non-root user - this works for named volumes or image content
RUN mkdir -p /hf_cache /app /home/appuser \
  && chown -R appuser:appgroup /hf_cache /app /home/appuser

# Ship the folders that Prod needs, in the image (in Dev, these are hidden by bind mount)
COPY --chown=appuser:appgroup frontend/ /app/frontend/

USER appuser

# Minimal healthcheck via backend.main (fast import + exit 0)
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=30 \
  CMD ["python", "-m", "backend.main", "--healthcheck"]

# NB! ENV variables like ${VENV_PATH} are NOT expanded inside JSON-array CMD or ENTRYPOINT.
CMD ["python", "-m", "backend.main"]
