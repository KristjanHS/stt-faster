#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/new_project_bootstrap.sh <project-slug>

Copies this repo into ~/projects/<project-slug> as a clean starting point for a new project.
- Excludes .git, .venv, caches, and generated artifacts.
- Leaves cleanup and renaming to scripts/new_project_cleanup.sh inside the copied repo.
EOF
}

PROJECT_SLUG=${1:-}

if [[ -z "${PROJECT_SLUG}" ]]; then
    usage
    exit 1
fi

TARGET_DIR="${HOME}/projects/${PROJECT_SLUG}"

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but not found in PATH." >&2
    exit 1
fi

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPROJECT_PATH="${SOURCE_ROOT}/pyproject.toml"
ORIGINAL_SLUG="$(
    if [[ -f "${PYPROJECT_PATH}" ]]; then
        python3 - <<PY 2>/dev/null || true
import tomllib, pathlib
data = tomllib.loads(pathlib.Path("${PYPROJECT_PATH}").read_text())
print(data.get("project", {}).get("name", ""))
PY
    fi
)"
if [[ -z "${ORIGINAL_SLUG}" ]]; then
    ORIGINAL_SLUG="$(basename "${SOURCE_ROOT}")"
fi
ORIGINAL_MODULE="${ORIGINAL_SLUG//-/_}"
ABS_TARGET="$(realpath -m "${TARGET_DIR}")"

if [[ -e "${ABS_TARGET}" ]] && [[ -n "$(ls -A "${ABS_TARGET}")" ]]; then
    echo "Target directory '${ABS_TARGET}' is not empty. Choose an empty path." >&2
    exit 1
fi

mkdir -p "${ABS_TARGET}"
echo "About to copy template:"
echo " - source: ${SOURCE_ROOT}"
echo " - target: ${ABS_TARGET}"
echo " - original slug/module: ${ORIGINAL_SLUG} / ${ORIGINAL_MODULE}"
echo " - new slug: ${PROJECT_SLUG}"
echo " - exclusions: .git, .venv, logs, reports, caches, egg-info, db/log artifacts"
read -r -p "Proceed with rsync? [y/N] " copy_reply
if [[ ! "${copy_reply}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

EXCLUDES=(
    "--exclude=.git"
    "--exclude=.venv"
    "--exclude=logs/*"
    "--exclude=reports/*"
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=*.db"
    "--exclude=*.log"
    "--exclude=${ORIGINAL_MODULE}.egg-info"
    "--exclude=semgrep_local.sarif"
    "--exclude=transcribe_state.db"
)

rsync -a --info=name0 "${EXCLUDES[@]}" "${SOURCE_ROOT}/" "${ABS_TARGET}/"

# Persist the source repo origin URL for cleanup-time URL rewriting, if available
if command -v git >/dev/null 2>&1; then
    ORIGIN_URL="$(cd "${SOURCE_ROOT}" && git config --get remote.origin.url || true)"
    if [[ -n "${ORIGIN_URL}" ]]; then
        echo "${ORIGIN_URL}" > "${ABS_TARGET}/.bootstrap_origin_url"
    fi
fi
echo "${PROJECT_SLUG}" > "${ABS_TARGET}/.bootstrap_slug"

echo "Copied repo to ${ABS_TARGET}"
echo "Next: cd \"${ABS_TARGET}\" && ./scripts/new_project_cleanup.sh \"${PROJECT_SLUG}\""
echo "Optionally set up git in the new repo: make new-project-git-setup"
