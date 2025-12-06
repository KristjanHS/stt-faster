#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/new_project_cleanup.sh [project-slug] [--module-name <python_name>] [--force]

Removes app-specific code and leaves a minimal scaffold for a new project.
- project-slug: new repo/app slug; if omitted, derived from .bootstrap_slug or directory name
- module-name: Python package name (defaults to project-slug with '-' -> '_')
- --force skips the confirmation prompt
EOF
}

PROJECT_SLUG=""
MODULE_NAME=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --module-name)
            MODULE_NAME=${2:-}
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "${PROJECT_SLUG}" ]]; then
                PROJECT_SLUG="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found." >&2
    exit 1
fi
if [[ -z "${PROJECT_SLUG}" ]]; then
    if [[ -f "${ROOT_DIR}/.bootstrap_slug" ]]; then
        PROJECT_SLUG="$(cat "${ROOT_DIR}/.bootstrap_slug")"
    else
        PROJECT_SLUG="$(basename "${ROOT_DIR}")"
    fi
fi
if [[ -z "${MODULE_NAME}" ]]; then
    MODULE_NAME="${PROJECT_SLUG//-/_}"
fi
PYPROJECT_PATH="${ROOT_DIR}/pyproject.toml"
ORIGINAL_SLUG="$(
    if [[ -f "${PYPROJECT_PATH}" ]]; then
        "${PYTHON_BIN}" - <<PY 2>/dev/null || true
import tomllib, pathlib
data = tomllib.loads(pathlib.Path("${PYPROJECT_PATH}").read_text())
print(data.get("project", {}).get("name", ""))
PY
    fi
)"
if [[ -z "${ORIGINAL_SLUG}" ]]; then
    ORIGINAL_SLUG="$(basename "${ROOT_DIR}")"
fi
ORIGINAL_MODULE="${ORIGINAL_SLUG//-/_}"

echo "Derived values for cleanup:"
echo " - root: ${ROOT_DIR}"
echo " - original slug/module: ${ORIGINAL_SLUG} / ${ORIGINAL_MODULE}"
echo " - new slug/module: ${PROJECT_SLUG} / ${MODULE_NAME}"

if [[ ${FORCE} -ne 1 ]]; then
    read -r -p "Remove app-specific code under ${ROOT_DIR}? [y/N] " reply
    if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

remove_path() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        rm -rf "${path}"
        echo "Removed ${path}"
    fi
}

declare -a TO_REMOVE=(
    "${ROOT_DIR}/backend"
    "${ROOT_DIR}/frontend"
    "${ROOT_DIR}/tests"
    "${ROOT_DIR}/reports"
    "${ROOT_DIR}/logs"
    "${ROOT_DIR}/${ORIGINAL_MODULE}.egg-info"
    "${ROOT_DIR}/transcribe_state.db"
    "${ROOT_DIR}/semgrep_local.sarif"
    "${ROOT_DIR}/tests/test.mp3"
    "${ROOT_DIR}/scripts/transcribe_manager.py"
    "${ROOT_DIR}/scripts/validate_estonian_models.py"
    "${ROOT_DIR}/scripts/check_gpu.py"
    "${ROOT_DIR}/scripts/transcribe-docker"
    "${ROOT_DIR}/scripts/transcription"
    "${ROOT_DIR}/scripts/windows"
    "${ROOT_DIR}/docs/Transcription_solution.md"
    "${ROOT_DIR}/docs/Transcription_helpfiles_win"
)

echo "The following paths will be deleted:"
for path in "${TO_REMOVE[@]}"; do
    echo " - ${path}"
done
if [[ ${FORCE} -eq 1 ]]; then
    echo "--force supplied; proceeding without additional confirmation."
else
    read -r -p "Proceed with deletion? [y/N] " delete_reply
    if [[ ! "${delete_reply}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

for path in "${TO_REMOVE[@]}"; do
    remove_path "${path}"
done

cat <<'EOF'
Manual steps required (automation removed for safety):
- Create your package scaffold (e.g., mkdir -p <module>/ and add __init__.py/cli stubs).
- Create your tests scaffold under tests/unit, tests/integration, tests/e2e.
- Update pyproject.toml manually: name/description/homepage/repository, runtime deps baseline, coverage/tool sources.
- Search/replace old slug/module strings manually (e.g., rg "stt-faster" ., rg "stt_faster" .).
- Set up git manually (git init, git remote add origin <url>, git branch -M main), or run make new-project-git-setup for a guided helper.
EOF

if [[ -f "${ROOT_DIR}/.bootstrap_slug" ]]; then
    rm -f "${ROOT_DIR}/.bootstrap_slug"
fi

echo "Cleanup complete. Review pyproject.toml, Makefile, docker configs, and README for remaining project-specific details."
