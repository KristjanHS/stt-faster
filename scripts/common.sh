#!/usr/bin/env bash
# This file provides shared functionality for all scripts in the scripts/ directory

set -euo pipefail

# Get the project root directory (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project root
if [ ! -f "$PROJECT_ROOT/docker/docker-compose.yml" ]; then
    echo "Error: Please run scripts from the project root directory."
    exit 1
fi

# Change to project root for all operations
cd "$PROJECT_ROOT"

# Define all important paths relative to project root
export DOCKER_COMPOSE_FILE="docker/docker-compose.yml"
export BACKEND_DIR="backend"
export LOGS_DIR="logs"

# Docker service names
export APP_SERVICE="app"
export OLLAMA_SERVICE="ollama"

# Default values that can be overridden by environment variables
export DEFAULT_LOG_LEVEL="${DEFAULT_LOG_LEVEL:-INFO}"

## Removed unused helpers: validate_directories, get_relative_path, resolve_path

# --------------------------- Logging utilities ------------------------------

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# LOG_LEVEL controls verbosity (DEBUG, INFO, WARN, ERROR). Default: INFO
export LOG_LEVEL=${LOG_LEVEL:-INFO}

_log_level_to_num() {
    case "$1" in
        DEBUG) echo 10 ;;
        INFO)  echo 20 ;;
        WARN)  echo 30 ;;
        ERROR) echo 40 ;;
        *)     echo 20 ;;
    esac
}

# Print a single log line to stdout with optional color if TTY
log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts=$(date -Is)
    local threshold=$(_log_level_to_num "${LOG_LEVEL}")
    local nlevel=$(_log_level_to_num "$level")
    if [ "$nlevel" -lt "$threshold" ]; then
        return 0
    fi
    if [ -t 1 ]; then
        case "$level" in
            DEBUG) printf "%s [\033[36m%s\033[0m] %s\n" "$ts" "$level" "$msg" ;;
            INFO)  printf "%s [\033[32m%s\033[0m] %s\n" "$ts" "$level" "$msg" ;;
            WARN)  printf "%s [\033[33m%s\033[0m] %s\n" "$ts" "$level" "$msg" ;;
            ERROR) printf "%s [\033[31m%s\033[0m] %s\n" "$ts" "$level" "$msg" ;;
            *)     printf "%s [%s] %s\n" "$ts" "$level" "$msg" ;;
        esac
    else
        printf "%s [%s] %s\n" "$ts" "$level" "$msg"
    fi
}

## Removed unused wrapper: log_message

# Initialize script logging: make timestamped log and a stable symlink.
# Keeps the 5 most recent logs for a script prefix.
init_script_logging() {
    local script_name="$1"
    local stable_log_path="$LOGS_DIR/${script_name}.log"
    local keep_count=5

    mkdir -p "$LOGS_DIR"

    # Prune old logs (keep newest keep_count-1 files)
    find "$LOGS_DIR" -maxdepth 1 -type f -name "${script_name}[-_]*.log" -printf "%T@ %p\n" \
      | sort -nr \
      | tail -n +$keep_count \
      | cut -d' ' -f2- \
      | xargs -r rm --

    # Create timestamped log for this run
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    local new_log_file="$LOGS_DIR/${script_name}-${ts}.log"
    : > "$new_log_file"

    # Point stable symlink to new log
    ln -sf "$(basename "$new_log_file")" "$stable_log_path"

    echo "$new_log_file"
}

# Error trap: logs failing command and line number to both stdout and log file
enable_error_trap() {
    local log_file="$1"
    local script_name="$2"
    # Use a double-quoted trap string and escape $ so expansions happen at trap time.
    trap "rc=\$?; ln=\${BASH_LINENO[0]:-?}; cm=\${BASH_COMMAND:-?}; msg='${script_name}: line '"\$ln"': '"\$cm"' (exit '"\$rc"')'; log ERROR \"\$msg\" | tee -a \"${log_file}\"; exit \$rc" ERR
}

# Optional debug trace to the log file only (export DEBUG=1 to enable)
enable_debug_trace() {
    local log_file="$1"
    exec 3>>"$log_file"
    export BASH_XTRACEFD=3
    export PS4='+ $(date -Is) ${BASH_SOURCE##*/}:${LINENO}: '
    if [ "${DEBUG:-}" = "1" ]; then
        set -x
    fi
}

## Removed unused helper: run_step

## Removed unused helpers: get_script_name and legacy log_* wrappers
