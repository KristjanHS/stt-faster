#!/usr/bin/env bash
# Emit the active uv extras variant ("cpu" or "cu130") for this checkout.
#
# Reads <repo>/.stt-variant.local (gitignored); a linked worktree without one
# inherits the main checkout's. No file anywhere => "cpu". Anything other than
# "cpu" or "cu130" is rejected with a clear error so a typo doesn't silently
# fall back to the default.
#
# Wrappers (run_uv.sh, Makefile targets) call this and pass `--extra ${variant}`
# to `uv sync`. See docs/plans/archived/2026-05-21-cuda-deps-cpu-gpu-extras-design.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VARIANT_FILE="$REPO_ROOT/.stt-variant.local"

if [ ! -f "$VARIANT_FILE" ]; then
    # --git-common-dir is <main checkout>/.git from any linked worktree.
    COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [ -n "$COMMON_DIR" ] && [ -f "$(dirname "$COMMON_DIR")/.stt-variant.local" ]; then
        VARIANT_FILE="$(dirname "$COMMON_DIR")/.stt-variant.local"
    fi
fi

if [ -f "$VARIANT_FILE" ]; then
    VARIANT="$(tr -d '[:space:]' < "$VARIANT_FILE")"
else
    VARIANT="cpu"
fi

case "$VARIANT" in
    cpu|cu130)
        printf '%s\n' "$VARIANT"
        ;;
    "")
        printf '%s\n' "cpu"
        ;;
    *)
        printf "unknown variant '%s' in .stt-variant.local -- expected 'cpu' or 'cu130'\n" "$VARIANT" >&2
        exit 1
        ;;
esac
