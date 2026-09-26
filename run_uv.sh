#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install from https://astral.sh/uv" >&2
  exit 1
fi

# Resolve the per-machine uv extras variant (cpu | cu130) from .stt-variant.local
# via scripts/select_variant.sh. Selector validates the value and rejects typos.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="$("$SCRIPT_DIR/scripts/select_variant.sh")"

echo "[stt-faster] variant=${VARIANT} (set via 'make use-cpu' / 'make use-gpu' to switch)"

# Ensure a .venv exists and is seeded with pip; newer uv refuses to recreate
# an existing one, so a re-run keeps it and lets `uv sync` update it.
if [ ! -x .venv/bin/python ]; then
  uv venv --seed
fi

# Fallback: if pip still missing for any reason, try ensurepip
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade || true
fi

# Sync the test group with the selected extra, plus `gui` (tkinterdnd2) so the
# pre-commit pyright hook can resolve backend/gui.py's import.
# Pass-through args ($@) let callers add flags like --frozen or extra groups.
uv sync --extra "$VARIANT" --extra gui --group test "$@"

# Quick sanity print
uv run python -V
echo "Env ready. Use: 'uv run <cmd>' or '.venv/bin/<cmd>'."
