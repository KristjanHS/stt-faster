#!/usr/bin/env bash
# Decide whether the release-installer workflow rebuilds Transcribe-Setup.exe or reuses the newest
# earlier release's exe. A rebuild changes the file hash, which resets SmartScreen reputation, so the
# exe is rebuilt only when installer/ changed (the exe fetches the latest release's source at install
# time, so app changes need no rebuild).
# Usage: scripts/installer_reuse.sh <tag> <out_dir>
# Writes REBUILD=true|false and REUSE_FROM=<tag> to $GITHUB_ENV (stdout when unset); on reuse the
# earlier exe is downloaded into <out_dir>. FORCE_REBUILD=true skips the check.

set -euo pipefail

EXE_NAME="Transcribe-Setup.exe"
tag="${1:?usage: installer_reuse.sh <tag> <out_dir>}"
out_dir="${2:?usage: installer_reuse.sh <tag> <out_dir>}"
env_file="${GITHUB_ENV:-/dev/stdout}"

decide() {
    echo "installer_reuse: $2"
    printf 'REBUILD=%s\nREUSE_FROM=%s\n' "$1" "${3:-}" >>"$env_file"
    exit 0
}

[[ "${FORCE_REBUILD:-false}" != "true" ]] || decide true "rebuild forced"

releases="$(gh release list --exclude-drafts --limit 50 --json tagName --jq '.[].tagName')"
prev=""
while read -r t; do
    [[ -n "$t" && "$t" != "$tag" ]] || continue
    if gh release view "$t" --json assets --jq '.assets[].name' | grep -qxF "$EXE_NAME"; then
        prev="$t"
        break
    fi
done <<<"$releases"
[[ -n "$prev" ]] || decide true "rebuild: no earlier release carries ${EXE_NAME}"

rc=0
git diff --quiet "$prev" "$tag" -- installer/ || rc=$?
case "$rc" in
    0) ;;
    1) decide true "rebuild: installer/ changed between ${prev} and ${tag}" ;;
    *)
        echo "installer_reuse: git diff ${prev} ${tag} failed (rc ${rc}) — are both tags fetched?" >&2
        exit 1
        ;;
esac

gh release download "$prev" --pattern "$EXE_NAME" --dir "$out_dir" --clobber
decide false "reuse: installer/ unchanged since ${prev}" "$prev"
