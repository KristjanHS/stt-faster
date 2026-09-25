#!/usr/bin/env bash
# Cut a GitHub release: bump version, tag, push main + tag, attach Transcribe-Setup.exe.
# Usage: make release V=X.Y.Z   (or scripts/release.sh X.Y.Z)

set -euo pipefail

EXE_NAME="Transcribe-Setup.exe"
NOTES="Download **${EXE_NAME}** and run it — no admin rights needed. The app is not code-signed yet, so Windows warns twice:
1. Browser says the file *isn't commonly downloaded*: click **Keep** (Edge: **… → Keep → Show more → Keep anyway**).
2. *Windows protected your PC*: click **More info → Run anyway**."
SUBMIT_URL="https://www.microsoft.com/en-us/wdsi/filesubmission"

die() {
    echo "release: $*" >&2
    exit 1
}

version="${1:-}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "usage: make release V=X.Y.Z (got '${version}')"
tag="v${version}"

root="$(git rev-parse --show-toplevel)"
cd "$root"

# --- Preflight: nothing below mutates the repo until every check passes ---
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || die "not on main"
[[ -z "$(git status --porcelain)" ]] || die "working tree not clean"
if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    die "tag ${tag} already exists locally"
fi
rc=0
git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null || rc=$?
[[ "$rc" -ne 0 ]] || die "tag ${tag} already on origin"
[[ "$rc" -eq 2 ]] || die "origin unreachable (git ls-remote rc=${rc})"
git fetch -q origin main
[[ "$(git rev-list --count HEAD..origin/main)" == "0" ]] || die "main is behind origin/main — pull first"
gh auth status >/dev/null 2>&1 || die "gh is not logged in — run: gh auth login"

tmp=""
hint=""
cleanup() {
    local status=$?
    [[ -z "$tmp" ]] || rm -rf "$tmp"
    [[ "$status" -eq 0 || -z "$hint" ]] || echo "release: stopped part-way — to recover: ${hint}" >&2
}
trap cleanup EXIT

rebuilt=""
if [[ -f "dist/${EXE_NAME}" ]]; then
    exe="dist/${EXE_NAME}"
    rebuilt=1
    echo "release: attaching rebuilt ${exe}"
else
    tmp="$(mktemp -d)"
    gh release download --pattern "$EXE_NAME" --dir "$tmp" ||
        die "no dist/${EXE_NAME} and none on the latest release — build it with installer/build_installer.bat"
    exe="${tmp}/${EXE_NAME}"
    echo "release: re-attaching ${EXE_NAME} from the latest release"
fi

# --- Bump the project version (skipped when a previous run already did it) ---
current="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
if [[ "$current" != "$version" ]]; then
    hint="git checkout -- pyproject.toml uv.lock, then re-run make release V=${version}"
    sed -i "0,/^version = \".*\"\$/s//version = \"${version}\"/" pyproject.toml
    uv lock
    git commit -m "chore(release): ${tag}" -- pyproject.toml uv.lock
fi

hint="git tag -d ${tag}, then re-run make release V=${version} (the version commit is kept)"
git tag -a "$tag" -m "$tag"
git push --atomic origin main "$tag"
hint="gh release create ${tag} <path to ${EXE_NAME}> --title ${tag} --generate-notes --latest --verify-tag"
gh release create "$tag" "$exe" --title "$tag" --generate-notes --notes "$NOTES" --latest --verify-tag
hint=""
echo "release: ${tag} published"
# A re-attached exe keeps its hash, so only a rebuild needs a new SmartScreen submission.
[[ -z "$rebuilt" ]] || echo "release: new exe hash — submit ${EXE_NAME} to ${SUBMIT_URL} (Software developer → Microsoft Defender SmartScreen → Incorrectly detected)"
