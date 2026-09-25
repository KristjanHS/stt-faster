#!/usr/bin/env bash
# Cut a GitHub release: bump version, tag, push main + tag, attach Transcribe-Setup.exe.
# Usage: make release V=X.Y.Z   (or scripts/release.sh X.Y.Z)

set -euo pipefail

EXE_NAME="Transcribe-Setup.exe"
NOTES="Download **${EXE_NAME}** and run it — no admin rights needed.
Windows SmartScreen warns about unsigned apps: click **More info → Run anyway**."

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
[[ "$rc" -eq 2 ]] || die "tag ${tag} already on origin (or origin unreachable, rc=${rc})"

if [[ -f "dist/${EXE_NAME}" ]]; then
    exe="dist/${EXE_NAME}"
    echo "release: attaching rebuilt ${exe}"
else
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    gh release download --pattern "$EXE_NAME" --dir "$tmp" ||
        die "no dist/${EXE_NAME} and none on the latest release — build it with installer/build_installer.bat"
    exe="${tmp}/${EXE_NAME}"
    echo "release: re-attaching ${EXE_NAME} from the latest release"
fi

# --- Bump the project version (skipped when a previous run already did it) ---
current="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
if [[ "$current" != "$version" ]]; then
    sed -i "0,/^version = \".*\"\$/s//version = \"${version}\"/" pyproject.toml
    uv lock
    git commit -m "chore(release): ${tag}" -- pyproject.toml uv.lock
fi

git tag -a "$tag" -m "$tag"
git push origin main
git push origin "$tag"
gh release create "$tag" "$exe" --title "$tag" --generate-notes --notes "$NOTES" --latest --verify-tag
echo "release: ${tag} published"
