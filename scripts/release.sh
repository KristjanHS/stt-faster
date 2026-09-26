#!/usr/bin/env bash
# Cut a GitHub release: bump version, tag, push main + tag, publish. The release-installer workflow attaches the exe.
# Usage: make release [BUMP=patch|minor|major] [V=X.Y.Z]   (or scripts/release.sh [patch|minor|major|X.Y.Z])

set -euo pipefail

EXE_NAME="Transcribe-Setup.exe"
KEEP_EXES=5 # releases (newest first, the new one included) that keep their exe; older ones lose it
NOTES="Download **${EXE_NAME}** and run it — no admin rights needed. The app is not code-signed, so Windows warns twice:
1. Browser says the file *isn't commonly downloaded*: click **Keep** (Edge: **… → Keep → Show more → Keep anyway**).
2. *Windows protected your PC*: click **More info → Run anyway**."

die() {
    echo "release: $*" >&2
    exit 1
}

bump="${1:-patch}"
root="$(git rev-parse --show-toplevel)"
cd "$root"

current="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
if [[ "$bump" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    version="$bump"
elif [[ "$bump" =~ ^(patch|minor|major)$ ]]; then
    [[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "pyproject.toml version '${current}' is not X.Y.Z"
    major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}"
    if ! git ls-remote --exit-code --tags origin "refs/tags/v${current}" >/dev/null; then
        version="$current" # an earlier run bumped but never tagged: finish that release
    elif [[ "$bump" == major ]]; then
        version="$((major + 1)).0.0"
    elif [[ "$bump" == minor ]]; then
        version="${major}.$((minor + 1)).0"
    else
        version="${major}.${minor}.$((patch + 1))"
    fi
else
    die "usage: make release [BUMP=patch|minor|major] [V=X.Y.Z] (got '${bump}')"
fi
tag="v${version}"

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

if [[ -t 0 ]]; then
    read -r -p "release: publish ${tag} (currently ${current})? [y/N] " answer
    [[ "$answer" == [yY] ]] || die "cancelled"
fi

hint=""
cleanup() {
    local status=$?
    [[ "$status" -eq 0 || -z "$hint" ]] || echo "release: stopped part-way — to recover: ${hint}" >&2
}
trap cleanup EXIT

# --- Bump the project version (skipped when a previous run already did it) ---
if [[ "$current" != "$version" ]]; then
    hint="git checkout -- pyproject.toml uv.lock, then re-run make release V=${version}"
    sed -i "0,/^version = \".*\"\$/s//version = \"${version}\"/" pyproject.toml
    uv lock
    git commit -m "chore(release): ${tag}" -- pyproject.toml uv.lock
fi

hint="git tag -d ${tag}, then re-run make release V=${version} (the version commit is kept)"
git tag -a "$tag" -m "$tag"
git push --atomic origin main "$tag"
hint="gh release create ${tag} --title ${tag} --generate-notes --latest --verify-tag"
gh release create "$tag" --title "$tag" --generate-notes --notes "$NOTES" --latest --verify-tag
hint=""

# --- Prune the exe from all but the newest KEEP_EXES releases (best-effort: the release is already out) ---
# The new release counts as one of them: CI attaches its exe, reusing the newest earlier one if installer/ is unchanged.
n=0
while read -r t; do
    [[ -n "$t" ]] || continue
    n=$((n + 1))
    ((n > KEEP_EXES)) || continue
    # not grep -q: its early exit SIGPIPEs gh, and pipefail then reads the match as a miss
    if gh release view "$t" --json assets --jq '.assets[].name' | grep -xF "$EXE_NAME" >/dev/null; then
        gh release delete-asset "$t" "$EXE_NAME" --yes >/dev/null && echo "release: removed ${EXE_NAME} from ${t}" ||
            echo "release: warning — could not remove ${EXE_NAME} from ${t}" >&2
    fi
done < <(gh release list --exclude-drafts --limit 100 --json tagName --jq '.[].tagName' ||
    echo "release: warning — could not list releases to prune old exes" >&2)

echo "release: ${tag} published — the release-installer workflow attaches ${EXE_NAME} (rebuilt only if installer/ changed)"
