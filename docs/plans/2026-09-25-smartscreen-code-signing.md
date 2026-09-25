# SmartScreen + code signing — plan

**Status:** A shipped; C's repo commit + D's CI build shipped with signing dormant; left: B/C owner steps, D's after-approval steps, E. Parent: `2026-09-25-windows-installer-gui-design.md` (slice 7).

## Rulings (owner, 2026-09-25)

| Topic | Ruling |
|---|---|
| Now | Tell users how to click through, and submit every rebuilt exe to Microsoft. |
| Signing | **SignPath Foundation** (free for OSS; publisher shows "SignPath Foundation"). No paid OV/EV certificate. |
| Build | The exe gets built + signed on a `windows-latest` Actions job (free on a public repo). This replaces the local `build_installer.bat` build for releases. |
| Store / MSIX | Backlog; reconsider after D. |
| CI build before approval | Move the build to CI now; SignPath steps run only once `vars.SIGNPATH_ORG_ID` is set; the Microsoft submission reminder moves to the job summary while dormant. |
| Rebuild trigger | CI rebuilds the exe only when `installer/` changed since the newest earlier release carrying it; otherwise it re-attaches that exe. Every rebuild changes the hash, which resets SmartScreen reputation and needs a new Microsoft submission while unsigned. |
| Privacy text | SignPath's verbatim sentence + the host list (the app downloads models at runtime, so "sends nothing" was wrong). |

## A. Click-through instructions + submit reminder — shipped

The README has a *Windows app* section with the two warnings (browser *Keep anyway*, then SmartScreen *Run anyway*). `release.sh` `NOTES` carries the same text and prints the Microsoft submission link when it attached a rebuilt `dist/` exe. The v1.1.0 release page carries the same text (`gh release edit`). Guard: `tests/unit/test_release_script.py` (a re-attached exe prints no reminder).

## B. Owner, now (no code)

- [ ] Submit the **current v1.1.0** `Transcribe-Setup.exe` (it predates the reminder) at https://www.microsoft.com/en-us/wdsi/filesubmission → *Software developer* → *Microsoft Defender SmartScreen* → *Incorrectly detected*. Microsoft reviews it; the warning may clear, but that isn't guaranteed.
- [ ] Optional: two screenshots (Edge *Keep anyway*, SmartScreen *Run anyway*) → `docs/img/`, linked from the README section. They help non-IT users most.

## C. SignPath Foundation application

Check the current terms on signpath.org first; the items below reflect my information and may have changed.

- [x] **Repo:** README `## Code signing policy` (attribution, roles, privacy sentence + hosts) and the `LICENSE` holder.
- [ ] **Owner:** turn on MFA for GitHub and for the SignPath account.
- [ ] **Owner:** apply at signpath.org. List every pinned download (uv `0.12.19`, GyanD ffmpeg `9.0.2` essentials, HF model repos, the release source zip), because reviewers may question an installer that downloads and runs third-party binaries.
- [ ] **Owner, SignPath project:** artifact configuration with a `<zip-file>` root (upload-artifact zips the exe); signing policy must allow `refs/tags/v*`.
- [ ] **Owner, after approval:** add repo secret `SIGNPATH_API_TOKEN` and variables `SIGNPATH_ORG_ID`, `SIGNPATH_PROJECT_SLUG`, `SIGNPATH_POLICY_SLUG`.

## D. Build + sign in CI

- [x] `.github/workflows/release-installer.yml` (`release: published` + `workflow_dispatch tag`): `build_installer.bat < NUL` → upload-artifact → SignPath v3 + Authenticode assert (only if `vars.SIGNPATH_ORG_ID`) → `gh release upload --clobber`.
- [x] Rebuild gate: `scripts/installer_reuse.sh`, run from the workflow's own commit so older-tag dry-runs work (`git diff --quiet <prev> <tag> -- installer/`) → re-attach the earlier exe, or rebuild. Also rebuilds on `rebuild`/`skip_signing` inputs, when no earlier release has the exe, and when signing is live but the reused exe isn't Valid-signed. Guard: `tests/unit/test_installer_reuse.py`.
- [x] `scripts/release.sh` attaches no exe; tests rewritten; design doc updated. The publish→upload gap is accepted.
- [ ] **Owner, before the next release:** dry-run the untested bat in CI: `gh workflow run release-installer.yml -f tag=v1.1.0 -f rebuild=true -f attach=false` must go green.
- [ ] **After approval:** set the SignPath secret + vars, then run the falsifier: `gh workflow run release-installer.yml -f tag=<tag> -f skip_signing=true` must fail at the Authenticode assert.
- [ ] **After the first signed release:** replace the "not code-signed yet … warns twice" text in the README and `NOTES` with a one-line fallback ("if Windows still warns: More info → Run anyway").
- **Done when:** a fresh download from `releases/latest/download/Transcribe-Setup.exe` on a Windows box shows publisher *SignPath Foundation* in Properties → Digital Signatures.

## E. Fresh-Windows smoke test (fold into the owed smoke test)

- [ ] Check Smart App Control (Windows Security → App & browser control → Smart App Control). If it is **On**, it blocks unsigned exes and offers no Run-anyway. Record whether the unsigned installer runs, and whether `uv.exe`, the uv-installed Python or any wheel DLL gets blocked. If they are blocked, signing only the installer is not enough; widen D's scope or document the limit.

## Backlog

- Microsoft Store channel (MSIX, Store-signed, no SmartScreen at all). The blocker to size: MSIX's packaged-app container vs an installer that builds a uv venv and downloads models at install time.
