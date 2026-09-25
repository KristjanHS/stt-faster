# SmartScreen + code signing — plan

**Status:** A shipped; B + C are owner actions (no code), and C gates D; E is folded into the owed fresh-Windows smoke test. Parent: `2026-09-25-windows-installer-gui-design.md` (slice 7).

## Rulings (owner, 2026-09-25)

| Topic | Ruling |
|---|---|
| Now | Tell users how to click through, and submit every rebuilt exe to Microsoft. |
| Signing | **SignPath Foundation** (free for OSS; publisher shows "SignPath Foundation"). No paid OV/EV certificate. |
| Build | The exe gets built + signed on a `windows-latest` Actions job (free on a public repo). This replaces the local `build_installer.bat` build for releases. |
| Store / MSIX | Backlog; reconsider after D. |

## A. Click-through instructions + submit reminder — shipped

The README has a *Windows app* section with the two warnings (browser *Keep anyway*, then SmartScreen *Run anyway*). `release.sh` `NOTES` carries the same text and prints the Microsoft submission link when it attached a rebuilt `dist/` exe. The v1.1.0 release page carries the same text (`gh release edit`). Guard: `tests/unit/test_release_script.py` (a re-attached exe prints no reminder).

## B. Owner, now (no code)

- [ ] Submit the **current v1.1.0** `Transcribe-Setup.exe` (it predates the reminder) at https://www.microsoft.com/en-us/wdsi/filesubmission → *Software developer* → *Microsoft Defender SmartScreen* → *Incorrectly detected*. Microsoft reviews it; the warning may clear, but that isn't guaranteed.
- [ ] Optional: two screenshots (Edge *Keep anyway*, SmartScreen *Run anyway*) → `docs/img/`, linked from the README section. They help non-IT users most.

## C. SignPath Foundation application

Check the current terms on signpath.org first; the items below reflect my information and may have changed.

- [ ] **Repo (code, one commit):** add `## Code signing policy` to the README:
  - "Free code signing provided by SignPath.io, certificate by SignPath Foundation."
  - Roles: committers/reviewers and approvers = owner.
  - Privacy line: the installer contacts only GitHub (uv + app source), Hugging Face (models) and the ffmpeg zip host, and the app sends nothing anywhere.
- [ ] **Owner:** turn on MFA for GitHub and for the SignPath account.
- [ ] **Owner:** apply at signpath.org. List every pinned download (uv `0.12.19`, GyanD ffmpeg `9.0.2` essentials, HF model repos, the release source zip), because reviewers may question an installer that downloads and runs third-party binaries.
- [ ] **Owner, after approval:** add repo secret `SIGNPATH_API_TOKEN` and variables `SIGNPATH_ORG_ID`, `SIGNPATH_PROJECT_SLUG`, `SIGNPATH_POLICY_SLUG`.

## D. Sign in CI (after approval, one `/qimpag` session)

- [ ] `.github/workflows/release-installer.yml`, run `on: release: types: [published]` on `windows-latest`:
  1. Build the exe using the same uv + PyInstaller pins as `installer/build_installer.bat`. Reuse the bat if it runs non-interactively; otherwise single-source the pins.
  2. `actions/upload-artifact`.
  3. `signpath/github-action-submit-signing-request` (`wait-for-completion: true`, `output-artifact-directory`).
  4. Assert `(Get-AuthenticodeSignature <exe>).Status -eq 'Valid'`.
  5. `gh release upload <tag> <exe> --clobber`.
- [ ] `scripts/release.sh`: stop attaching an exe. Drop the `dist/` / `gh release download` branch, `SUBMIT_URL` and the reminder. Rewrite the matching tests: the no-exe preflight test and both happy-path exe assertions.
- [ ] Remove the "not code-signed yet … warns twice" text from the README and `NOTES`. Keep a one-line fallback ("if Windows still warns: More info → Run anyway") until the signed exe has built reputation.
- [ ] Update `2026-09-25-windows-installer-gui-design.md`: the *Build / release* ruling (no more local build for releases) and the slice-4 wording on attaching `dist/`.
- **Falsifier:** once, run the job with the signing step skipped and confirm the Authenticode assert fails the job.
- **Done when:** a fresh download from `releases/latest/download/Transcribe-Setup.exe` on a Windows box shows publisher *SignPath Foundation* in Properties → Digital Signatures.
- **Open:** the link returns the old asset (or 404) for the few minutes between publish and upload. Accept that, or keep re-attaching the previous exe and let the job overwrite it with `--clobber` (proposed: accept).

## E. Fresh-Windows smoke test (fold into the owed smoke test)

- [ ] Check Smart App Control (Windows Security → App & browser control → Smart App Control). If it is **On**, it blocks unsigned exes and offers no Run-anyway. Record whether the unsigned installer runs, and whether `uv.exe`, the uv-installed Python or any wheel DLL gets blocked. If they are blocked, signing only the installer is not enough; widen D's scope or document the limit.

## Backlog

- Microsoft Store channel (MSIX, Store-signed, no SmartScreen at all). The blocker to size: MSIX's packaged-app container vs an installer that builds a uv venv and downloads models at install time.
