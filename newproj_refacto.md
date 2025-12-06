# Plan: Hardening new-project automation

## Objectives
- Remove Makefile heredoc tab issues by invoking a standalone script for git setup (Better).
- Make replacements safer using Python (Best) and narrow to text files only (Best).
- Rewrite pyproject using structured TOML edits instead of regex (Better).
- Remove git prompts from cleanup; keep git setup solely in post step (Best).
- Limit replacement scope to text files and skip binaries/artifacts (Best).

## Steps
1) Move git setup out of Makefile heredoc into a callable script (existing `scripts/new_project_git_setup.py`), update Makefile to call it.
2) In `new_project_cleanup.sh`, switch replacement to a Python helper that:
   - Reads list of text files (whitelist extensions + MIME check).
   - Performs safe slug/module replacements via Python `str.replace`.
3) Replace pyproject mutations with a Python TOML rewrite that:
   - Updates name/description/homepage/repo
   - Sets runtime deps to baseline
   - Updates package include and coverage/source
4) Remove git init/remote prompts from cleanup; leave git init/origin/branch to `new-project-post`.
5) Ensure `.bootstrap_*` artifacts are skipped/cleaned appropriately.

## Validation
- Run a dry bootstrap+cleanup in isolated HOME to confirm replacements and pyproject rewrite.
- Verify Makefile target `new-project-post` calls the standalone git setup script.
