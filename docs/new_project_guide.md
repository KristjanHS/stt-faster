# How you can Reuse this repo as a template:

Scripts automate most of the cloning/cleanup so you can start a fresh project (e.g., `xls-analyser`) while keeping configs and automations. Assumes new repos live under `~/projects/`.

## Prereqs
- Run commands from the repo root.
- Tools: `rsync`, `rg` (ripgrep), Bash.
- Ensure the target path (e.g., `~/projects/xls-analyser`) is empty or missing.

## Steps
1) Copy this repo into `~/projects/<slug>` (from the source repo)  
`make new-project-bootstrap SLUG=xls-analyser`
- Script will show source/target and exclusions before copying; confirm to proceed.

2) Enter the new repo and remove app-specific code  
```
cd ~/projects/xls-analyser
make new-project-cleanup  # slug auto-detected from bootstrap; module defaults to slug->snake_case and cleanup runs non-interactively
```
- Removes app-specific code/tests/artifacts while keeping shared configs.
- Script will display derived slugs/modules and list deletions before you confirm.

3) Manual edits to align with the new project  
- Create your package/tests scaffold (e.g., module with `__init__.py`/`cli.py`; tests/unit|integration|e2e placeholders).
- Update `pyproject.toml`: project name/description/homepage/repository, runtime deps baseline (typer, pydantic, python-dotenv, rich), coverage/tool sources, and package discovery to point at your module.
- Search/replace old slug/module strings manually (e.g., `rg "xls-analyser" .`, `rg "xls_analyser" .`). If you prefer assistance, have an AI coding agent perform reviewed replacements instead of scripts.
- Set up git manually (`git init`, `git remote add origin <url>`, `git branch -M main`), or run `make new-project-git-setup` to call the helper script with prompts (init/remote/branch).
- Update README, `AGENTS.md`, docs references, badges, Docker names/env templates, and any remaining config that mentions the old project.

4) Rebootstrap tooling in the new repo  
```
# installs dev/test env; installs hooks; runs unit tests
make new-project-post
# Optional when network is available:
uv run pre-commit run --all-files
```

5) Optional checks  
- Run broader tests when you add code: `.venv/bin/python -m pytest tests/integration -q`.  
- If you edit secrets/env templates, rerun detect-secrets via pre-commit or `uv run pre-commit run detect-secrets --all-files`.
