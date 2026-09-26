# Zero `monkeypatch` in tests

**Status:** in progress. Execute stage by stage (`/impag`).

## Contract (user-ruled 2026-09-26)

- **Done** = `grep -rn monkeypatch tests/` returns 0 hits; every existing assertion unchanged and green; `make pyright` clean.
- **Behaviour-preserving:** production defaults stay the real calls. No new coverage, no behaviour change.
- **Seam = per-function defaulted keyword params** (existing style: `latest_release_zip(fetch=fetch_json)`, `uninstall(..., reg=None)`) plus dataclass/`__init__` fields on classes (`Installer`, `SetupWindow`). **No** single `Host`/ports object.
- **Tests build their fakes in pytest fixtures** and pass them in as arguments — never `monkeypatch`, `mock.patch`, or direct attribute assignment on modules/globals.
- Env vars (`setenv`) → pass an explicit env mapping / path param into the code under test, or run the subprocess with an explicit `env=`.

## Stages

1. **`tests/unit/test_setup_gui.py`** (46 hits) against `installer/setup_gui.py`. Seams needed: `subprocess.Popen`/`subprocess.run`, `tempfile` dir, `sys.platform` (also `backend.gui.default_app_paths`), `Path.rename`, `shutil.copytree`, `shutil.rmtree`, `os.rename`, `app_in_use`, `run_process`, `seed_model_cache`, `expected_model_size`, `_winreg`, `running_exe`, `schedule_self_delete`, `uninstall`, `tk.Tk`, `tkinter.messagebox.*`, root-logger handlers.
2. **`tests/unit/test_diarize_pyannote_runner.py`** (22 hits) against its runner module.
3. **`tests/unit/test_variants.py:212`** (`executor_mod.transcribe_with_baseline_params`) + **`tests/integration/test_cli_commands.py:71`** (`XDG_DATA_HOME`).
4. **Guard + verify + review:** add a local `pygrep` pre-commit hook failing on `monkeypatch` under `tests/` (red demo: re-add one use, see it fail, restore); `pytest tests/unit tests/integration -q --no-cov`; `make pyright`; code review over the plan's range.
