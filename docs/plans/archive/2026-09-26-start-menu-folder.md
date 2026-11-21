# Installer — Start-menu folder with tools

Status: ✅ SHIPPED 2026-09-26. Owed: Windows eyeball of Start ▸ All apps (project_state OPEN).

## Target

```
Programs\Transcribe\            (folder; Desktop keeps a single Transcribe.lnk)
  Transcribe.lnk            -> .venv\Scripts\stt-faster-gui.exe
  Repair Transcribe.lnk     -> setup_copy (no args: window opens on "Repair / update")
  Uninstall Transcribe.lnk  -> setup_copy --uninstall
  Setup log.lnk             -> logs\setup.log
```

## Moves

- `shortcut_script` / `remove_shortcuts_script` (`installer/setup_gui.py`): Desktop gets one link; Programs gets the
  folder. Tool links only when `setup_copy` exists (dev runs have none).
- Remove the old top-level `Programs\Transcribe.lnk` on install (updates from ≤1.2.x) and on uninstall.
- Uninstall removes the whole `Programs\Transcribe` folder.
- `Transcribe.lnk` WorkingDirectory = install dir, not `Scripts` (matches `launch_gui`'s never-cwd-in-venv rule).
- Falsifier: unit tests on the generated PowerShell (paths, args, old-link removal); Windows eyeball of Start ▸ All apps.
