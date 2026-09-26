# Installer — visible uninstall progress

Status: open. Staged from the 2026-09-26 Windows eyeball run. Implement in a fresh session.

## Problem (user report)

Uninstall from the setup window closed the window at once and showed nothing until the final "Transcribe was
removed." box, so it looked as if the uninstall had crashed.

## Where (`installer/setup_gui.py`)

- `SetupWindow.uninstall`: after confirming, spawns `setup --uninstall --yes` and calls `root.destroy()` immediately.
- `uninstall_main`: `root.withdraw()`, hops to a `%TEMP%` copy (`relaunch_from_temp`: a second onefile start),
  runs `uninstall()` on the UI thread with no window, then shows `showinfo` / `showerror`.
- Silent gaps: the onefile unpack of the temp copy (seconds), then the whole `uninstall()` (venv rmtree, minutes).

## Moves

- The process that runs `uninstall()` shows a small "Uninstalling Transcribe" window: status line + indeterminate
  bar, `remove()` on a worker thread, per-step text (app files / settings / shortcuts / Apps & features entry).
- The window turns into the result ("Transcribe was removed." or the InstallError text) with a Close button;
  it replaces the separate `showinfo` / `showerror` boxes.
- Cover the hop gap with a `--ready-file` marker: the remover touches it once its window is up; the setup window
  (and, on the Start-menu / Apps & features path, the hopping install-dir copy) shows "Starting uninstall…" until then (20 s cap).
- X is ignored while removing; headless uninstall keeps its current path.
- Falsifier: unit tests on the step reporting and result state with injected `remove`; Windows eyeball: no blank gap
  longer than ~2 s from clicking Uninstall to the result.
- Owed (review of b64b3b6): `UninstallWindow` starts `remove` only once the waiter consumed the marker + ~1 s (cap ~5 s),
  then unlinks it; the waiter may run from the install dir. Also: disable `launch_button` while waiting; test `--ready-file` parsing.
