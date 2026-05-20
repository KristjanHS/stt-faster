# Windows Batch Files for Transcription

One-click audio transcription for Windows + WSL users. Each `.bat` runs `scripts/transcribe_manager.py` under WSL with a preset, language, and one or more variant numbers baked into the file.

## Files

| File | Language | Default variant(s) | Tuned for |
|------|----------|--------------------|-----------|
| `transcribe_estonian_Desk.bat` | Estonian | 44 | Desk-mic / close-talk Estonian |
| `transcribe_estonian_Teams.bat` | Estonian | 61 | Teams / video-call Estonian |
| `transcribe_estonian_Online.bat` | Estonian | 53, 52 | Online / mixed-source Estonian (runs both variants) |
| `transcribe_estonian_32bit_cpu.bat` | Estonian | 7 (preset `et-32`) | CPU-only machines, no GPU |
| `transcribe_english_Desk.bat` | English | 44 | Desk-mic English (`turbo` preset) |
| `transcribe_english_Teams.bat` | English | 61 | Teams / video-call English (`turbo` preset) |
| `compare_variants.bat` | Estonian | all builtin variants | Comparison run across every variant in the registry; also generates the HTML report |
| `report_only.bat` | – | – | Regenerates the variant HTML report from existing outputs |
| `check_status.bat` | – | – | Prints the transcription queue status |

To change which variant a file uses, edit the `set "VARIANTS=..."` line at the top — space-separated for multiple (e.g. `set "VARIANTS=1 36 44"`).

## Usage

1. **Copy** the appropriate `.bat` file into the folder that holds your audio.
2. **Add audio files** to that same folder. Supported: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`.
3. **Double-click** the batch file.
4. **Wait** for completion. First run downloads the model (~1.5–3 GB) to `~/.cache/hf/` inside WSL; later runs hit the cache.
5. **Find results** next to the audio file:
   - Single-variant runs write `processed/filename.json` and `processed/filename.txt`.
   - Multi-variant runs create a timestamped folder (e.g. `2026-05-19T14-30-00/`) next to the audio, containing one `variant_NNN_<name>_<audio>.json` (and `.txt`) per variant plus a `run_meta.json` describing the run.
   - Failed inputs land in `failed/`.

## Requirements

- Windows 10/11 with WSL2
- Python venv at `~/projects/stt-faster/.venv/` (see top-level `README.md`)
- Optional: CUDA for GPU acceleration
- Models cached at `~/.cache/hf/` in WSL (auto-downloaded on first run)

## Troubleshooting

**Processing is slow**
→ First run downloads the model; GPU needs CUDA; CPU works but is slower.

**"Input folder does not exist"**
→ WSL needs access to the Windows folder via `/mnt/c/Users/...`. The batch files auto-convert `C:\`, `D:\`, `E:\`.

**Files stuck in pending**
→ Look in `<audio_folder>/failed/` for files that errored; check `logs/` for stack traces.
→ Inspect recent run history: `.venv/bin/python -m backend.cli.main db recent --limit 10`.

## Customisation

The core command line inside each `.bat` looks like:

```batch
wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process '<WSL_INPUT_DIR>' --language et --output-format both --variant 44"
```

Swap `--language et` for `--language en` (and add `--preset turbo`) for English, change `--variant` / `--variants`, or override `--preset` to one of `turbo`, `distil`, `large8gb`, `et-large`, `et-32`, `small`.

## Model presets

| Preset | Model | Size | Language | Notes |
|--------|-------|------|----------|-------|
| `et-large` (default for Estonian) | TalTechNLP/whisper-large-v3-turbo-et-verbatim | ~1.5 GB | Estonian | Best Estonian accuracy |
| `et-32` | Same model, float32 CPU | ~1.5 GB | Estonian | CPU-only |
| `turbo` | Systran/faster-distil-whisper-large-v3 | ~1.5 GB | Multi | Used by English `.bat`s |
| `large8gb` | Systran/faster-whisper-large-v3 | ~3 GB | Multi | Highest accuracy |
| `distil` | Distilled Whisper | ~756 MB | English | Fastest |

See `docs/Transcription_solution.md` for the full preset / variant matrix.
