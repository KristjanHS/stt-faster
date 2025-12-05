# Windows Batch Files for Transcription

One-click audio transcription for Windows + WSL users.

## Files

| File | Language | Model | Use Case |
|------|----------|-------|----------|
| `transcribe_estonian.bat` 🇪🇪 | Estonian | TalTech Whisper large-v3-turbo | Estonian speech |
| `transcribe_english.bat` 🇬🇧 | English/Multi | Whisper large-v3 | English & other languages |

## Usage

1. **Copy** the appropriate `.bat` file to your audio folder  
   Example: `C:\Users\YourName\Downloads\transcribe\`

2. **Add audio files** to the same folder  
   Supported: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wma`

3. **Double-click** the batch file

4. **Wait** for completion  
   First run: Downloads model (~1.5-3GB depending on preset)  
   Later runs: Uses cached model (fast)

5. **Find results** in `processed/` subfolder  
   - `filename.json` - Transcript with timestamps
   - `filename.wav` - Original audio file

## Example

```
Before:
  transcribe/
  ├── transcribe_estonian.bat
  ├── audio1.wav
  └── audio2.mp3

After:
  transcribe/
  ├── transcribe_estonian.bat
  ├── processed/
  │   ├── audio1.wav + audio1.json
  │   └── audio2.mp3 + audio2.json
  └── transcribe_state.db
```

## Requirements

- Windows 10/11 with WSL2
- Python virtual environment (see main README.md)
- Optional: CUDA for GPU acceleration

## Troubleshooting

**Processing is slow**  
→ First run downloads model; GPU needs CUDA; CPU is slower but works

**"Input folder does not exist"**  
→ WSL needs access to Windows folders via `/mnt/c/Users/...`

**Files stuck in pending**  
→ Check errors: `.venv/bin/python scripts/transcribe_manager.py status --verbose`  
→ Or delete `transcribe_state.db` to start fresh

## Customization

Edit the batch file to change paths or presets:

```batch
REM Original:
wsl -e bash -c "cd ~/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe"

REM Custom project path:
wsl -e bash -c "cd ~/YOUR/PROJECT/PATH && .venv/bin/python scripts/transcribe_manager.py process ..."

REM Custom audio folder:
... process /mnt/c/YOUR/PATH/HERE

REM Different preset:
... process /mnt/c/YOUR/PATH --preset turbo

REM Custom database:
... process /mnt/c/YOUR/PATH --db-path /mnt/c/YOUR/PATH/custom.db
```

## Model Details

| Preset | Model | Size | Language | Speed | Accuracy |
|--------|-------|------|----------|-------|----------|
| `et-large` | TalTech Whisper large-v3-turbo | ~1.5GB | Estonian | Fast | High (ET) |
| `large8gb` | Whisper large-v3 | ~3GB | Multi | Medium | Highest |
| `turbo` | Whisper large-v3-turbo | ~1.5GB | Multi | Fast | High |
| `distil` | Distilled Whisper | ~756MB | English | Fastest | High |

**More info:** [README_TRANSCRIPTION.md](../../README_TRANSCRIPTION.md)
