# Audio Transcription Automation - Quick Start

## Overview
Alternative 1 (Basic SQLite Tracker) has been implemented. This provides a simple, reliable way to process audio files in batch with state tracking.

## File Structure
```
scripts/
├── transcribe_manager.py           # CLI entry point
├── transcription/
│   ├── __init__.py
│   ├── database.py                 # SQLite state tracking
│   └── processor.py                # File processing logic
tests/unit/
├── test_transcribe_database.py    # Database tests
└── test_transcribe_processor.py   # Processor tests
```

## Usage

### 1. Process a folder of audio files
```bash
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe
```

Options:
- `--preset turbo|distil|large8gb` - Choose model preset (default: distil)
- `--db-path PATH` - Specify database location (default: transcribe_state.db)

### 2. Check processing status
```bash
# Basic summary
.venv/bin/python scripts/transcribe_manager.py status

# Detailed file list with errors
.venv/bin/python scripts/transcribe_manager.py status --verbose
```

## How It Works

1. **Scan**: Finds all audio files (.wav, .mp3, .m4a, .flac, .ogg, .wma) in the input folder
2. **Register**: Adds new files to SQLite database with 'pending' status
3. **Process**: Transcribes each pending file using `backend/transcribe.py`
4. **Archive**: Moves processed files to `processed/` or `failed/` subfolders
5. **Track**: Updates database with completion status and errors

## File Organization

After processing, your folder will look like:
```
/mnt/c/Users/PC/Downloads/transcribe/
├── processed/                   # Successfully transcribed
│   ├── audio1.wav
│   ├── audio1.json
│   ├── audio2.mp3
│   └── audio2.json
├── failed/                      # Failed files
│   └── corrupted_audio.wav
└── [any new files to process]
```

## Database Schema

The SQLite database (`transcribe_state.db`) tracks:
- `file_path` - Full path to audio file
- `status` - 'pending', 'completed', or 'failed'
- `error_message` - Error details for failed files

## Testing

### Run unit tests
```bash
# Test database operations
.venv/bin/python -m pytest tests/unit/test_transcribe_database.py -v

# Test processor logic
.venv/bin/python -m pytest tests/unit/test_transcribe_processor.py -v

# Run all transcription tests
.venv/bin/python -m pytest tests/unit/test_transcribe*.py -v
```

### Manual test with sample files
```bash
# Create a test folder
mkdir -p /tmp/test_transcribe
# Add some audio files
# Then process
.venv/bin/python scripts/transcribe_manager.py process /tmp/test_transcribe
```

## Troubleshooting

### "Input folder does not exist"
- Verify the path is correct
- For Windows folders in WSL2, use `/mnt/c/Users/...`
- Check folder permissions

### "No files tracked in database"
- No audio files found in the specified folder
- Check that files have supported extensions (.wav, .mp3, etc.)
- Files in subfolders are ignored (only root level files are processed)

### "Failed to process"
- Check the error message with `--verbose` flag
- Verify GPU/CUDA is working if using GPU presets
- Try a smaller preset (e.g., `--preset distil`)

## Next Steps

If you need more features, see `docs/transcription_automation_plan.md` for:
- **Alternative 2**: Automatic retry logic, concurrent processing, timestamps
- **Alternative 3**: Production-grade with config files, migrations, comprehensive error handling

## Re-running Failed Files

Simply run the process command again:
```bash
.venv/bin/python scripts/transcribe_manager.py process /mnt/c/Users/PC/Downloads/transcribe
```

The system will:
1. Skip already completed files
2. Find any new files added to the folder
3. You can manually reset failed files in the database if needed

## Database Management

### View database contents
```bash
sqlite3 transcribe_state.db "SELECT * FROM transcriptions;"
```

### Reset a file's status
```bash
sqlite3 transcribe_state.db "UPDATE transcriptions SET status='pending' WHERE file_path='/path/to/file.wav';"
```

### Clear database (start fresh)
```bash
rm transcribe_state.db
```
