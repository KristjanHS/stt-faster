# Mixed-language (English + Japanese) transcription

Status: IN PROGRESS
Date: 2026-10-16

## Problem

A meeting recording that is mostly English but partly Japanese cannot be transcribed well today:

1. The `turbo` preset (GUI "English" profile) is `Systran/faster-distil-whisper-large-v3` —
   distil-whisper, **English-only** training data. Japanese comes out translated/garbled. The
   name is misleading (it is not Whisper large-v3-turbo) and `distil` is an exact duplicate.
2. The GUI forces `--language en`, so even a multilingual model translates Japanese into English.
3. With language unset, faster-whisper detects language **once** from the first 30 s
   (`multilingual=False` default), so an English-opening meeting decodes entirely as English.

## Model research (2026-10-16)

`kotoba-tech/kotoba-whisper-bilingual-v1.0-faster` — official CT2 conversion, MIT, distilled from
large-v3 for Japanese + English (2 decoder layers, ~6x faster). Authors' own benchmarks:

| Set | kotoba-bilingual | kotoba-v2.0 (JA-only) | large-v3 |
|---|---|---|---|
| CommonVoice 8 JA (CER) | 9.8 | 9.2 | **8.5** |
| JSUT basic5000 JA (CER) | 9.3 | 8.4 | **7.1** |
| ReazonSpeech JA (CER) | 16.8 | **11.6** | 14.9 |
| ESB AMI EN meetings (WER) | **16.7** | n/a (EN broken) | 17.9 |

No distillation beats large-v3 on Japanese overall; kotoba-bilingual trades a little Japanese
accuracy for speed with on-par English. Untested: whether its distilled decoder handles
per-segment language detection (`multilingual=True`) — its card only shows forced language.
→ Ship both, decide the GUI default by A/B on the user's real recording (Stage 4).

## Contract (questions round 2026-10-16)

- **Seam:** `--language multi` → `transcribe()` passes `language=None, multilingual=True`.
  The language string already flows GUI → CLI → run_config → processor → `transcribe()`
  untouched, so no new flag is plumbed. `et-*` presets keep the Estonian default; `multi`
  overrides it.
- **New preset** `ja-en-distil` = `kotoba-tech/kotoba-whisper-bilingual-v1.0-faster`
  (cuda/float16).
- **GUI profile** "Mixed (auto)" = `large8gb` + `--language multi` until the A/B says otherwise.
  Installer prefetches `Systran/faster-whisper-large-v3` for it.
- **Rename:** `turbo` + `distil` merge into `en-distil`. Old names stay accepted as aliases
  that log a "renamed" warning. All docs/.bat/tests/GUI/scripts say `en-distil`.
- **Out of scope:** a real large-v3-turbo preset, a JA-only kotoba-v2.0 preset, translation.

## Stages

### Stage 1 — rename `turbo`/`distil` → `en-distil` with aliases
Focus: `backend/model_config.py` (PRESETS, `PRESET_ALIASES`, `get_preset` warns + resolves),
`backend/transcribe.py` docstring, `backend/gui.py`, `scripts/transcribe_manager.py` choices,
`scripts/windows/*.bat` + README, `scripts/transcribe-docker`, `Makefile`, `README.md`,
`docker/*.md`, `docs/Transcription_solution.md`, `docs/stt_stages.md`,
`installer/setup_gui.py` comment, tests. Alias test: `get_preset("turbo")` returns the
`en-distil` config.

### Stage 2 — `--language multi`
Focus: `backend/transcribe.py` — resolve `multi` before `_apply_language_default`; pass
`multilingual=` to `model.transcribe()`; log "auto-detect per segment". CLI help text mentions
`multi`. Unit test through the `model_picker` seam: fake model records kwargs; `multi` ⇒
`language=None, multilingual=True`; `en` ⇒ `language="en", multilingual=False`.

### Stage 3 — `ja-en-distil` preset + GUI "Mixed (auto)" profile
Focus: `model_config.py`, `gui.py` GUI_PROFILES, `installer/setup_gui.py` MODELS (+ large-v3),
`scripts/transcribe_manager.py` choices, docs preset tables. Test: `build_command` for the Mixed
profile emits `--preset large8gb --language multi`.

### Stage 4 — verify, review, A/B [USER]
Done: suite 463 passed, pyright 0 errors, review fixes in bf96797.
Synthetic smoke (JSUT JA sentence looped between EN, scratch clip) — inconclusive, real audio owed:
- large8gb: Japanese decoded once a 30 s window is mostly JA; `multi` made no visible difference; content
  at EN↔JA boundaries dropped (synthetic repeats exaggerate this).
- ja-en-distil: per-window switching works, but emits **no timestamps** (all segments `0.0-0.0`) —
  breaks timestamps + diarization alignment unless fixed (try `without_timestamps`/`chunk_length=15`).
- Test-recording language scan: accented English windows detect as **`et` at 0.8-0.96**, so `multi`
  may decode them as Estonian. Candidate fix: restrict per-window detection to an allowed set (`en`,`ja`).
A/B clip: `data/test_mixed_en_ja.wav` (gitignored), run via `transcribe_mixed_Desk.bat` vs the same
with `--preset ja-en-distil` (DIARIZE=0):
```
.venv/bin/python -m backend.cli.main transcribe process <dir> --preset large8gb --language multi --variant 61 --output-format txt --no-diarize
.venv/bin/python -m backend.cli.main transcribe process <dir> --preset ja-en-distil --language multi --variant 61 --output-format txt --no-diarize
```
User reads the Japanese stretches → pick the GUI Mixed default.
