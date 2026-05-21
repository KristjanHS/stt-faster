# Diarization Setup

stt-faster uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) (model `pyannote/speaker-diarization-3.1`) to attribute each transcribed segment to a speaker. This is the default for the 1:1-meeting Windows bats (`transcribe_estonian_*.bat`, `transcribe_english_*.bat`) and for any direct `transcribe_manager.py process` run without `--no-diarize`.

The model is HuggingFace-gated, so first-time setup is a one-time HF account + token + license-accept dance. After that the model is cached at `~/.cache/hf/` (WSL/Linux) or `%USERPROFILE%\.cache\hf\` (Windows) and reused across runs.

## What diarization adds

- **TXT output** gains per-segment range timestamps **and** speaker labels:

  ```
  [00:00:00.40 --> 00:00:03.05] SPEAKER_00: Tere, kuidas läheb?
  [00:00:03.10 --> 00:00:05.70] SPEAKER_01: Aitäh, hästi. Ja sinul?
  ```

  With `--no-diarize`, timestamps stay but speaker labels are omitted.

- **JSON output** gains a `speaker` field per segment when diarization is on (additive — downstream consumers that don't know about `speaker` are unaffected).

- `SPEAKER_00` is **deterministically** the chronologically-first speaker in the file (so re-runs of the same audio produce stable labels).

Defaults: `--diarize` on, `--num-speakers 2`. Override either at the CLI (`--no-diarize`, `--num-speakers N`).

## One-time setup

### 1. Create a HuggingFace account and token

- Sign up at <https://huggingface.co/join>.
- Generate a read-only access token at <https://huggingface.co/settings/tokens> (the default *Read* role is enough — no Write/Inference scope required).

### 2. Accept the pyannote model license

Open <https://huggingface.co/pyannote/speaker-diarization-3.1> while signed in and accept the gated-model terms. **This must be done with the same HF account whose token you'll use below** — accepting on a different account does nothing.

The same applies to the upstream segmentation model that pyannote pulls in transitively: <https://huggingface.co/pyannote/segmentation-3.0>. Accept that one too.

### 3. Expose the token to stt-faster

stt-faster reads the token from either env var, in this order:

1. `HF_TOKEN` (preferred)
2. `HUGGINGFACE_HUB_TOKEN`

Pick one and set it before launching the bats or the CLI.

**WSL / Linux** (`~/.bashrc`, `~/.zshrc`, or a project `.env` file loaded by your shell):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

**Windows native** — set a *User* env var via `System Properties → Environment Variables → New…`:

- Name: `HF_TOKEN`
- Value: `hf_xxxxxxxxxxxxxxxxxxxx`

Restart any open terminals (and re-open File Explorer if you launch bats by double-clicking) so they pick up the new var.

### 4. Verify the setup

Run the prefetch script — it downloads the Estonian, English, and pyannote weights into the cache, and surfaces token/license errors up front (before any transcription run):

```bash
# WSL / Linux
.venv/bin/python scripts/prefetch_models.py

# Windows native (run from the repo root)
.venv\Scripts\python scripts\prefetch_models.py
```

Expected last line: `Prefetch complete.` If you see `DiarizationConfigError`, jump to [Troubleshooting](#troubleshooting).

## Runtime constraints

- **`torchaudio<2.7`** is a hard pin — pyannote 3.4.0 calls into `torchaudio.functional.resample` with a signature that breaks in 2.7+. Don't loosen this constraint in `pyproject.toml` without first running the integration test (`make integration` or `pytest tests/integration/test_diarize_e2e.py`).
- pyannote runs sequentially **after** whisper. Both models are released and CUDA caches are flushed between loads (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`), so peak VRAM stays bounded by the larger of the two — not the sum.
- GPU is preferred but the existing GPU→CPU fallback in `backend/model_loader.py` handles boxes without CUDA; you'll see the same `[stt-faster] GPU unavailable, falling back to CPU` banner that whisper uses.

## Troubleshooting

### `DiarizationConfigError: HF_TOKEN is not set`

`HF_TOKEN` (and `HUGGINGFACE_HUB_TOKEN`) are both unset in the environment that launched the bat / CLI. Follow [step 3](#3-expose-the-token-to-stt-faster) and restart the terminal.

If the env var *is* set in your shell but the bat still complains, the Windows double-click launch context doesn't inherit shell `export`s — set the var as a *User* env var via System Properties, not just in `.bashrc`.

### `DiarizationConfigError: HF_TOKEN was rejected (401)`

The token reached HuggingFace but was rejected. Either:

- The token was revoked or rotated. Generate a fresh one at <https://huggingface.co/settings/tokens> and replace the env var.
- The token has the wrong scope. The default *Read* role is sufficient; *Fine-grained* tokens need at least *Read access to public gated repos*.

### `DiarizationConfigError: HuggingFace returned 403 ... Accept the model license`

The token is valid but the HF account behind it hasn't accepted the gated-model terms for `pyannote/speaker-diarization-3.1`. Re-do [step 2](#2-accept-the-pyannote-model-license), making sure you're signed in as the same account that owns the token.

Also check `pyannote/segmentation-3.0` — pyannote 3.1 depends on it and the 403 can come from either model.

### `DiarizationRuntimeError: pyannote inference failed for <file>`

pyannote loaded but crashed on a specific file. This is per-file, not batch-level — the file routes to `failed/` and the run continues. Common causes:

- Corrupted or zero-length audio. Check the file plays in a media player.
- Format pyannote rejects (very rare — pyannote uses torchaudio, which accepts everything in our supported list).

### GPU OOM (CUDA out of memory)

pyannote 3.1 needs roughly 2 GB VRAM on top of whisper at peak. If your card is tight (≤4 GB):

- Use a smaller whisper preset (`distil`, `et-32`) — frees more headroom for pyannote.
- Force CPU on the diarize stage by unsetting CUDA: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/transcribe_manager.py process ...` runs both whisper and pyannote on CPU.
- Run with `--no-diarize` if you don't need speaker labels for this particular batch.

### Zero turns returned (no speaker labels in output)

If pyannote returns no diarization turns (silence, music, or sub-threshold speech), `annotate()` logs a single WARNING and returns the segments unchanged — TXT emits timestamps without speaker prefixes. This is expected behaviour, not an error.

If you're seeing this on audio that clearly contains speech, check:

- Sample rate / channel count — pyannote handles a wide range but extreme cases (e.g. 8 kHz telephony) sometimes drop below detection threshold. Try the FFmpeg-pipeline variants (3-11) which resample to 16 kHz.
- `--num-speakers` matches the actual count. Forcing `--num-speakers 5` on a single-speaker file is unstable.

### Diarization is slow on long files

pyannote 3.1 is ~real-time on GPU and ~3-5× real-time on CPU. For batch runs of long files (1 hour+) on CPU-only machines, expect the diarize stage alone to take 10-20 minutes per file. Use `--no-diarize` if speaker labels aren't needed; the TXT output still gets timestamps.

## See also

- [docs/Transcription_solution.md](Transcription_solution.md) — full output format reference (TXT line shape, JSON schema with `speaker`).
- [scripts/windows/HOW_TO_USE.txt](../scripts/windows/HOW_TO_USE.txt) — Windows launcher reference.
- pyannote model card: <https://huggingface.co/pyannote/speaker-diarization-3.1>.
