# Diarization Setup

stt-faster uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) with the `speaker-diarization-community-1` pipeline to attribute each transcribed segment to a speaker. Diarization is on by default for the Windows bats and for `transcribe_manager.py process` (`--no-diarize` turns it off). pyannote ships with the `cpu`/`cu130` uv extras; a sync without either warns and skips `--diarize`.

No Hugging Face account or token is needed. The weights (about 33 MB) come from the ungated mirror `pyannote-community/speaker-diarization-community-1`, pinned to revision `8a527374977391da736e0daaef26855d949d9685` and sha256-verified at download. Transcription loads them from local disk only, never the network. Attribution: [NOTICE](../NOTICE).

## What diarization adds

- **TXT output** gains per-segment range timestamps **and** speaker labels:

  ```
  [00:00:00.40 --> 00:00:03.05] SPEAKER_00: Tere, kas sa kuuled mind?
  [00:00:03.10 --> 00:00:05.70] SPEAKER_01: Jah kuulen hästi.
  ```

  With `--no-diarize`, timestamps stay but speaker labels are omitted.

- **JSON output** gains a `speaker` field per segment when diarization is on (additive — downstream consumers that don't know about `speaker` are unaffected).

- `SPEAKER_00` is **deterministically** the chronologically-first speaker in the file (so re-runs of the same audio produce stable labels).

Defaults: `--diarize` on, `--num-speakers 2`. Override either at the CLI (`--no-diarize`, `--num-speakers N`).

## Install the model (one step)

- **Windows app:** the installer fetches it; Start ▸ Transcribe ▸ Repair restores it.
- **Dev (WSL/Linux):** `make diarization-model` (with `make rnnoise-model` for denoising). It downloads into the HF cache — `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` — and a rerun downloads nothing.

At run time the model resolves from `$STT_DIARIZATION_MODEL_DIR` if set, else the pinned snapshot in that same HF cache (`backend/diarize/model.py::resolve_model_dir`).

### Air-gapped machines

Copy the snapshot dir `<hf cache>/models--pyannote-community--speaker-diarization-community-1/snapshots/8a527374977391da736e0daaef26855d949d9685/` with `cp -rL` (Linux snapshot files are symlinks into `../../blobs`) and point `STT_DIARIZATION_MODEL_DIR` at the copy.

### Docker

The image carries no models. Mount the host's model repo dir read-only and point the env var at its snapshot (the whole repo dir, so the snapshot's blob symlinks resolve):

```bash
make diarization-model rnnoise-model
docker run --rm \
  -v "$PWD/audio:/workspace" \
  -v "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--pyannote-community--speaker-diarization-community-1:/models/diarization:ro" \
  -e STT_DIARIZATION_MODEL_DIR=/models/diarization/snapshots/8a527374977391da736e0daaef26855d949d9685 \
  -v "$PWD/models/sh.rnnn:/models/sh.rnnn:ro" \
  -e STT_PREPROCESS_RNNOISE_MODEL=/models/sh.rnnn \
  stt-faster:latest process /workspace --diarize
```

## Runtime constraints

- **`torchaudio>=2.8.0`** is required by `pyannote.audio==4.0.4`. The torch / torchaudio versions are pinned in the `cpu` / `cu130` extras of `pyproject.toml` (cu130 extra requires `>=2.11.0`); don't loosen them without first running the integration test (`make integration` or `pytest tests/integration/test_diarize_with_pyannote.py`). Default install is CPU; opt into GPU via `make use-gpu` (writes the gitignored `.stt-variant.local`) before `./run_uv.sh`.
- Audio is **decoded in-process with PyAV** (`av==16.0.1`, bundles its own FFmpeg) and handed to pyannote via its tensor-input API. This sidesteps `torchcodec`'s requirement for FFmpeg-6 shared libraries on the host, so the pipeline is portable across Win + WSL + Docker without any system-FFmpeg version dance. A `torchcodec` import warning may fire once at process start — it's filtered, but if you see it through a stray logger config: it's safe to ignore on the tensor-input path.
- pyannote runs sequentially **after** whisper. Both models are released and CUDA caches are flushed between loads (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`), so peak VRAM stays bounded by the larger of the two — not the sum.
- GPU is preferred but the existing GPU→CPU fallback in `backend/model_loader.py` handles boxes without CUDA; you'll see the same `[stt-faster] GPU unavailable, falling back to CPU` banner that whisper uses.

## Troubleshooting

### `DiarizationConfigError: Speaker model not installed`

Neither `$STT_DIARIZATION_MODEL_DIR` nor the HF cache holds the pinned snapshot. Run `make diarization-model` (dev) or Start ▸ Transcribe ▸ Repair (Windows). If you downloaded it under a different `HF_HOME`/`HF_HUB_CACHE`, run with that same value set.

### `DiarizationConfigError: STT_DIARIZATION_MODEL_DIR=… has no config.yaml`

The variable points at the wrong dir: it must be the snapshot dir itself (the one holding `config.yaml`, `segmentation/`, `embedding/`, `plda/`).

### `DiarizationConfigError: corrupt download (…)`

A weight file failed its sha256 check; the snapshot was deleted. Rerun `make diarization-model` (or Start ▸ Transcribe ▸ Repair).

### `DiarizationRuntimeError: pyannote inference failed for <file>`

pyannote loaded but crashed on a specific file. This is per-file, not batch-level — the file routes to `failed/` and the run continues. Common causes:

- Corrupted or zero-length audio. Check the file plays in a media player.
- Format pyannote rejects (very rare — pyannote uses torchaudio, which accepts everything in our supported list).

### GPU OOM (CUDA out of memory)

community-1 needs roughly 2 GB VRAM on top of whisper at peak. If your card is tight (≤4 GB):

- Use a smaller whisper preset (`distil`, `et-32`) — frees more headroom for pyannote.
- Force CPU on the diarize stage by unsetting CUDA: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/transcribe_manager.py process ...` runs both whisper and pyannote on CPU.
- Run with `--no-diarize` if you don't need speaker labels for this particular batch.

### Zero turns returned (no speaker labels in output)

If pyannote returns no diarization turns (silence, music, or sub-threshold speech), `annotate()` logs a single WARNING and returns the segments unchanged — TXT emits timestamps without speaker prefixes. This is expected behaviour, not an error.

If you're seeing this on audio that clearly contains speech, check:

- Sample rate / channel count — pyannote handles a wide range but extreme cases (e.g. 8 kHz telephony) sometimes drop below detection threshold. Try the FFmpeg-pipeline variants (3-11) which resample to 16 kHz.
- `--num-speakers` matches the actual count. Forcing `--num-speakers 5` on a single-speaker file is unstable.

### Diarization is slow on long files

community-1 is ~real-time on GPU and ~3-5× real-time on CPU. For batch runs of long files (1 hour+) on CPU-only machines, expect the diarize stage alone to take 10-20 minutes per file. Use `--no-diarize` if speaker labels aren't needed; the TXT output still gets timestamps.

## See also

- [docs/Transcription_solution.md](Transcription_solution.md) — full output format reference (TXT line shape, JSON schema with `speaker`).
- [scripts/windows/README.md](../scripts/windows/README.md) — Windows launcher reference.
- Model card: <https://huggingface.co/pyannote-community/speaker-diarization-community-1>.
