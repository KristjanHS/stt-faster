# Audio fixtures

License-clear audio fixtures for integration tests. Not all are checked
in — fixtures that depend on third-party licensed material are sourced
locally by the developer running the test and the corresponding test
skips when absent.

## `two_speakers_10s.wav`

Used by `tests/integration/test_diarize_with_pyannote.py`. A 10-second mono
16 kHz PCM WAV with two distinct speakers.

Not checked into git. The integration test skips cleanly when the
file is absent.

### Recommended source: pyannote-audio `sample.wav` (MIT)

The upstream pyannote-audio repo ships a 30s diarization demo at
`src/pyannote/audio/sample/sample.wav` with a companion `sample.rttm`
ground truth labelling two speakers (`speaker90`, `speaker91`)
alternating from ~6.7s onward. The first ~10s contain only short turns
(≤1.7s each) which pyannote community-1 collapses into a single speaker.
Skip to 10s and take the next 10s — that window lands on a 4.13s
speaker90 block followed by a 3.43s speaker91 block, which community-1
reliably separates:

```sh
curl -fsSL -o /tmp/pa_sample.wav \
  https://github.com/pyannote/pyannote-audio/raw/main/src/pyannote/audio/sample/sample.wav
ffmpeg -y -ss 10 -i /tmp/pa_sample.wav -t 10 -ac 1 -ar 16000 -c:a pcm_s16le \
  tests/fixtures/audio/two_speakers_10s.wav
```

License: MIT (same as the pyannote-audio repo).

### Fallback sources

- **Fresh recording**: record two people each saying ~5s. License is your own.
- **LibriSpeech** (CC BY 4.0): concatenate two short utterances from
  different speakers in `dev-clean/`. Trim to ≤ 10s with `ffmpeg`.
- **VoxConverse** (CC BY 4.0): excerpt a 10s span from a diarized
  clip — pick one that the supplied RTTM marks as two-speaker.
- **Public-domain podcast**: extract a 10s window from a
  CC0/public-domain two-speaker recording.

### Running the test

```sh
HF_TOKEN=... .venv/bin/python -m pytest tests/integration/test_diarize_with_pyannote.py -q
```

The test asserts: 2+ distinct speakers found; `SPEAKER_00` is the
chronologically-first speaker; `.txt` output matches
`[hh:mm:ss.ff --> hh:mm:ss.ff] SPEAKER_NN: text`.
