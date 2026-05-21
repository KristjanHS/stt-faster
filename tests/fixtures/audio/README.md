# Audio fixtures

License-clear audio fixtures for integration tests. Not all are checked
in — fixtures that depend on third-party licensed material are sourced
locally by the developer running the test and the corresponding test
skips when absent.

## `two_speakers_10s.wav`

Used by `tests/integration/test_diarize_e2e.py`. A ~10-second WAV with
two distinct speakers, mono or stereo, 16 kHz or higher.

Not checked into git. The integration test skips cleanly when the file
is absent. Acceptable sources:

- **Fresh recording**: record two people each saying ~5s. Simplest
  path; license is your own.
- **LibriSpeech** (CC BY 4.0): concatenate two short utterances from
  different speakers in `dev-clean/`. Trim to ≤ 10s with `ffmpeg`.
- **VoxConverse** (CC BY 4.0): excerpt a 10s span from a diarized
  meeting clip — pick one that the supplied RTTM marks as two-speaker.
- **Public-domain podcast**: extract a 10s window from a
  CC0/public-domain interview.

Place the file at `tests/fixtures/audio/two_speakers_10s.wav` and run:

```sh
HF_TOKEN=... .venv/bin/python -m pytest tests/integration/test_diarize_e2e.py -q
```

The test asserts: 2+ distinct speakers found; `SPEAKER_00` is the
chronologically-first speaker; `.txt` output matches
`[hh:mm:ss.ff --> hh:mm:ss.ff] SPEAKER_NN: text`.
