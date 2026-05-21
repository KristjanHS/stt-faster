from __future__ import annotations


class DiarizationConfigError(Exception):
    """Setup problem — HF_TOKEN missing, model license unaccepted, etc.

    Batch-level: must propagate past `processor.py`'s per-file `try/except`
    that routes failures to `failed/`. Aborts the whole run.
    Message should point to `docs/diarization_setup.md`.
    """


class DiarizationRuntimeError(Exception):
    """Pyannote ran but produced unusable output for a single file.

    File-level: handled by `processor.py`'s normal failure routing.
    Caller falls back to no-speaker TXT output and logs WARNING.
    """
