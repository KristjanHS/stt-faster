from __future__ import annotations


class DiarizationConfigError(Exception):
    """Setup problem — speaker model not installed, pyannote.audio missing, etc.

    Caught per file like any error (`components.py` routes the file to `failed/`);
    the GUI keys its no-speaker retry on this name.
    """


class DiarizationRuntimeError(Exception):
    """Pyannote ran but produced unusable output for a single file.

    Caught per file like any error (`components.py` routes the file to `failed/`);
    the GUI keys its no-speaker retry on this name.
    """
