"""Unit tests for scripts/prefetch_models.py — pinned, verified RNNoise prefetch."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from scripts.prefetch_models import RNNOISE_URL, prefetch_rnnoise

PAYLOAD = b"rnnoise-weights"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_downloads_pinned_url_and_verifies(tmp_path: Path) -> None:
    urls: list[str] = []

    def opener(url: str) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(PAYLOAD)

    dest = tmp_path / "models" / "sh.rnnn"
    assert prefetch_rnnoise(dest, opener=opener, sha256=PAYLOAD_SHA) == dest
    assert dest.read_bytes() == PAYLOAD
    assert urls == [RNNOISE_URL]
    assert "/master/" not in RNNOISE_URL
    prefetch_rnnoise(dest, opener=opener, sha256=PAYLOAD_SHA)
    assert urls == [RNNOISE_URL], "a verified file is not refetched"


def test_bad_hash_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "sh.rnnn"
    with pytest.raises(RuntimeError, match="sha256"):
        prefetch_rnnoise(dest, opener=lambda _url: io.BytesIO(b"tampered"), sha256=PAYLOAD_SHA)
    assert not dest.exists()
