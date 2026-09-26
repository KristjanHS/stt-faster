"""Pinned, tokenless speaker-diarization model: constants + local-dir resolution.

Weights come from the ungated `pyannote-community` mirror at a pinned revision
(CC-BY-4.0, see NOTICE). They are fetched once at install time; at run time
the model resolves from a local dir only, never the network.
Plan: docs/plans/2026-09-26-keyless-diarization.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

from backend.diarize.errors import DiarizationConfigError

DIARIZATION_REPO = "pyannote-community/speaker-diarization-community-1"
DIARIZATION_REVISION = "8a527374977391da736e0daaef26855d949d9685"  # pragma: allowlist secret
DIARIZATION_MODEL_DIR_ENV = "STT_DIARIZATION_MODEL_DIR"
DIARIZATION_INCLUDE = ("config.yaml", "README.md", "segmentation/*", "embedding/*", "plda/*")
DIARIZATION_SHA256 = {
    "segmentation/pytorch_model.bin": (
        "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e"  # pragma: allowlist secret
    ),
    "embedding/pytorch_model.bin": (
        "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929"  # pragma: allowlist secret
    ),
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",  # pragma: allowlist secret
    "plda/xvec_transform.npz": (
        "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f"  # pragma: allowlist secret
    ),
}
NOT_INSTALLED = (
    "Speaker model not installed — run `make diarization-model` (dev) or run setup again to repair (Windows); "
    "see docs/diarization_setup.md"
)


def _hub_cache_dir(env: Mapping[str, str]) -> str | None:
    """Passed explicitly: huggingface_hub reads HF_HUB_CACHE/HF_HOME once, at its import."""
    if env.get("HF_HUB_CACHE"):
        return env["HF_HUB_CACHE"]
    if env.get("HF_HOME"):
        return str(Path(env["HF_HOME"]) / "hub")
    return None


def _cached_snapshot(cache_dir: str | None) -> Path | None:
    """The pinned snapshot from the local HF cache; `local_files_only` makes no HTTP call."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(
            snapshot_download(
                DIARIZATION_REPO,
                revision="8a527374977391da736e0daaef26855d949d9685",  # pragma: allowlist secret
                local_files_only=True,
                cache_dir=cache_dir,
            )
        )
    except LocalEntryNotFoundError:
        return None


def _complete(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in ("config.yaml", *DIARIZATION_SHA256))


def resolve_model_dir(
    env: Mapping[str, str] = os.environ,
    *,
    cached_snapshot: Callable[[str | None], Path | None] = _cached_snapshot,
) -> Path:
    """`$STT_DIARIZATION_MODEL_DIR`, else the pinned HF-cache snapshot; raises when neither is usable."""
    override = env.get(DIARIZATION_MODEL_DIR_ENV)
    if override:
        model_dir = Path(override)
        if not (model_dir / "config.yaml").is_file():
            raise DiarizationConfigError(f"{DIARIZATION_MODEL_DIR_ENV}={override} has no config.yaml")
        return model_dir
    snapshot = cached_snapshot(_hub_cache_dir(env))
    if snapshot is None or not _complete(snapshot):
        raise DiarizationConfigError(NOT_INSTALLED)
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(model_dir: Path, sha256: Mapping[str, str] = DIARIZATION_SHA256) -> None:
    """Hash-check the weights; on a mismatch delete the snapshot (and its cache blobs) so a rerun refetches."""
    bad = [
        name for name, want in sha256.items() if not (model_dir / name).is_file() or _sha256(model_dir / name) != want
    ]
    if not bad:
        return
    for name in sha256:
        (model_dir / name).resolve().unlink(missing_ok=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    raise DiarizationConfigError(f"corrupt download ({', '.join(bad)}) — rerun `make diarization-model`")


def fetch_model(
    *,
    downloader: Callable[..., str] = snapshot_download,
    cache_dir: str | None = None,
    sha256: Mapping[str, str] = DIARIZATION_SHA256,
) -> Path:
    """Download (or reuse) the pinned snapshot tokenlessly, then hash-verify it."""
    model_dir = Path(
        downloader(
            DIARIZATION_REPO,
            revision="8a527374977391da736e0daaef26855d949d9685",  # pragma: allowlist secret
            allow_patterns=list(DIARIZATION_INCLUDE),
            token=False,
            cache_dir=cache_dir,
        )
    )
    verify_snapshot(model_dir, sha256)
    return model_dir
