"""Download an RNNoise .rnnn model to the repository `models/` folder.

Usage:
  python scripts/download_rnnoise_model.py --url <MODEL_URL> [--output models/sh.rnnn]

If no URL is supplied, set the environment variable `RNNOISE_MODEL_URL` to point to a download location.
This script avoids guessing URLs; if you want, provide the known URL for the model you trust.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_OUTPUT = Path("models/sh.rnnn")
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn"
)

LOGGER = logging.getLogger(__name__)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading RNNoise model from %s to %s", url, dest)
    try:
        # Validate URL scheme - only allow http/https for security
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsafe URL scheme '{parsed.scheme}' in URL: {url}. Only http/https allowed.")
        # Prefer HTTPS for secure downloads
        if parsed.scheme == "http":
            LOGGER.warning("Using HTTP instead of HTTPS for URL: %s", url)

        # Use httpx with SSL verification enabled by default for secure downloads
        with httpx.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in resp.iter_bytes(chunk_size=16_384):
                    out.write(chunk)
    except Exception as exc:  # pragma: no cover - network I/O
        LOGGER.error("Download failed: %s", exc)
        sys.exit(2)
    LOGGER.info("Download completed: %s", dest)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig()
    parser = argparse.ArgumentParser(description="Download RNNoise model (.rnnn) to models/")
    parser.add_argument(
        "--url",
        default=os.environ.get("RNNOISE_MODEL_URL") or DEFAULT_MODEL_URL,
        help="Model URL (optional, can be provided via RNNOISE_MODEL_URL env)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination path")
    args = parser.parse_args(argv)

    url = args.url or os.environ.get("RNNOISE_MODEL_URL")
    if not url:
        LOGGER.error("No model URL provided. Set --url or RNNOISE_MODEL_URL environment variable.")
        return 1

    dest = Path(args.output)
    download(url, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
