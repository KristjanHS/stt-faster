from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from backend.preprocess import PreprocessConfig, preprocess_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio preprocessing and keep the output file.")
    parser.add_argument("--input", required=True, help="Path to source audio (e.g., tests/test.mp3).")
    parser.add_argument("--output", required=True, help="Path to write the preprocessed WAV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    log = logging.getLogger("run_preprocess")
    cfg = PreprocessConfig.from_env()

    src = Path(args.input)
    dest = Path(args.output)

    result = preprocess_audio(src, cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(result.output_path, dest)

    steps: list[str] = []
    for m in result.metrics.steps:
        step_str = f"{m.name} {m.duration:.2f}s ({m.backend})"
        if m.metadata and "lra_used" in m.metadata:
            step_str += f" LRA={m.metadata['lra_used']:.1f}"
        steps.append(step_str)
    snr = (
        f"{result.metrics.snr_before:.2f}dB -> {result.metrics.snr_after:.2f}dB"
        if result.metrics.snr_before is not None and result.metrics.snr_after is not None
        else "n/a"
    )
    delta = f"{result.metrics.snr_delta:.2f}dB" if result.metrics.snr_delta is not None else "n/a"
    log.info("Profile: %s", result.profile)
    log.info("Input info: %s", result.input_info)
    log.info("Metrics: total=%.2fs; SNR=%s (Δ %s); steps=%s", result.metrics.total_duration, snr, delta, steps)
    log.info("Output: %s", dest)

    result.cleanup()


if __name__ == "__main__":
    main()
