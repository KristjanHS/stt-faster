"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.steps import (
    VolumeLimiterStepConfig,
)
from backend.variants.transcription_presets import (
    create_baseline_config,
    create_no_vad_baseline_config,
)
from backend.variants.variant import PreprocessStep, Variant


def _get_all_variants() -> list[Variant]:
    """Return all defined variants (both active and inactive).

    This is an internal function that defines all variants.
    Use get_builtin_variants() to get only active variants.

    Variants are ordered from simplest to most complex, with similar variants grouped together:
    - Variants 1-2: No preprocessing, baseline configs (simplest)
    - Variants 3-6: No preprocessing, diagnostic/config tuning
    - Variants 7-11: No preprocessing, production VAD tuning
    - Variants 12-21: Decode quality and silence guard variants
      - 12-15: Group A - Improve Variant 2 accuracy (beam 6/7, patience)
      - 16-18: Group B - Ultra conservative silence guard (Guard-lite)
      - 19-20: Group C - No context carryover variants
      - 21: Group D - Boundary effects (chunk 25)
    - Variant 22: Complex normalization (aresample + loudnorm)
    - Variants 23-32: Conservative level/volume tweaks
      - 23: P0 - Control (resample only)
      - 24: P1 - Limiter only
      - 25-26: P2-P3 - Fixed gain + limiter (+1.5dB, +3dB)
      - 27-29: P4-P6 - Peak normalize 2-pass (-6dBFS, -3dBFS, -3dBFS max +3dB)
      - 30: P7 - SoX peak normalize (-3dBFS)
      - 31-32: P8-P9 - Compressor and dynaudnorm (conservative)
    - Variant 33: Variant 2 with word timestamps
    - Variants 34-38: Variant 1 derivatives (best-bet combinations)
      - 34: Variant 1 + word timestamps
      - 35: Variant 1 + Beam 7
      - 36: Variant 1 + P2 (+1.5 dB + limiter)
      - 37: Variant 1 + P8 (compressor+limiter)
      - 38: Beam 7 + Guard-lite (softened)
    - Variants 39-46: Variant 36 derivatives (P2 preprocessing with parameter tweaks)
      - 39: Variant 36 + beam_size=6
      - 40: Variant 36 + beam_size=7
      - 41: Variant 36 + volume_db=2.0
      - 42: Variant 44 without beam_size, but with patience=1.2
      - 43: Variant 36 + beam_size=6 + patience=1.2
      - 44: Variant 36 + beam_size=7 + patience=1.2
      - 45: Variant 36 + chunk_length=25
      - 46: Variant 36 + word_timestamps=True
    - Variants 47-51: Variant 44 derivatives (one-change-only tweaks to beat variant 44)
      - 47: Variant 44 + patience=1.3 (increased patience, keep beam=7)
      - 48: Variant 44 + beam_size=8 (increased beam, keep patience=1.2)
      - 49: Variant 44 + repetition_penalty=1.1 (add repetition penalty)
      - 50: Variant 44 + no_repeat_ngram_size=3 (explicit n-gram blocking)
      - 51: Variant 44 + length_penalty=0.95 (tune length bias)
    """
    return [
        # Group 1: No preprocessing, baseline configs (simplest) - Variants 1-2
        # Variant 1: No preprocessing + true baseline (no overrides, uses library defaults)
        Variant(
            name="baseline_true_defaults",
            number=1,
            preprocess_steps=[],
            transcription_config=create_baseline_config(),  # empty overrides
        ),
        # Variant 2: No preprocessing + raw decode baseline (no VAD by design)
        Variant(
            name="baseline_no_vad",
            number=2,
            preprocess_steps=[],
            transcription_config=create_no_vad_baseline_config(),
        ),
        # Variant 36: Variant 1 + P2 (+1.5 dB + limiter)
        # Combine P2 loudness tweak with baseline decoder to test preprocessing impact
        Variant(
            name="baseline_p2_volume_1_5db",
            number=36,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=create_baseline_config(),
        ),
        # Variant 42: Variant 44 without beam_size, but with patience=1.2
        Variant(
            name="p2_volume_1_5db_pat12",
            number=42,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("patience", 1.2),
                    config,
                )[2]
            )(),
        ),
        # Variant 44: Variant 36 + beam_size=7 + patience=1.2
        Variant(
            name="p2_volume_1_5db_beam7_pat12",
            number=44,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config.set("patience", 1.2),
                    config,
                )[3]
            )(),
        ),
        # Group 8: Variant 44 derivatives (one-change-only tweaks) - Variants 47-51
        # Variant 47: Variant 44 + patience=1.3 (increased patience, keep beam=7)
        Variant(
            name="p2_volume_1_5db_beam7_pat13",
            number=47,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config.set("patience", 1.3),
                    config,
                )[3]
            )(),
        ),
        # Variant 48: Variant 44 + beam_size=8 (increased beam, keep patience=1.2)
        Variant(
            name="p2_volume_1_5db_beam8_pat12",
            number=48,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 8),
                    config.set("patience", 1.2),
                    config,
                )[3]
            )(),
        ),
        # Variant 49: Variant 44 + repetition_penalty=1.1 (add repetition penalty)
        Variant(
            name="p2_volume_1_5db_beam7_pat12_reppat11",
            number=49,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config.set("patience", 1.2),
                    config.set("repetition_penalty", 1.1),
                    config,
                )[4]
            )(),
        ),
        # Variant 50: Variant 44 + no_repeat_ngram_size=3 (explicit n-gram blocking)
        Variant(
            name="p2_volume_1_5db_beam7_pat12_norep3",
            number=50,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config.set("patience", 1.2),
                    config.set("no_repeat_ngram_size", 3),
                    config,
                )[4]
            )(),
        ),
        # Variant 51: Variant 44 + length_penalty=0.95 (tune length bias)
        Variant(
            name="p2_volume_1_5db_beam7_pat12_lenpat095",
            number=51,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config.set("patience", 1.2),
                    config.set("length_penalty", 0.95),
                    config,
                )[4]
            )(),
        ),
    ]


def get_all_variants() -> list[Variant]:
    """Return all defined variants (both active and inactive).

    This is useful for testing and validation purposes.
    For normal usage, use get_builtin_variants() to get only active variants.

    Returns:
        List of all Variant instances
    """
    return _get_all_variants()


# Baseline variant number used for comparison and diff calculations in reports
baseline_variant_num = 44


def get_builtin_variants() -> list[Variant]:
    """Return list of active built-in variants.

    All variants are now active. Variants can still be accessed individually
    via get_variant_by_number() or get_variant_by_name().
    All variants now use declarative preprocessing steps.
    """
    all_variants = _get_all_variants()
    # Active variants: 1, 36, 42, 44
    active_variant_numbers = {1} | {36} | {42} | {44}
    # active_variant_numbers = set(range(1, 52))
    return [v for v in all_variants if v.number in active_variant_numbers]


def get_variant_by_name(name: str) -> Variant | None:
    """Get a variant by its name.

    Searches through all variants (both active and inactive).

    Args:
        name: Variant name (e.g., "no_preprocessing")

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.name == name:
            return variant
    return None


def get_variant_by_number(number: int) -> Variant | None:
    """Get a variant by its number.

    Searches through all variants (both active and inactive).

    Args:
        number: Variant number (1-51)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None
