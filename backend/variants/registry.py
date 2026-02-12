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
        # Variant 36: Variant 1 + P2 (+1.1 dB + limiter)
        # Combine P2 loudness tweak with baseline decoder to test preprocessing impact
        Variant(
            name="baseline_p2_volume_1_1db",
            number=36,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_1db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.1),
                ),
            ],
            transcription_config=create_baseline_config(),
        ),
        # Variant 52: Variant 36 with volume_db=1.0
        Variant(
            name="baseline_p2_volume_1_0db",
            number=52,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_0db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.0),
                ),
            ],
            transcription_config=create_baseline_config(),
        ),
        # Variant 42: Variant 36 + patience=1.1
        Variant(
            name="p2_volume_1_1db_pat11",
            number=42,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_1db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.1),
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
        # Variant 53: Variant 42 with volume_db=1.0
        Variant(
            name="p2_volume_1_0db_pat12",
            number=53,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_0db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.0),
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
        # Variant 43: Variant 42 + volume_db=1.0 + patience=1.1
        Variant(
            name="p2_volume_1_0db_pat13",
            number=43,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_0db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.0),
                ),
            ],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("patience", 1.3),
                    config,
                )[2]
            )(),
        ),
        # Variant 44: Variant 36 + beam_size=7 + patience=1.2 with volume_db=1.5
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
        # Variant 45: Variant 44 + volume_db=1.0
        Variant(
            name="p2_volume_1_0db_beam7_pat12",
            number=45,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_0db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.0),
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
        # Variant 56: Variant 44 + repetition_penalty=1.05
        Variant(
            name="p2_volume_1_5db_beam7_pat12_rep105",
            number=56,
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
                    config.set("repetition_penalty", 1.05),
                    config,
                )[4]
            )(),
        ),
        # Variant 57: Variant 44 + no_repeat_ngram_size=2
        Variant(
            name="p2_volume_1_5db_beam7_pat12_norep2",
            number=57,
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
                    config.set("no_repeat_ngram_size", 2),
                    config,
                )[4]
            )(),
        ),
        # Variant 58: Variant 44 + length_penalty=0.98
        Variant(
            name="p2_volume_1_5db_beam7_pat12_len098",
            number=58,
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
                    config.set("length_penalty", 0.98),
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

    Variants can still be accessed individually via get_variant_by_number() or get_variant_by_name().
    All variants now use declarative preprocessing steps.

    Variants are ordered from simplest to most complex, with similar variants grouped together:
    - Variants 1-2: No preprocessing, baseline configs (simplest)
    - Variant 36: Variant 1 + P2 (+1.1 dB + limiter)
    - Variant 42: Variant 36 + patience=1.2
    - Variant 43: Variant 36 + patience=1.1
    - Variant 44: Variant 36 + beam_size=7 + patience=1.2 with volume_db=1.5
    - Variant 45: Variant 44 with volume_db=1.1
    - Variant 54: Variant 44 + patience=1.3
    - Variant 55: Variant 44 + beam_size=8
    - Variant 56: Variant 44 + repetition_penalty=1.1
    - Variant 57: Variant 44 + no_repeat_ngram_size=3
    - Variant 58: Variant 44 + length_penalty=0.95
    - Variant 52: Variant 36 with volume_db=1.0
    - Variant 53: Variant 42 with volume_db=1.0
    """
    all_variants = _get_all_variants()
    # active_variant_numbers = {1} | {36} | set(range(42, 46)) | set(range(52, 54))
    active_variant_numbers = {44, 56, 57, 58}

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
        number: Variant number (1-70)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None
