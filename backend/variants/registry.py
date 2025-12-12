"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.transcription_presets import (
    create_minimal_config,
    create_project_config,
)
from backend.variants.variant import PreprocessStep, Variant


def _get_all_variants() -> list[Variant]:
    """Return all defined variants (both active and inactive).

    This is an internal function that defines all variants.
    Use get_builtin_variants() to get only active variants.

    Variants are ordered from simplest to most complex, with similar variants grouped together:
    - Variants 1-7: No preprocessing (simplest)
    - Variants 8-12: Single preprocessing step
    - Variants 13-15: Single normalization step
    - Variant 16: Multiple preprocessing steps
    - Variants 17-20: Complex normalization
    - Variants 21-26: Conservative sweep variants (no preprocessing)
    """
    return [
        # Group 1: No preprocessing (simplest) - Variants 1-7
        # Variant 1: No preprocessing + minimal transcription parameters (simplest)
        Variant(
            name="noprep_noparamtrans",
            number=1,
            preprocess_steps=[],
            transcription_config=create_minimal_config(),
        ),
        # Variant 2: No preprocessing + project defaults
        Variant(
            name="no_preprocessing",
            number=2,
            preprocess_steps=[],
            transcription_config=create_project_config(),
        ),
        # Variant 3: No preprocessing + minimal defaults
        Variant(
            name="noprep_minimal",
            number=3,
            preprocess_steps=[],
            transcription_config=create_minimal_config(),
        ),
        # Variant 4: No preprocessing + minimal params + no_speech_threshold
        Variant(
            name="noprep_minimal_no_speech_threshold",
            number=4,
            preprocess_steps=[],
            transcription_config=create_minimal_config(no_speech_threshold=0.3),
        ),
        # Variant 5: No preprocessing + minimal params + chunk_length
        Variant(
            name="noprep_minimal_chunk_length",
            number=5,
            preprocess_steps=[],
            transcription_config=create_minimal_config(chunk_length=20),
        ),
        # Variant 6: No preprocessing + minimal params + condition_on_previous_text
        Variant(
            name="noprep_minimal_condition_on_previous_text",
            number=6,
            preprocess_steps=[],
            transcription_config=create_minimal_config(condition_on_previous_text=False),
        ),
        # Variant 7: No preprocessing + minimal params + beam_size override
        Variant(
            name="noprep_minimal_beam_size",
            number=7,
            preprocess_steps=[],
            transcription_config=create_minimal_config(beam_size=8),
        ),
        # Group 2: Single preprocessing step - Variants 8-12
        # Variant 8: Only normalization + minimal transcription parameters
        Variant(
            name="normonly_noparamtrans",
            number=8,
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 9: Only ffmpeg pipeline + project defaults
        Variant(
            name="ffmpeg_only",
            number=9,
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_config=create_project_config(),
        ),
        # Variant 10: Only ffmpeg pipeline + minimal defaults
        Variant(
            name="ffmpeg_minimal",
            number=10,
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 11: Only denoise_light + project defaults
        Variant(
            name="denoise_only",
            number=11,
            preprocess_steps=[
                PreprocessStep(name="resample", enabled=True, step_type="resample"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_config=create_project_config(),
        ),
        # Variant 12: Only denoise + minimal transcription parameters
        Variant(
            name="onlyden_noparamtrans",
            number=12,
            preprocess_steps=[
                PreprocessStep(name="resample", enabled=True, step_type="resample"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Group 3: Single normalization step - Variants 13-15
        # Variant 13: Loudness normalization + highpass + minimal params
        Variant(
            name="norm_highp_noparamtrans",
            number=13,
            preprocess_steps=[
                PreprocessStep(
                    name="loudnorm_highpass",
                    enabled=True,
                    step_type="loudnorm_highpass",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 14: Dynamic audio normalization + minimal params
        Variant(
            name="norm_dynaud_noparamtrans",
            number=14,
            preprocess_steps=[
                PreprocessStep(name="dynaudnorm", enabled=True, step_type="dynaudnorm"),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 15: 2-pass loudnorm in linear mode (file-wide gain change) + minimal params
        Variant(
            name="loudnorm_2pass_linear_noparamtrans",
            number=15,
            preprocess_steps=[
                PreprocessStep(
                    name="loudnorm_2pass_linear",
                    enabled=True,
                    step_type="loudnorm_2pass_linear",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Group 4: Multiple preprocessing steps - Variant 16
        # Variant 16: Full preprocessing (ffmpeg + denoise) + minimal defaults
        Variant(
            name="full_minimal",
            number=16,
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Group 5: Complex normalization - Variants 17-20
        # Variant 17: Highpass + lowpass + aformat + loudness normalization + minimal params
        Variant(
            name="lnorm_highlow_aform_noparamtrans",
            number=17,
            preprocess_steps=[
                PreprocessStep(
                    name="highlow_aform_loudnorm",
                    enabled=True,
                    step_type="highlow_aform_loudnorm",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 18: Highpass + lowpass + loudness normalization (no aformat) + minimal params
        Variant(
            name="lnorm_highlow_nosampl_noparamtrans",
            number=18,
            preprocess_steps=[
                PreprocessStep(
                    name="highlow_nosampl_loudnorm",
                    enabled=True,
                    step_type="highlow_nosampl_loudnorm",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 19: Aresample to 16kHz + loudness normalization (fixed params) + minimal params
        Variant(
            name="lnorm2_aresampl_noparamtrans",
            number=19,
            preprocess_steps=[
                PreprocessStep(
                    name="aresampl_loudnorm_fixed",
                    enabled=True,
                    step_type="aresampl_loudnorm_fixed",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Variant 20: Aresample to 16kHz + loudness normalization (I=-24, LRA=15) + minimal params
        Variant(
            name="lnorm3_aresampl_noparamtrans",
            number=20,
            preprocess_steps=[
                PreprocessStep(
                    name="aresampl_loudnorm_fixed2",
                    enabled=True,
                    step_type="aresampl_loudnorm_fixed2",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Group 6: Conservative sweep variants (no preprocessing) - Variants 21-26
        # Variant 21: No preprocessing + minimal params + beam_size=4
        Variant(
            name="noprep_minimal_beam_size_4",
            number=21,
            preprocess_steps=[],
            transcription_config=create_minimal_config(beam_size=4),
        ),
        # Variant 22: No preprocessing + minimal params + beam_size=6
        Variant(
            name="noprep_minimal_beam_size_6",
            number=22,
            preprocess_steps=[],
            transcription_config=create_minimal_config(beam_size=6),
        ),
        # Variant 23: No preprocessing + minimal params + chunk_length=18
        Variant(
            name="noprep_minimal_chunk_length_18",
            number=23,
            preprocess_steps=[],
            transcription_config=create_minimal_config(chunk_length=18),
        ),
        # Variant 24: No preprocessing + minimal params + chunk_length=22
        Variant(
            name="noprep_minimal_chunk_length_22",
            number=24,
            preprocess_steps=[],
            transcription_config=create_minimal_config(chunk_length=22),
        ),
        # Variant 25: No preprocessing + minimal params + no_speech_threshold=0.45
        Variant(
            name="noprep_minimal_no_speech_045",
            number=25,
            preprocess_steps=[],
            transcription_config=create_minimal_config(no_speech_threshold=0.45),
        ),
        # Variant 26: No preprocessing + minimal params + no_speech_threshold=0.55
        Variant(
            name="noprep_minimal_no_speech_055",
            number=26,
            preprocess_steps=[],
            transcription_config=create_minimal_config(no_speech_threshold=0.55),
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


def get_builtin_variants() -> list[Variant]:
    """Return list of active built-in variants.

    All other variants are disabled but remain defined and can be accessed
    via get_variant_by_number() or get_variant_by_name().
    All variants now use declarative preprocessing steps.
    """
    all_variants = _get_all_variants()
    # Only return active variants:
    # 1 (noprep_noparamtrans), 4 (noprep_minimal_no_speech_threshold), 7 (noprep_minimal_beam_size)
    active_variant_numbers = {1, 4, 7}
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
        number: Variant number (1-26)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None
