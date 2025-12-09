"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.variant import PreprocessStep, Variant


def get_builtin_variants() -> list[Variant]:
    """Return list of all 9 built-in variants.

    These match the variants currently defined in compare_transcription_variants.py.
    """
    return [
        # Variant 1: No preprocessing + project defaults
        Variant(
            name="no_preprocessing",
            number=1,
            description="nopre_projdef",
            preprocess_steps=[],
            transcription_preset="project",
        ),
        # Variant 2: No preprocessing + industry defaults
        Variant(
            name="industry_defaults",
            number=2,
            description="nopre_inddef",
            preprocess_steps=[],
            transcription_preset="industry",
        ),
        # Variant 3: Only ffmpeg pipeline + project defaults
        Variant(
            name="ffmpeg_only",
            number=3,
            description="ffmpegonly_projdef",
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_preset="project",
        ),
        # Variant 4: Only denoise_light + project defaults
        Variant(
            name="denoise_only",
            number=4,
            description="onlyden_projdef",
            preprocess_steps=[
                PreprocessStep(name="resample", enabled=True, step_type="resample"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_preset="project",
        ),
        # Variant 5: Only ffmpeg pipeline + industry defaults
        Variant(
            name="ffmpeg_industry_defaults",
            number=5,
            description="ffmpegonly_inddef",
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_preset="industry",
        ),
        # Variant 6: Full preprocessing (ffmpeg + denoise) + industry defaults
        Variant(
            name="full_industry_defaults",
            number=6,
            description="fullpre_inddef",
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_preset="industry",
        ),
        # Variant 7: No preprocessing + minimal transcription parameters
        Variant(
            name="noprep_noparamtrans",
            number=7,
            description="nopre_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
        ),
        # Variant 8: Only normalization + minimal transcription parameters
        Variant(
            name="normonly_noparamtrans",
            number=8,
            description="normonly_noparam",
            preprocess_steps=[
                PreprocessStep(name="ffmpeg_pipeline", enabled=True, step_type="ffmpeg"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 9: Only denoise + minimal transcription parameters
        Variant(
            name="onlyden_noparamtrans",
            number=9,
            description="onlyden_noparam",
            preprocess_steps=[
                PreprocessStep(name="resample", enabled=True, step_type="resample"),
                PreprocessStep(name="denoise_light", enabled=True, step_type="denoise"),
            ],
            transcription_preset="minimal",
        ),
    ]


def get_variant_by_name(name: str) -> Variant | None:
    """Get a variant by its name.

    Args:
        name: Variant name (e.g., "no_preprocessing")

    Returns:
        Variant instance or None if not found
    """
    for variant in get_builtin_variants():
        if variant.name == name:
            return variant
    return None


def get_variant_by_number(number: int) -> Variant | None:
    """Get a variant by its number.

    Args:
        number: Variant number (1-9)

    Returns:
        Variant instance or None if not found
    """
    for variant in get_builtin_variants():
        if variant.number == number:
            return variant
    return None
