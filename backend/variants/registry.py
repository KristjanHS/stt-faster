"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.variant import PreprocessStep, Variant


def get_builtin_variants() -> list[Variant]:
    """Return list of all built-in variants (1-16).

    These match the variants currently defined in compare_transcription_variants.py.
    All variants now use declarative preprocessing steps.
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
        # Variant 10: Loudness normalization + highpass + minimal params
        Variant(
            name="norm_highp_noparamtrans",
            number=10,
            description="norm_highp_noparam",
            preprocess_steps=[
                PreprocessStep(name="loudnorm_highpass", enabled=True, step_type="loudnorm_highpass"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 11: Dynamic audio normalization + minimal params
        Variant(
            name="norm_dynaud_noparamtrans",
            number=11,
            description="norm_dynaud_noparam",
            preprocess_steps=[
                PreprocessStep(name="dynaudnorm", enabled=True, step_type="dynaudnorm"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 12: Highpass + lowpass + aformat + loudness normalization + minimal params
        Variant(
            name="lnorm_highlow_aform_noparamtrans",
            number=12,
            description="lnorm_highlow_aform_noparam",
            preprocess_steps=[
                PreprocessStep(name="highlow_aform_loudnorm", enabled=True, step_type="highlow_aform_loudnorm"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 13: Highpass + lowpass + loudness normalization (no aformat) + minimal params
        Variant(
            name="lnorm_highlow_nosampl_noparamtrans",
            number=13,
            description="lnorm_highlow_nosampl_noparam",
            preprocess_steps=[
                PreprocessStep(name="highlow_nosampl_loudnorm", enabled=True, step_type="highlow_nosampl_loudnorm"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 14: Aresample to 16kHz + loudness normalization (fixed params) + minimal params
        Variant(
            name="lnorm2_aresampl_noparamtrans",
            number=14,
            description="lnorm2_aresampl_noparam",
            preprocess_steps=[
                PreprocessStep(name="aresampl_loudnorm_fixed", enabled=True, step_type="aresampl_loudnorm_fixed"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 15: Aresample to 16kHz + loudness normalization (I=-24, LRA=15) + minimal params
        Variant(
            name="lnorm3_aresampl_noparamtrans",
            number=15,
            description="lnorm3_aresampl_noparam",
            preprocess_steps=[
                PreprocessStep(name="aresampl_loudnorm_fixed2", enabled=True, step_type="aresampl_loudnorm_fixed2"),
            ],
            transcription_preset="minimal",
        ),
        # Variant 16: 2-pass loudnorm in linear mode (file-wide gain change) + minimal params
        Variant(
            name="loudnorm_2pass_linear_noparamtrans",
            number=16,
            description="lnorm2p_linear_noparam",
            preprocess_steps=[
                PreprocessStep(name="loudnorm_2pass_linear", enabled=True, step_type="loudnorm_2pass_linear"),
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
        number: Variant number (1-16)

    Returns:
        Variant instance or None if not found
    """
    for variant in get_builtin_variants():
        if variant.number == number:
            return variant
    return None
