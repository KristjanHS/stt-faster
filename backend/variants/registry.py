"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.variant import PreprocessStep, Variant


def _get_all_variants() -> list[Variant]:
    """Return all defined variants (both active and inactive).

    This is an internal function that defines all variants.
    Use get_builtin_variants() to get only active variants.
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
                PreprocessStep(
                    name="loudnorm_2pass_linear",
                    enabled=True,
                    step_type="loudnorm_2pass_linear",
                    config={"I": -24.0, "TP": -2.0, "LRA": 15.0},
                ),
            ],
            transcription_preset="minimal",
        ),
        # Variant 17: No preprocessing + minimal params + no_speech_threshold=0.5
        Variant(
            name="noprep_noparamtrans_thres1",
            number=17,
            description="nopre_thres1",
            preprocess_steps=[],
            transcription_preset="minimal",
            transcription_overrides={"no_speech_threshold": 0.5},
        ),
        # Variant 18: No preprocessing + minimal params + chunk_length=20
        Variant(
            name="noprep_noparamtrans_chunk1",
            number=18,
            description="nopre_chunk1",
            preprocess_steps=[],
            transcription_preset="minimal",
            transcription_overrides={"chunk_length": 20},
        ),
        # Variant 19: No preprocessing + minimal params + condition_on_previous_text=False
        Variant(
            name="noprep_noparamtrans_condprev1",
            number=19,
            description="nopre_condprev1",
            preprocess_steps=[],
            transcription_preset="minimal",
            transcription_overrides={"condition_on_previous_text": False},
        ),
    ]


def get_builtin_variants() -> list[Variant]:
    """Return list of active built-in variants.

    Currently only variants 7, 17, 18, 19 are enabled.
    All other variants are disabled but remain defined and can be accessed
    via get_variant_by_number() or get_variant_by_name().
    All variants now use declarative preprocessing steps.
    """
    all_variants = _get_all_variants()
    # Only return active variants: 7, 17, 18, 19
    active_variant_numbers = {7, 17, 18, 19}
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
        number: Variant number (1-19)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None
