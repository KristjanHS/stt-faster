"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.variant import PreprocessStep, Variant


def _get_custom_preprocess_functions():
    """Import custom preprocessing functions from compare_transcription_variants script.

    These are used for variants 10-15 that require custom preprocessing logic.
    Uses importlib to safely import from the script file.
    """
    import importlib.util
    from pathlib import Path

    # Get the scripts directory
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    script_path = scripts_dir / "compare_transcription_variants.py"

    if not script_path.exists():
        raise ImportError(f"Could not find compare_transcription_variants.py at {script_path}")

    # Use importlib to load the module from the file path
    spec = importlib.util.spec_from_file_location("compare_transcription_variants", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {script_path}")

    module = importlib.util.module_from_spec(spec)
    # Execute the module (this will define all the functions)
    spec.loader.exec_module(module)

    return {
        "loudnorm_with_highpass": module.preprocess_loudnorm_with_highpass,
        "dynaudnorm_only": module.preprocess_dynaudnorm_only,
        "highlow_aform_loudnorm": module.preprocess_highlow_aform_loudnorm,
        "highlow_nosampl_loudnorm": module.preprocess_highlow_nosampl_loudnorm,
        "aresampl_loudnorm_fixed": module.preprocess_aresampl_loudnorm_fixed,
        "aresampl_loudnorm_fixed2": module.preprocess_aresampl_loudnorm_fixed2,
    }


def get_builtin_variants() -> list[Variant]:
    """Return list of all built-in variants (1-15).

    These match the variants currently defined in compare_transcription_variants.py.
    Variants 10-15 use custom preprocessing functions.
    """
    # Get custom preprocessing functions for variants 10-15
    custom_funcs = _get_custom_preprocess_functions()

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
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["loudnorm_with_highpass"],
        ),
        # Variant 11: Dynamic audio normalization + minimal params
        Variant(
            name="norm_dynaud_noparamtrans",
            number=11,
            description="norm_dynaud_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["dynaudnorm_only"],
        ),
        # Variant 12: Highpass + lowpass + aformat + loudness normalization + minimal params
        Variant(
            name="lnorm_highlow_aform_noparamtrans",
            number=12,
            description="lnorm_highlow_aform_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["highlow_aform_loudnorm"],
        ),
        # Variant 13: Highpass + lowpass + loudness normalization (no aformat) + minimal params
        Variant(
            name="lnorm_highlow_nosampl_noparamtrans",
            number=13,
            description="lnorm_highlow_nosampl_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["highlow_nosampl_loudnorm"],
        ),
        # Variant 14: Aresample to 16kHz + loudness normalization (fixed params) + minimal params
        Variant(
            name="lnorm2_aresampl_noparamtrans",
            number=14,
            description="lnorm2_aresampl_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["aresampl_loudnorm_fixed"],
        ),
        # Variant 15: Aresample to 16kHz + loudness normalization (I=-24, LRA=15) + minimal params
        Variant(
            name="lnorm3_aresampl_noparamtrans",
            number=15,
            description="lnorm3_aresampl_noparam",
            preprocess_steps=[],
            transcription_preset="minimal",
            custom_preprocess_runner=custom_funcs["aresampl_loudnorm_fixed2"],
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
        number: Variant number (1-15)

    Returns:
        Variant instance or None if not found
    """
    for variant in get_builtin_variants():
        if variant.number == number:
            return variant
    return None
