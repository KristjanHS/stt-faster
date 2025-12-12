"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.transcription_presets import (
    create_baseline_config,
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
    - Variants 90-93: Validation sweep variants (diagnostic, vad_filter=False)
    - Variants 94-96: Production candidate variants (no_speech_threshold + logprob_threshold tuning)
    """
    return [
        # Group 1: No preprocessing (simplest) - Variants 1-7
        # Variant 1: No preprocessing + raw decode baseline (no VAD by design)
        Variant(
            name="baseline_no_vad_defaults",
            number=1,
            preprocess_steps=[],
            transcription_config=create_baseline_config(vad_filter=False),
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
        # Group 7: Validation sweep variants (diagnostic, vad_filter=False) - Variants 90-93
        # Variant 90: Control for diagnostic sweep (vad_filter=False, chunk_length=20, beam_size=5)
        Variant(
            name="diag_control_no_vad_chunk20_beam5",
            number=90,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
            ),
        ),
        # Variant 91: Conservative - catches avg_lp≈-0.79 (no_speech_threshold=0.75, logprob_threshold=-0.75)
        Variant(
            name="diag_ns_075_lp_075",
            number=91,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.75,
                logprob_threshold=-0.75,
            ),
        ),
        # Variant 92: Balanced - catches avg_lp≈-0.79 (no_speech_threshold=0.75, logprob_threshold=-0.7)
        # Isolates lp effect by keeping ns=0.75 same as variant 91
        Variant(
            name="diag_ns_075_lp_070",
            number=92,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.75,
                logprob_threshold=-0.7,
            ),
        ),
        # Variant 93: More active - catches avg_lp≈-0.79 with lower ns
        # (no_speech_threshold=0.65, logprob_threshold=-0.7)
        Variant(
            name="diag_ns_065_lp_070",
            number=93,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.65,
                logprob_threshold=-0.7,
            ),
        ),
        # Group 8: Production candidate variants - Variants 94-96
        # Production regime: vad_filter=True, chunk_length=20, beam_size=5
        # Variant 94: Safest - almost only skips truly silent windows
        Variant(
            name="prod_ns_075_lp_080",
            number=94,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.75,
                logprob_threshold=-0.8,
            ),
        ),
        # Variant 95: Balanced hallucination guard
        Variant(
            name="prod_ns_070_lp_070",
            number=95,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.70,
                logprob_threshold=-0.7,
            ),
        ),
        # Variant 96: Only if you still see hallucinations in silence
        Variant(
            name="prod_ns_065_lp_070",
            number=96,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                no_speech_threshold=0.65,
                logprob_threshold=-0.7,
            ),
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
    # 1 (baseline_no_vad_defaults), 4 (noprep_minimal_no_speech_threshold), 7 (noprep_minimal_beam_size)
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
        number: Variant number (1-96)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None


def get_conservative_sweep_variants() -> list[int]:
    """Return variant numbers for conservative sweep.

    Conservative sweep includes:
    - Variant 1: Baseline (baseline_no_vad_defaults)
    - Variant 6: condition_on_previous_text=False
    - Variants 21-26: Conservative sweep variants (no preprocessing, minimal config with small deltas)

    Returns:
        List of variant numbers for conservative sweep
    """
    return [1, 6, 21, 22, 23, 24, 25, 26]
