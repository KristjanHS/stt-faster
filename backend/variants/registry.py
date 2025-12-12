"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.transcription_presets import (
    create_baseline_config,
    create_minimal_config,
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
    - Variants 12-21: Hybrid variants (2 + 5 + 21 combinations)
      - 12-15: Variant 5 baseline with more decoding power (beam 6/7, patience)
      - 16-18: Variant 21 baseline (no context) with more decoding power
      - 19-21: Gentler silence gates for quiet speech recovery
    - Variant 22: Complex normalization
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
        # Group 2: No preprocessing, diagnostic/config tuning - Variants 3-6
        # Variant 3: Control for diagnostic sweep (vad_filter=False, chunk_length=20, beam_size=5)
        Variant(
            name="diag_control_no_vad_chunk20_beam5",
            number=3,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
            ),
        ),
        # Keep diag control fixed: vad_filter=False, chunk_length=20, beam_size=5, condition_on_previous_text=True
        # Variant 4: Target both silence segments hard (ns=0.90, lp=-0.45)
        Variant(
            name="diag_ns_090_lp_045",
            number=4,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.45,
            ),
        ),
        # Variant 5: Slightly safer (less aggressive on logprob) (ns=0.90, lp=-0.50)
        Variant(
            name="diag_ns_090_lp_050",
            number=5,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 6: More selective using higher ns (ns=0.93, lp=-0.45)
        Variant(
            name="diag_ns_093_lp_045",
            number=6,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                no_speech_threshold=0.93,
                logprob_threshold=-0.45,
            ),
        ),
        # Group 3: No preprocessing, production VAD tuning - Variants 7-11
        # Production regime: vad_filter=True, chunk_length=20, beam_size=5,
        # word_timestamps=True, condition_on_previous_text=True
        # Variant 7: More permissive VAD threshold + moderate padding
        Variant(
            name="prod_vad_thr035_sil800_pad300",
            number=7,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_silence_duration_ms=800,
                    speech_pad_ms=300,
                ),
            ),
        ),
        # Variant 8: Even more permissive (for very quiet/far speakers)
        Variant(
            name="prod_vad_thr025_sil600_pad400",
            number=8,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_parameters=dict(
                    threshold=0.25,
                    min_silence_duration_ms=600,
                    speech_pad_ms=400,
                ),
            ),
        ),
        # Variant 9: Catch short utterances ("yes", "mh-mh") from far away
        Variant(
            name="prod_vad_thr035_minspeech150_sil800_pad300",
            number=9,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=800,
                    speech_pad_ms=300,
                ),
            ),
        ),
        # Variant 10: Same threshold but much more context padding
        Variant(
            name="prod_vad_thr035_sil800_pad500",
            number=10,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_silence_duration_ms=800,
                    speech_pad_ms=500,
                ),
            ),
        ),
        # Variant 11: Control: keep VAD permissive, but turn off context carryover
        Variant(
            name="prod_vad_thr035_sil800_pad300_noctx",
            number=11,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=True,
                chunk_length=20,
                beam_size=5,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_silence_duration_ms=800,
                    speech_pad_ms=300,
                ),
            ),
        ),
        # Group 4: Hybrid variants (2 + 5 + 21 combinations) - Variants 12-21
        # A) Variant 5 baseline, but more decoding power (aim: fewer word mistakes like v2)
        # Variant 12: Variant 5 + beam 6
        Variant(
            name="hyb_v5_beam6",
            number=12,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 13: Variant 5 + beam 7
        Variant(
            name="hyb_v5_beam7",
            number=13,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=7,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 14: Variant 5 + beam 6 + patience 1.2
        Variant(
            name="hyb_v5_beam6_pat12",
            number=14,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                patience=1.2,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 15: Variant 5 + beam 7 + patience 1.2
        Variant(
            name="hyb_v5_beam7_pat12",
            number=15,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=7,
                patience=1.2,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # B) Variant 21 baseline (no context), but more decoding power (aim: keep recall without extra errors)
        # Variant 16: No context + beam 6
        Variant(
            name="hyb_v21_noctx_beam6",
            number=16,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                condition_on_previous_text=False,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 17: No context + beam 7
        Variant(
            name="hyb_v21_noctx_beam7",
            number=17,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=7,
                condition_on_previous_text=False,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 18: No context + beam 6 + patience 1.2
        Variant(
            name="hyb_v21_noctx_beam6_pat12",
            number=18,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                patience=1.2,
                condition_on_previous_text=False,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.50,
            ),
        ),
        # C) Slightly gentler silence gate (aim: recover quiet speech if v5 is dropping it)
        # Variant 19: Beam 6 + ns 0.88 + lp -0.50
        Variant(
            name="hyb_gate_ns088_lp050_beam6",
            number=19,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.88,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 20: Beam 6 + ns 0.90 + lp -0.55 (more conservative skip; may keep quiet speech)
        Variant(
            name="hyb_gate_ns090_lp055_beam6",
            number=20,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                condition_on_previous_text=True,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.90,
                logprob_threshold=-0.55,
            ),
        ),
        # Variant 21: No context + beam 6 + ns 0.88 + lp -0.50
        Variant(
            name="hyb_noctx_gate_ns088_lp050_beam6",
            number=21,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                beam_size=6,
                condition_on_previous_text=False,
                vad_filter=False,
                chunk_length=20,
                word_timestamps=True,
                no_speech_threshold=0.88,
                logprob_threshold=-0.50,
            ),
        ),
        # Variant 22: Aresample to 16kHz + loudness normalization (I=-24, LRA=15) + minimal params
        Variant(
            name="lnorm3_aresampl_noparamtrans",
            number=22,
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
    # Only return active variants for batch jobs:
    # Variants 2, 5: baseline_no_vad, diagnostic/config tuning
    # Variants 12-21: Hybrid variants (2 + 5 + 21 combinations with more decoding power and gentler gates)
    active_variant_numbers = {2, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
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
        number: Variant number (1-22)

    Returns:
        Variant instance or None if not found
    """
    for variant in _get_all_variants():
        if variant.number == number:
            return variant
    return None
