"""Built-in variant registry."""

from __future__ import annotations

from backend.variants.steps import (
    AresamplLoudnormFixed2StepConfig,
    PeakNormalize2passStepConfig,
    SoxPeakNormalizeStepConfig,
    VolumeLimiterStepConfig,
)
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
      - 42: Variant 36 + patience=1.2
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
        # Group 4: Decode quality and silence guard variants - Variants 12-21
        # Group A — Improve Variant 2 accuracy without changing behavior much (decode quality)
        # Variant 12: Beam 6
        Variant(
            name="decode_beam6",
            number=12,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
            ),
        ),
        # Variant 13: Beam 7
        Variant(
            name="decode_beam7",
            number=13,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=7,
            ),
        ),
        # Variant 14: Beam 6 + patience 1.2
        Variant(
            name="decode_beam6_pat12",
            number=14,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                patience=1.2,
            ),
        ),
        # Variant 15: Beam 7 + patience 1.2
        Variant(
            name="decode_beam7_pat12",
            number=15,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=7,
                patience=1.2,
            ),
        ),
        # Group B — Add Variant 5's silence guard, but ultra conservative
        # Variant 16: Guard-lite + Beam 6
        Variant(
            name="guardlite_beam6",
            number=16,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                no_speech_threshold=0.95,
                logprob_threshold=-0.46,
            ),
        ),
        # Variant 17: Guard-lite + Beam 7
        Variant(
            name="guardlite_beam7",
            number=17,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=7,
                no_speech_threshold=0.95,
                logprob_threshold=-0.46,
            ),
        ),
        # Variant 18: Guard-lite + Beam 6 + patience 1.2
        Variant(
            name="guardlite_beam6_pat12",
            number=18,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                patience=1.2,
                no_speech_threshold=0.95,
                logprob_threshold=-0.46,
            ),
        ),
        # Group C — Add Variant 21's "no context carryover", but stabilize it
        # Variant 19: No-context + Beam 6 + patience 1.2
        Variant(
            name="noctx_beam6_pat12",
            number=19,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                patience=1.2,
                condition_on_previous_text=False,
            ),
        ),
        # Variant 20: No-context + Guard-lite + Beam 6 + patience 1.2
        Variant(
            name="noctx_guardlite_beam6_pat12",
            number=20,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                patience=1.2,
                condition_on_previous_text=False,
                no_speech_threshold=0.95,
                logprob_threshold=-0.46,
            ),
        ),
        # Group D — Boundary effects (quiet speakers at chunk edges)
        # Variant 21: Guard-lite + Beam 6 + patience 1.2 + chunk 25
        Variant(
            name="guardlite_beam6_pat12_chunk25",
            number=21,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=6,
                patience=1.2,
                chunk_length=25,
                no_speech_threshold=0.95,
                logprob_threshold=-0.46,
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
                    config=AresamplLoudnormFixed2StepConfig(
                        integrated_loudness=-24.0, true_peak=-2.0, loudness_range=15.0
                    ),
                ),
            ],
            transcription_config=create_minimal_config(),
        ),
        # Group 5: Conservative level/volume tweaks - Variants 23-32
        # Variant 23: P0 — Control (no level change, just resample/downmix if needed)
        Variant(
            name="level_p0_control",
            number=23,
            preprocess_steps=[
                PreprocessStep(
                    name="resample_only",
                    enabled=True,
                    step_type="resample",
                    config=None,
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 24: P1 — Limiter only
        Variant(
            name="level_p1_limiter",
            number=24,
            preprocess_steps=[
                PreprocessStep(
                    name="limiter_only",
                    enabled=True,
                    step_type="limiter_only",
                    config=None,
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 25: P2 — Tiny fixed gain (+1.5 dB) + limiter
        Variant(
            name="level_p2_volume_1_5db",
            number=25,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_1_5db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=1.5),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 26: P3 — Small fixed gain (+3 dB) + limiter
        Variant(
            name="level_p3_volume_3db",
            number=26,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_3db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=3.0),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 27: P4 — Peak-normalize to -6 dBFS (two-pass, only if truly quiet)
        Variant(
            name="level_p4_peaknorm_n6db",
            number=27,
            preprocess_steps=[
                PreprocessStep(
                    name="peak_normalize_n6db",
                    enabled=True,
                    step_type="peak_normalize_2pass",
                    config=PeakNormalize2passStepConfig(target_db=-6.0, max_gain_db=6.0),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 28: P5 — Peak-normalize to -3 dBFS
        Variant(
            name="level_p5_peaknorm_n3db",
            number=28,
            preprocess_steps=[
                PreprocessStep(
                    name="peak_normalize_n3db",
                    enabled=True,
                    step_type="peak_normalize_2pass",
                    config=PeakNormalize2passStepConfig(target_db=-3.0, max_gain_db=6.0),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 29: P6 — Peak-normalize to -3 dBFS, but cap gain harder (+3 dB max)
        Variant(
            name="level_p6_peaknorm_n3db_max3db",
            number=29,
            preprocess_steps=[
                PreprocessStep(
                    name="peak_normalize_n3db_max3db",
                    enabled=True,
                    step_type="peak_normalize_2pass",
                    config=PeakNormalize2passStepConfig(target_db=-3.0, max_gain_db=3.0),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 30: P7 — SoX peak normalize to -3 dBFS
        Variant(
            name="level_p7_sox_peaknorm_n3db",
            number=30,
            preprocess_steps=[
                PreprocessStep(
                    name="sox_peak_normalize_n3db",
                    enabled=True,
                    step_type="sox_peak_normalize",
                    config=SoxPeakNormalizeStepConfig(target_db=-3.0),
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 31: P8 — Super-gentle compressor + tiny makeup + limiter
        Variant(
            name="level_p8_compressor",
            number=31,
            preprocess_steps=[
                PreprocessStep(
                    name="compressor_limiter",
                    enabled=True,
                    step_type="compressor_limiter",
                    config=None,
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 32: P9 — Super-gentle dynaudnorm (ultra conservative) + limiter
        Variant(
            name="level_p9_dynaudnorm",
            number=32,
            preprocess_steps=[
                PreprocessStep(
                    name="dynaudnorm_conservative",
                    enabled=True,
                    step_type="dynaudnorm_conservative",
                    config=None,
                ),
            ],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
            ),
        ),
        # Variant 33: Same as variant 2 but with word_timestamps=True
        Variant(
            name="baseline_no_vad_word_timestamps",
            number=33,
            preprocess_steps=[],
            transcription_config=(
                lambda: (
                    config := create_no_vad_baseline_config(),
                    config.set("word_timestamps", True),
                    config,
                )[2]
            )(),
        ),
        # Group 6: Variant 1 derivatives (best-bet combinations) - Variants 34-38
        # Variant 34: Variant 1 + word timestamps
        # Keep the winning baseline behavior, but flip on word timestamps
        Variant(
            name="baseline_word_timestamps",
            number=34,
            preprocess_steps=[],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("word_timestamps", True),
                    config,
                )[2]
            )(),
        ),
        # Variant 35: Variant 1 + Beam 7
        # Apply beam=7 to the baseline (with VAD still on) to see if it beats Variant 1
        Variant(
            name="baseline_beam7",
            number=35,
            preprocess_steps=[],
            transcription_config=(
                lambda: (
                    config := create_baseline_config(),
                    config.set("beam_size", 7),
                    config,
                )[2]
            )(),
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
        # Variant 37: Variant 1 + P8 (compressor+limiter)
        # Combine P8 compressor with baseline transcription to isolate preprocessing impact
        Variant(
            name="baseline_p8_compressor",
            number=37,
            preprocess_steps=[
                PreprocessStep(
                    name="compressor_limiter",
                    enabled=True,
                    step_type="compressor_limiter",
                    config=None,
                ),
            ],
            transcription_config=create_baseline_config(),
        ),
        # Variant 38: Beam 7 + Guard-lite (softened)
        # More realistic combo: beam=7 + patience≈1.2 + chunk≈25 + no_speech_threshold ~0.8–0.9
        # + logprob_threshold ~-0.6 to -0.45 (still guardy, less "mute button")
        Variant(
            name="beam7_guardlite_softened",
            number=38,
            preprocess_steps=[],
            transcription_config=create_minimal_config(
                vad_filter=False,
                task="transcribe",
                word_timestamps=True,
                beam_size=7,
                patience=1.2,
                chunk_length=25,
                no_speech_threshold=0.85,  # Middle of 0.8-0.9 range
                logprob_threshold=-0.525,  # Middle of -0.6 to -0.45 range
            ),
        ),
        # Group 7: Variant 36 derivatives (P2 preprocessing with parameter tweaks) - Variants 39-46
        # Variant 39: Variant 36 + beam_size=6
        Variant(
            name="p2_volume_1_5db_beam6",
            number=39,
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
                    config.set("beam_size", 6),
                    config,
                )[2]
            )(),
        ),
        # Variant 40: Variant 36 + beam_size=7
        Variant(
            name="p2_volume_1_5db_beam7",
            number=40,
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
                    config,
                )[2]
            )(),
        ),
        # Variant 41: Variant 36 + volume_db=2.0
        Variant(
            name="p2_volume_2_0db",
            number=41,
            preprocess_steps=[
                PreprocessStep(
                    name="volume_2_0db_limiter",
                    enabled=True,
                    step_type="volume_limiter",
                    config=VolumeLimiterStepConfig(volume_db=2.0),
                ),
            ],
            transcription_config=create_baseline_config(),
        ),
        # Variant 42: Variant 36 + patience=1.15
        Variant(
            name="p2_volume_1_5db_pat115",
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
                    config.set("patience", 1.15),
                    config,
                )[2]
            )(),
        ),
        # Variant 43: Variant 36 + beam_size=6 + patience=1.15
        Variant(
            name="p2_volume_1_5db_beam6_pat115",
            number=43,
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
                    config.set("beam_size", 6),
                    config.set("patience", 1.15),
                    config,
                )[3]
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
        # Variant 45: Variant 36 + chunk_length=25
        Variant(
            name="p2_volume_1_5db_chunk25",
            number=45,
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
                    config.set("chunk_length", 25),
                    config,
                )[2]
            )(),
        ),
        # Variant 46: Variant 36 + word_timestamps=True
        Variant(
            name="p2_volume_1_5db_word_timestamps",
            number=46,
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
                    config.set("word_timestamps", True),
                    config,
                )[2]
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
    # Active variants: 44, and 47-51
    # active_variant_numbers = {1} | {36} | {44} | set(range(47, 52))
    active_variant_numbers = set(range(1, 52))
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
