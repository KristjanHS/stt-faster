"""Variant system for flexible transcription pipeline configurations."""

from backend.variants.executor import (
    create_variant_preprocess_runner,
    create_variant_transcribe_config,
    execute_variant,
)
from backend.variants.registry import (
    get_all_variants,
    get_builtin_variants,
    get_variant_by_name,
    get_variant_by_number,
)
from backend.variants.variant import PreprocessStep, Variant

__all__ = [
    "Variant",
    "PreprocessStep",
    "execute_variant",
    "get_all_variants",
    "get_builtin_variants",
    "get_variant_by_name",
    "get_variant_by_number",
    "create_variant_preprocess_runner",
    "create_variant_transcribe_config",
]
