"""Variant system for flexible transcription pipeline configurations."""

from backend.variants.executor import execute_variant
from backend.variants.registry import get_builtin_variants

__all__ = [
    "execute_variant",
    "get_builtin_variants",
]
