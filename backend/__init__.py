from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Keep imports lazy so lightweight entrypoints (e.g., Docker healthchecks) stay fast.
__all__ = ["transcribe", "transcribe_to_json", "pick_model"]

if TYPE_CHECKING:
    from backend.transcribe import pick_model, transcribe, transcribe_to_json


def __getattr__(name: str) -> Any:
    if name in __all__:
        from backend import transcribe as transcribe_module

        return getattr(transcribe_module, name)
    raise AttributeError(f"module 'backend' has no attribute '{name}'")
