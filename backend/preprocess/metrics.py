from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast


@dataclass(slots=True)
class StepMetrics:
    name: str
    backend: str
    duration: float


@dataclass(slots=True)
class PreprocessMetrics:
    total_duration: float
    steps: list[StepMetrics] = field(
        default_factory=lambda: cast(list[StepMetrics], []),
    )
