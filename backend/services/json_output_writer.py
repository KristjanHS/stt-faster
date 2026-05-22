"""Concrete implementation of OutputWriter for JSON/text output."""

import json
from pathlib import Path
from typing import Any

from backend.transcribe import format_segments_as_text


class JsonOutputWriter:
    """Concrete implementation for writing transcription output."""

    def write(self, output_path: str, payload: dict[str, Any], format: str) -> list[Path]:
        """Write transcription output in specified format. Returns list of created files."""
        created_files: list[Path] = []
        base_path = Path(output_path)

        if format in ("txt", "both"):
            txt_path = base_path.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            segments: list[dict[str, Any]] = payload.get("segments", [])
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(format_segments_as_text(segments))
            created_files.append(txt_path)

        if format in ("json", "both"):
            json_path = base_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            created_files.append(json_path)

        return created_files
