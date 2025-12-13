"""Concrete implementation of FileMover using shutil."""

import shutil


class ShutilFileMover:
    """Concrete implementation using shutil."""

    def move(self, source: str, destination: str) -> str | None:
        return shutil.move(source, destination)
