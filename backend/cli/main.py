"""Unified CLI entrypoint for stt-faster."""

from __future__ import annotations

from typing import NoReturn

import typer

from backend.cli import db, transcription_commands

app = typer.Typer(
    name="stt-faster",
    help="Fast audio transcription tool using Whisper models",
    add_completion=False,
)

# Add subcommands
app.add_typer(transcription_commands.app, name="transcribe")
app.add_typer(db.app, name="db")


def main() -> NoReturn:
    """Main entrypoint for stt-faster CLI."""
    app()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
