"""UI utilities for CLI commands using Rich."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_config_table(
    input_folder: str, preset: str, variant_number: int, variant_name: str, language: str | None, output_format: str
) -> None:
    """Display transcription configuration using Rich.

    Args:
        input_folder: Path to input folder
        preset: Model preset name
        variant_number: Variant number
        variant_name: Variant name
        language: Optional language code
        output_format: Output format
    """
    config_table = Table.grid(padding=(0, 2))
    config_table.add_row("[bold]Input folder:[/bold]", input_folder)
    config_table.add_row("[bold]Model preset:[/bold]", preset)
    config_table.add_row("[bold]Variant:[/bold]", f"{variant_number}: {variant_name}")
    if language:
        config_table.add_row("[bold]Language:[/bold]", language)
    config_table.add_row("[bold]Output format:[/bold]", output_format)

    console.print("\n[bold]Transcription Configuration[/bold]")
    console.print(Panel(config_table, border_style="blue", padding=(0, 1)))
    console.print()


def display_processing_summary(results: dict[str, Any]) -> None:
    """Display processing summary using Rich.

    Args:
        results: Results dictionary from processor.process_folder()
    """
    console.print()
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_row("[bold]Files found:[/bold]", str(results.get("files_found", 0)))
    summary_table.add_row("[bold]Successfully processed:[/bold]", f"[green]{results.get('succeeded', 0)}[/green]")
    summary_table.add_row("[bold]Failed:[/bold]", f"[red]{results.get('failed', 0)}[/red]")

    console.print("[bold]Processing Summary[/bold]")
    console.print(Panel(summary_table, border_style="green", padding=(0, 1)))


def display_run_statistics(run_stats: dict[str, Any]) -> None:
    """Display run statistics using Rich.

    Args:
        run_stats: Run statistics dictionary
    """
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_row(
        "[bold]Total time:[/bold]",
        f"{run_stats.get('total_processing_time', 0):.2f} s",
    )
    stats_table.add_row(
        "[bold]Preprocessing:[/bold]",
        f"{run_stats.get('total_preprocess_time', 0):.2f} s",
    )
    stats_table.add_row(
        "[bold]Transcription:[/bold]",
        f"{run_stats.get('total_transcribe_time', 0):.2f} s",
    )
    stats_table.add_row(
        "[bold]Average speed:[/bold]",
        f"{float(run_stats.get('average_speed_ratio') or 0):.2f}x realtime",
    )
    if run_stats.get("detected_languages"):
        stats_table.add_row(
            "[bold]Detected languages:[/bold]",
            ", ".join(run_stats["detected_languages"]),
        )

    console.print()
    console.print("[bold]Run Statistics[/bold]")
    console.print(Panel(stats_table, border_style="cyan", padding=(0, 1)))


def display_multi_variant_summary(
    variants_count: int, total_succeeded: int, total_failed: int, output_root: str
) -> None:
    """Display overall summary for multi-variant runs.

    Args:
        variants_count: Number of variants processed
        total_succeeded: Total successful file processings
        total_failed: Total failed file processings
        output_root: Output directory path
    """
    console.print("\n[bold]Overall Summary[/bold]")
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_row("[bold]Total variants:[/bold]", str(variants_count))
    summary_table.add_row("[bold]Total succeeded:[/bold]", f"[green]{total_succeeded}[/green]")
    summary_table.add_row("[bold]Total failed:[/bold]", f"[red]{total_failed}[/red]")
    summary_table.add_row("[bold]Output directory:[/bold]", output_root)
    console.print(Panel(summary_table, border_style="green", padding=(0, 1)))
