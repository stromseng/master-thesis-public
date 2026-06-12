"""Convert PDF files to markdown using Marker.

This script converts PDFs from data/rag/raw/ to markdown files in data/rag/processed/.

Copy processed files back to local machine (run from repo root):
    cd <repo-root>
    mkdir -p data/rag
    scp -r idun:repos/master-thesis/data/rag/processed ./data/rag/
"""

from __future__ import annotations

import os
import sys

# Suppress warnings on HPC clusters and prevent threading issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_DISABLE_THREAD_AFFINITY", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

if sys.platform != "darwin":
    try:
        import onnxruntime

        onnxruntime.set_default_logger_severity(4)
    except ImportError:
        pass

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

# File prefixes to exclude from conversion (uses startswith matching)
# These are the 14 files from git status (M=modified, A=added)
EXCLUDE_PREFIXES: list[str] = [
    # "Admiralty manual of navigation",
    # "COLREG-Consolidated-2018",
    # "GuideCollisionAvoidanceRules",
    # "H D MCGEORGE - Marine Auxiliary Machinery",
    # "IMDG code _ International Maritime Dangerous Goods Code",
    # "IMO - Standard Marine Communication Phrases",
    # "IMSBC Code _ International Maritime Solid Bulk Cargoes Code",
    # "International Maritime Organisation_ Life-Safing Appliances",
    # "International Maritime Organisation_ MARPOL_ Consolidated",
    # "International Maritime Organisation_ STCW",
    # "International Safety Management Code _ ISM Code",
    # "Marine Electrical Equipment and Practice",
    # "Ship Construction, Seventh Edition",
    # "Ship Stability for Masters and Mates",
    # "SOLAS",
    # "Pounder",
    # "Nicholls",
    # "Radar and ARPA Manual",
]


def _matches_exclude_prefix(stem: str) -> bool:
    """Check if a file stem matches any exclusion prefix."""
    return any(stem.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def _detect_accelerator() -> str:
    """Detect the best available accelerator."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def convert_pdfs(
    input_dir: Path,
    output_dir: Path,
    accelerator: str | None = None,
    force: bool = False,
    test: bool = False,
) -> None:
    """Convert all PDFs in input_dir to markdown in output_dir using Marker."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator is None:
        accelerator = _detect_accelerator()

    # Print GPU diagnostics for CUDA
    if accelerator == "cuda":
        try:
            import torch

            console.print(f"[blue]PyTorch version: {torch.__version__}[/blue]")
            console.print(f"[blue]CUDA available: {torch.cuda.is_available()}[/blue]")
            if torch.cuda.is_available():
                console.print(f"[blue]CUDA version: {torch.version.cuda}[/blue]")
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                console.print(f"[green]GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)[/green]")
        except ImportError:
            console.print("[yellow]PyTorch not available for GPU diagnostics[/yellow]")

    pdfs = sorted(input_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
    console.print(f"[blue]Input directory: {input_dir}[/blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    console.print(f"[blue]Total PDFs found: {len(pdfs)}[/blue]")

    if not pdfs:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return

    # Test mode: select only the smallest PDF
    if test:
        smallest_pdf = min(pdfs, key=lambda p: p.stat().st_size)
        size_mb = smallest_pdf.stat().st_size / (1024 * 1024)
        console.print("[magenta]Test mode: selecting smallest PDF[/magenta]")
        console.print(f"[magenta]  {smallest_pdf.name} ({size_mb:.2f} MB)[/magenta]")
        pdfs = [smallest_pdf]
        force = True

    # Filter out excluded files using prefix matching
    if EXCLUDE_PREFIXES:
        console.print(
            f"[blue]Exclusion list has {len(EXCLUDE_PREFIXES)} prefixes[/blue]"
        )
        excluded = [p for p in pdfs if _matches_exclude_prefix(p.stem)]
        not_excluded = [p for p in pdfs if not _matches_exclude_prefix(p.stem)]

        # Debug: show which files matched
        if excluded:
            console.print(f"[yellow]Excluding {len(excluded)} files:[/yellow]")
            for p in excluded:
                console.print(f"[yellow]  - {p.stem[:80]}...[/yellow]")

        pdfs = not_excluded

    # Check which files need conversion
    to_convert = []
    skipped = []
    for pdf_path in pdfs:
        md_output_path = output_dir / f"{pdf_path.stem}.md"
        if md_output_path.exists() and not force:
            skipped.append(pdf_path)
        else:
            to_convert.append(pdf_path)

    # Log status
    accel_colors = {"cuda": "green", "cpu": "yellow"}
    color = accel_colors.get(accelerator, "yellow")
    console.print(f"[blue]Accelerator: [{color}]{accelerator}[/{color}][/blue]")
    console.print(f"[blue]After filtering: {len(pdfs)} PDF files[/blue]")
    console.print(f"[blue]  - To convert: {len(to_convert)}[/blue]")
    console.print(f"[blue]  - Skipping (already exist): {len(skipped)}[/blue]")

    if to_convert:
        console.print("[blue]Files to convert (sorted by size):[/blue]")
        for i, p in enumerate(to_convert, 1):
            size_mb = p.stat().st_size / (1024 * 1024)
            console.print(f"[blue]  {i}. {p.name[:60]}... ({size_mb:.1f} MB)[/blue]")

    if not to_convert:
        console.print(
            "[green]All files already converted. Use force=True to reconvert.[/green]"
        )
        return

    # Initialize Marker converter
    console.print("[blue]Initializing Marker converter...[/blue]")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    artifact_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=artifact_dict)
    console.print("[green]Marker converter initialized[/green]")

    succeeded = []
    failed = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting PDFs", total=len(to_convert))

        for idx, pdf_path in enumerate(to_convert, 1):
            output_path = output_dir / f"{pdf_path.stem}.md"
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            console.print(
                f"[yellow][{idx}/{len(to_convert)}] Converting {pdf_path.name[:60]}... ({size_mb:.1f} MB)[/yellow]"
            )

            try:
                # Convert using Marker
                console.print("[dim]  Calling converter...[/dim]")
                rendered = converter(str(pdf_path))
                console.print("[dim]  Extracting text...[/dim]")
                markdown, _, _ = text_from_rendered(rendered)

                console.print(
                    f"[dim]  Writing {len(markdown)} chars to {output_path.name}[/dim]"
                )
                output_path.write_text(markdown, encoding="utf-8")
                console.print(f"[green]  Saved {output_path.name}[/green]")
                succeeded.append(pdf_path)
            except Exception as e:
                console.print(f"[red]Error converting {pdf_path.name}: {e}[/red]")
                import traceback

                traceback.print_exc()
                console.print("[yellow]  Continuing with next file...[/yellow]")
                failed.append((pdf_path, str(e)))

            progress.advance(task)

    # Summary
    console.print("\n[bold]===== CONVERSION SUMMARY =====[/bold]")
    console.print(f"[green]Succeeded: {len(succeeded)}[/green]")
    console.print(f"[red]Failed: {len(failed)}[/red]")
    if failed:
        console.print("[red]Failed files:[/red]")
        for p, err in failed:
            console.print(f"[red]  - {p.name[:60]}...: {err[:50]}[/red]")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDFs to markdown using Marker"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only convert the smallest PDF file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if output file exists",
    )
    parser.add_argument(
        "--accelerator",
        choices=["cuda", "cpu"],
        help="Accelerator to use (auto-detects if not specified)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    raw_dir = repo_root / "data" / "rag" / "raw"
    processed_dir = repo_root / "data" / "rag" / "processed"
    convert_pdfs(
        raw_dir,
        processed_dir,
        accelerator=args.accelerator,
        force=args.force,
        test=args.test,
    )


main()
