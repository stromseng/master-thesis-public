"""Interactive prompt utilities for IDUN CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterable

import typer
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator, PathValidator

from idun.config import (
    IdunConfig,
    IDUN_SCRIPTS_DIR,
    GPU_CUSTOM_SENTINEL,
    GPU_TYPE_VRAM_OPTIONS,
)
from idun.jobs import (
    fetch_jobs,
    fetch_completed_jobs,
    sort_jobs_by_start_time,
    job_choice_label,
)


def prompt_choice(
    message: str, choices: Iterable[str], default: str | None = None
) -> str:
    """Prompt user to select from choices."""
    return inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message=message, choices=list(choices), default=default
    ).execute()


def prompt_text(message: str, default: str | None = None) -> str:
    """Prompt user for text input."""
    return inquirer.text(message=message, default=default or "").execute()  # pyright: ignore[reportPrivateImportUsage]


def prompt_int(message: str, default: int) -> int:
    """Prompt user for integer input."""
    value = inquirer.text(  # pyright: ignore[reportPrivateImportUsage]
        message=message,
        default=str(default),
        validate=NumberValidator(message="Enter a valid integer"),
    ).execute()
    return int(value)


def prompt_confirm(message: str, default: bool = False) -> bool:
    """Prompt user for confirmation.  Returns *default* when stdin is not a TTY."""
    if not sys.stdin.isatty():
        return default
    return bool(inquirer.confirm(message=message, default=default).execute())  # pyright: ignore[reportPrivateImportUsage]


def prompt_job_id() -> str:
    """Prompt user for job ID."""
    return prompt_text("Job ID").strip()


def format_vram_label(vram: str) -> str:
    """Format VRAM size for display."""
    if vram.endswith("g"):
        return f"{vram[:-1]}GB"
    return vram.upper()


def gpu_spec_choices(
    *, allowed_types: set[str] | None = None, include_any: bool = True
) -> list[dict[str, object]]:
    """Build GPU selection choices."""
    choices: list[dict[str, object]] = []
    if include_any:
        choices.append({"name": "Any GPU", "value": ("", "")})
        choices.append({"name": "Any GPU 16GB", "value": ("", "16g")})
        choices.append({"name": "Any GPU 40GB", "value": ("", "40g")})
        choices.append({"name": "Any GPU 40/80GB", "value": ("", "40g|80g")})
        choices.append({"name": "Any GPU 80GB", "value": ("", "80g")})
        choices.append({"name": "Custom GPU...", "value": (GPU_CUSTOM_SENTINEL, "")})
    for gpu_type, vrams in GPU_TYPE_VRAM_OPTIONS.items():
        if allowed_types is not None and gpu_type not in allowed_types:
            continue
        if len(vrams) > 1:
            choices.append({"name": f"{gpu_type} Any VRAM", "value": (gpu_type, "")})
        for vram in vrams:
            label = f"{gpu_type} {format_vram_label(vram)}"
            choices.append({"name": label, "value": (gpu_type, vram)})
    return choices


def prompt_custom_gpu_settings() -> tuple[str, str]:
    """Prompt for custom GPU settings."""
    from rich.console import Console

    console = Console()
    while True:
        custom_type = prompt_text("Custom GPU type (leave empty for any)", "").strip()
        custom_constraint = prompt_text(
            "Custom Slurm constraint (optional)", ""
        ).strip()
        if custom_type or custom_constraint:
            return custom_type, custom_constraint
        console.print("Provide a GPU type or constraint.")


def prompt_gpu_spec(
    *,
    default_type: str = "",
    default_vram: str = "",
    allowed_types: set[str] | None = None,
) -> tuple[str, str, str]:
    """Prompt for GPU specification. Returns (gpu_type, gpu_vram, constraint_override)."""
    include_any = allowed_types is None
    choices = gpu_spec_choices(allowed_types=allowed_types, include_any=include_any)
    if not choices:
        raise typer.BadParameter("No GPU options available")
    default_value = (default_type, default_vram)
    default_label = next(
        (choice["name"] for choice in choices if choice["value"] == default_value),
        None,
    )
    selection = inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message="GPU model + VRAM",
        choices=choices,
        default=default_label or choices[0]["name"],
    ).execute()
    if selection[0] == GPU_CUSTOM_SENTINEL:
        custom_type, custom_constraint = prompt_custom_gpu_settings()
        return custom_type, "", custom_constraint
    return selection[0], selection[1], ""


def list_idun_scripts(repo_root: Path) -> list[Path]:
    """List available IDUN scripts."""
    scripts_dir = repo_root / IDUN_SCRIPTS_DIR
    if not scripts_dir.exists():
        return []
    return sorted(
        [path for path in scripts_dir.glob("*.py") if path.name != "__init__.py"],
        key=lambda path: path.name.lower(),
    )


def prompt_idun_script(repo_root: Path) -> Path:
    """Prompt user to select an IDUN script."""
    scripts = list_idun_scripts(repo_root)
    if not scripts:
        return Path(
            inquirer.text(  # pyright: ignore[reportPrivateImportUsage]
                message="Path to local Python script",
                validate=PathValidator(is_file=True),
            ).execute()
        )
    choices: list[dict[str, object]] = [
        {"name": path.name, "value": path} for path in scripts
    ]
    return inquirer.select(message="Select script to run", choices=choices).execute()  # pyright: ignore[reportPrivateImportUsage]


def module_from_script(repo_root: Path, script_path: Path) -> str:
    """Convert script path to module path."""
    try:
        relative = script_path.resolve().relative_to(repo_root / "code" / "python")
    except ValueError as exc:
        raise typer.BadParameter(
            f"Script must live under {repo_root / 'code' / 'python'}"
        ) from exc
    parts = relative.with_suffix("").parts
    if len(parts) < 2 or parts[0] != "scripts" or parts[1] != "idun":
        raise typer.BadParameter("Script must be under code/python/scripts/idun")
    return ".".join(parts)


def script_display_path(repo_root: Path, script_path: Path) -> str:
    """Get display path for script relative to python directory."""
    try:
        relative = script_path.resolve().relative_to(repo_root / "code" / "python")
    except ValueError:
        return str(script_path)
    return str(relative)


def choose_job(
    config: IdunConfig,
    jobs: list[dict[str, str]],
    message: str = "Select a job",
    include_start: bool = False,
) -> dict[str, str] | None:
    """Prompt user to choose a job from list."""
    if not jobs:
        return None
    choices: list[dict[str, object]] = [
        {
            "name": job_choice_label(config, job, include_start=include_start),
            "value": job,
        }
        for job in jobs
    ]
    choices.append({"name": "Exit", "value": None})
    return inquirer.select(message=message, choices=choices).execute()  # pyright: ignore[reportPrivateImportUsage]


def select_job(
    config: IdunConfig,
    *,
    include_completed: bool = False,
    completed_hours: int = 24,
    running_only: bool = False,
    message: str = "Select a job",
    sort_newest: bool = False,
) -> dict[str, str] | None:
    """Fetch jobs and prompt user to select one."""
    from rich.console import Console

    console = Console()

    jobs = fetch_jobs(config)
    if running_only:
        jobs = [job for job in jobs if job.get("state") == "RUNNING"]
    if include_completed:
        jobs += fetch_completed_jobs(config, hours=completed_hours)
    if sort_newest:
        jobs = sort_jobs_by_start_time(config, jobs)
    if not jobs:
        console.print("No jobs found")
        return None
    return choose_job(
        config, jobs, message=message, include_start=sort_newest or include_completed
    )


def normalize_machine(value: str) -> str:
    """Normalize machine type input."""
    lowered = value.strip().lower()
    if lowered == "gpu":
        return "GPU"
    if lowered == "cpu":
        return "CPU"
    raise typer.BadParameter("Machine must be GPU or CPU")


def normalize_post_action(value: str) -> str:
    """Normalize post-action input."""
    lowered = value.strip().lower()
    if lowered in {"background", "logs", "shell"}:
        return lowered
    raise typer.BadParameter("Post action must be background, logs, or shell")


def normalize_log_choice(value: str) -> str:
    """Normalize log choice input."""
    lowered = value.strip().lower()
    if lowered in {"stdout", "stderr", "both"}:
        return lowered
    raise typer.BadParameter("Log choice must be stdout, stderr, or both")


def normalize_gpu_vram(value: str) -> str:
    """Normalize GPU VRAM input."""
    from idun.config import GPU_VRAM_CONSTRAINTS

    lowered = value.strip().lower()
    if lowered in {"", "any"}:
        return ""
    if lowered.startswith("gpu"):
        lowered = lowered[3:]
    if lowered in GPU_VRAM_CONSTRAINTS:
        return lowered
    raise typer.BadParameter("GPU VRAM must be 16g, 32g, 40g, 40g|80g, 80g, or any")


def normalize_gpu_type(value: str) -> str:
    """Normalize GPU type input."""
    lowered = value.strip().lower()
    if lowered in {"", "any"}:
        return ""
    return lowered
