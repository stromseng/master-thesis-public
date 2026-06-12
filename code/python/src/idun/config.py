"""IDUN configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tomllib
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel

console = Console()

# Load .env from repo root (walk up from this file to find it)
_THIS_DIR = Path(__file__).resolve().parent
for _parent in [_THIS_DIR, *_THIS_DIR.parents]:
    _env_file = _parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        break

CONFIG_PATH = Path.home() / ".config" / "idun" / "config.toml"
REMOTE_REPO_PATH = "$HOME/repos/master-thesis"
REMOTE_REPO_DISPLAY = "~/repos/master-thesis"
IDUN_SCRIPTS_DIR = Path("code") / "python" / "scripts" / "idun"

TERMINAL_STATES = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}
QUEUED_STATES = {"PENDING"}

GPU_CUSTOM_SENTINEL = "__custom__"
GPU_TYPE_VRAM_OPTIONS = {
    "p100": ["16g"],
    "v100": ["16g", "32g"],
    "a100": ["40g", "80g"],
    "h100": ["80g"],
    "h200": ["80g"],
}
GPU_VRAM_CONSTRAINTS = {
    "16g": "gpu16g",
    "32g": "gpu32g",
    "40g": "gpu40g",
    "40g|80g": "gpu40g|gpu80g",
    "80g": "gpu80g",
}


@dataclass
class IdunConfig:
    jump_host: str
    ssh_user: str
    account: str
    partition_cpu: str
    partition_gpu: str
    time_limit: str
    cpus_per_task: int
    mem: str
    job_name_prefix: str
    python_cmd: str
    sleep_command: str
    user_email: str


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _data_int(data: dict[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def load_config() -> IdunConfig:
    """Load IDUN configuration from file and environment."""
    data: dict[str, object] = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as handle:
                raw = tomllib.load(handle)
            if isinstance(raw, dict):
                data = (
                    raw.get("defaults", raw)
                    if isinstance(raw.get("defaults", raw), dict)
                    else raw
                )
        except Exception as exc:
            console.print(
                Panel(str(exc), title="Failed to read config", border_style="red")
            )

    default_user = _env_str(
        "IDUN_USER",
        str(data.get("ssh_user", os.environ.get("USER", ""))),
    )

    jump_host = _env_str("IDUN_JUMP_HOST", str(data.get("jump_host", "idun")))
    ssh_user = _env_str("IDUN_SSH_USER", default_user)
    account = _env_str("IDUN_ACCOUNT", str(data.get("account", "share-ie-idi")))
    partition_cpu = _env_str(
        "IDUN_PARTITION_CPU", str(data.get("partition_cpu", "CPUQ"))
    )
    partition_gpu = _env_str(
        "IDUN_PARTITION_GPU", str(data.get("partition_gpu", "GPUQ"))
    )
    time_limit = _env_str("IDUN_TIME", str(data.get("time_limit", "0-01:00:00")))
    cpus_per_task = _env_int("IDUN_CPUS", _data_int(data, "cpus_per_task", 2))
    mem = _env_str("IDUN_MEM", str(data.get("mem", "32G")))
    job_name_prefix = _env_str(
        "IDUN_JOB_NAME_PREFIX", str(data.get("job_name_prefix", "idun"))
    )
    python_cmd = _env_str("IDUN_PYTHON", str(data.get("python_cmd", "python")))
    sleep_command = _env_str(
        "IDUN_SLEEP_CMD", str(data.get("sleep_command", "sleep 30d"))
    )
    user_email = _env_str("USER_EMAIL", str(data.get("user_email", "")))

    return IdunConfig(
        jump_host=jump_host,
        ssh_user=ssh_user,
        account=account,
        partition_cpu=partition_cpu,
        partition_gpu=partition_gpu,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        job_name_prefix=job_name_prefix,
        python_cmd=python_cmd,
        sleep_command=sleep_command,
        user_email=user_email,
    )
