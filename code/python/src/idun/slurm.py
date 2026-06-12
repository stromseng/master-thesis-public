"""SLURM job management utilities."""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from idun.config import IdunConfig, TERMINAL_STATES, QUEUED_STATES
from idun.ssh import run_ssh, scp_to_jump
from idun.fuse_overlay import (
    generate_run_id,
    get_overlay_paths,
    build_mount_commands,
)

console = Console()

_REMOTE_TZ_CACHE: dict[tuple[str, str], timezone | None] = {}


def parse_job_id(output: str) -> str | None:
    """Extract job ID from sbatch output."""
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else None


def parse_scontrol_value(output: str, key: str) -> str | None:
    """Extract a value from scontrol output."""
    match = re.search(rf"{re.escape(key)}=([^\s]+)", output)
    if not match:
        return None
    value = match.group(1)
    if value.lower() in {"(null)", "none", "unknown", "n/a"}:
        return None
    return value


def get_remote_tzinfo(config: IdunConfig) -> timezone | None:
    """Get timezone info from remote host."""
    key = (config.jump_host, config.ssh_user)
    if key in _REMOTE_TZ_CACHE:
        return _REMOTE_TZ_CACHE[key]
    result = run_ssh(config, "date +%z", check=False)
    tz_str = (result.stdout or result.stderr or "").strip()
    if not re.fullmatch(r"[+-][0-2]\d[0-5]\d", tz_str):
        _REMOTE_TZ_CACHE[key] = None
        return None
    sign = 1 if tz_str[0] == "+" else -1
    hours = int(tz_str[1:3])
    minutes = int(tz_str[3:5])
    total_minutes = sign * (hours * 60 + minutes)
    tzinfo = timezone(timedelta(minutes=total_minutes))
    _REMOTE_TZ_CACHE[key] = tzinfo
    return tzinfo


def parse_estimated_start(config: IdunConfig, iso_str: str | None) -> datetime | None:
    """Parse estimated start time from SLURM."""
    if not iso_str:
        return None
    lowered = iso_str.lower()
    if lowered in {"n/a", "not", "not available", "unknown", "none", "(null)"}:
        return None
    try:
        naive_dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return None
    tzinfo = get_remote_tzinfo(config)
    if tzinfo is None:
        return naive_dt
    return naive_dt.replace(tzinfo=tzinfo).astimezone()


def format_eta(est_start: datetime | None) -> str | None:
    """Format ETA as HH:MM:SS."""
    if est_start is None:
        return None
    now = datetime.now(tz=est_start.tzinfo) if est_start.tzinfo else datetime.now()
    total_seconds = int((est_start - now).total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_estimated_start_time(config: IdunConfig, job_id: str) -> str | None:
    """Get estimated start time from squeue."""
    result = run_ssh(
        config,
        f"squeue --start -j {shlex.quote(job_id)} -h -o '%S'",
        check=False,
    )
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return None
    if output.lower() in {"n/a", "not available", "unknown"}:
        return None
    return output


def get_job_state_and_host(
    config: IdunConfig, job_id: str
) -> tuple[str | None, str | None]:
    """Get job state and batch host from scontrol."""
    result = run_ssh(config, f"scontrol show job {shlex.quote(job_id)}", check=False)
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return None, None
    state = parse_scontrol_value(output, "JobState")
    host = parse_scontrol_value(output, "BatchHost")
    return state, host


def get_job_state_host_start(
    config: IdunConfig, job_id: str
) -> tuple[str | None, str | None, str | None]:
    """Get job state, host, and start time."""
    result = run_ssh(config, f"scontrol show job {shlex.quote(job_id)}", check=False)
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return None, None, None
    state = parse_scontrol_value(output, "JobState")
    host = parse_scontrol_value(output, "BatchHost")
    start_time = parse_scontrol_value(output, "StartTime")
    return state, host, start_time


def get_job_gpu_count(config: IdunConfig, job_id: str) -> int:
    """Get GPU count allocated to a job from SLURM."""
    result = run_ssh(config, f"scontrol show job {shlex.quote(job_id)}", check=False)
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return 1

    # Try to parse from TRES allocation (e.g., "TRES=cpu=32,mem=64G,node=1,gres/gpu=3")
    tres = parse_scontrol_value(output, "TRES")
    if tres:
        match = re.search(r"gres/gpu=(\d+)", tres)
        if match:
            return int(match.group(1))

    # Fallback: try to parse from Gres field (e.g., "Gres=gpu:a100:3")
    gres = parse_scontrol_value(output, "Gres")
    if gres:
        match = re.search(r"gpu(?::[^:]+)?:(\d+)", gres)
        if match:
            return int(match.group(1))

    # Default to 1 if we can't determine
    return 1


def get_job_logs(config: IdunConfig, job_id: str) -> tuple[str | None, str | None]:
    """Get stdout and stderr paths for a job."""
    result = run_ssh(config, f"scontrol show job {shlex.quote(job_id)}", check=False)
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return None, None
    stdout_path = parse_scontrol_value(output, "StdOut")
    stderr_path = parse_scontrol_value(output, "StdErr")
    return stdout_path, stderr_path


def get_batch_script(config: IdunConfig, job_id: str) -> str | None:
    """Retrieve the batch script content for a job."""
    result = run_ssh(
        config,
        f"scontrol write batch_script {shlex.quote(job_id)} -",
        check=False,
    )
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def is_script_done(config: IdunConfig, job_id: str) -> bool:
    """Check if script completion marker exists."""
    result = run_ssh(
        config,
        f"test -f $HOME/.cache/idun/done_{shlex.quote(job_id)}",
        check=False,
    )
    return result.returncode == 0


def wait_for_running(config: IdunConfig, job_id: str, poll: int = 10) -> str:
    """Wait for job to reach RUNNING state and return batch host."""
    attempt = 0
    last_est_start: str | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Waiting for job {job_id} to start...", total=None)
        while True:
            attempt += 1
            state, host, start_time = get_job_state_host_start(config, job_id)
            display_state = state or "unknown"
            if state in QUEUED_STATES:
                last_est_start = start_time or last_est_start
                est = get_estimated_start_time(config, job_id)
                if est:
                    last_est_start = est
            eta_fragment = ""
            if state in QUEUED_STATES and last_est_start:
                eta = format_eta(parse_estimated_start(config, last_est_start))
                if eta:
                    eta_fragment = f" | eta={eta}"
            progress.update(
                task,
                description=f"Job {job_id}: {display_state}{eta_fragment}",
            )
            if state == "RUNNING" and host:
                return host
            if state in TERMINAL_STATES:
                raise RuntimeError(f"Job {job_id} entered terminal state: {state}")
            time.sleep(poll)


def build_slurm_script(
    *,
    job_name: str,
    partition: str,
    account: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    gpus: int,
    gpu_type: str,
    constraint: str,
    sleep_command: str,
    run_module: str | None,
    overlay_path: str | None,
    repo_path: str,
    keep_alive: bool,
    notify: bool = False,
    user_email: str = "",
    script_args: str | None = None,
    begin_time: str | None = None,
    save_artifacts: bool = False,
    exclude_nodes: list[str] | None = None,
) -> str:
    """Build SLURM batch script."""
    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --account={account}")
    lines.append(f"#SBATCH --time={time_limit}")
    lines.append("#SBATCH --nodes=1")
    lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append("#SBATCH --output=slurm-%j.out")
    lines.append("#SBATCH --error=slurm-%j.err")
    if gpus > 0:
        gres = f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}"
        lines.append(f"#SBATCH --gres={gres}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if exclude_nodes:
        lines.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")
    if begin_time:
        lines.append(f"#SBATCH --begin={begin_time}")
    if notify and user_email:
        lines.append(f"#SBATCH --mail-user={user_email}")
        lines.append("#SBATCH --mail-type=ALL")
    if save_artifacts:
        # Send SIGTERM 5 min before time limit so cleanup can save artifacts
        lines.append("#SBATCH --signal=B:TERM@300")
    lines.append("")

    if run_module is None:
        lines.append(sleep_command)
        return "\n".join(lines) + "\n"

    lines.append("set -euo pipefail")
    lines.append(f'repo_dir="{repo_path}"')
    lines.append(
        'if [ ! -d "$repo_dir/.git" ]; then echo "Repo not found at $repo_dir" >&2; exit 1; fi'
    )

    # Generate unique run ID and setup fuse-overlayfs paths
    lines.append("run_id=$(head -c 6 /dev/urandom | xxd -p)")
    lines.append('overlay_base="/tmp/idun-overlay-${run_id}"')
    lines.append('overlay_upper="${overlay_base}/upper"')
    lines.append('overlay_work="${overlay_base}/work"')
    lines.append('overlay_merged="${overlay_base}/merged"')
    lines.append('mkdir -p "$overlay_upper" "$overlay_work" "$overlay_merged"')

    # Extract overlay bundle to upperdir if provided
    if overlay_path:
        lines.append(f'overlay_path="{overlay_path}"')
        lines.append(
            'if [ -f "$overlay_path" ]; then tar -xzf "$overlay_path" -C "$overlay_upper"; fi'
        )
        # Handle deleted files by creating whiteout character devices
        lines.append(
            'if [ -f "$overlay_upper/.idun_deleted.txt" ]; then '
            + "while IFS= read -r path; do "
            + '[ -n "$path" ] && mkdir -p "$(dirname "$overlay_upper/$path")" && '
            + 'mknod "$overlay_upper/$path" c 0 0 2>/dev/null || true; '
            + 'done < "$overlay_upper/.idun_deleted.txt"; '
            + 'rm -f "$overlay_upper/.idun_deleted.txt"; '
            + "fi"
        )

    # Mount fuse-overlayfs
    lines.append(
        'fuse-overlayfs -o "lowerdir=$repo_dir,upperdir=$overlay_upper,workdir=$overlay_work" "$overlay_merged"'
    )
    lines.append('echo "Using isolated overlay: $overlay_merged"')

    # Track start time for duration calculation
    lines.append("start_time=$(date +%s)")
    # Clean script name for artifact naming
    module_name = run_module.split(".")[-1] if run_module else "script"
    lines.append(f'script_name="{module_name}"')

    # Setup cleanup trap (as function to handle artifact saving)
    lines.append("cleanup() {")
    lines.append('  fusermount3 -u "$overlay_merged" 2>/dev/null || true')
    if save_artifacts:
        # Calculate duration and format artifact directory name
        lines.append("  end_time=$(date +%s)")
        lines.append("  duration=$((end_time - start_time))")
        lines.append("  hours=$((duration / 3600))")
        lines.append("  minutes=$(((duration % 3600) / 60))")
        lines.append("  seconds=$((duration % 60))")
        lines.append("  if [ $hours -gt 0 ]; then")
        lines.append('    duration_str="${hours}h${minutes}m${seconds}s"')
        lines.append("  elif [ $minutes -gt 0 ]; then")
        lines.append('    duration_str="${minutes}m${seconds}s"')
        lines.append("  else")
        lines.append('    duration_str="${seconds}s"')
        lines.append("  fi")
        lines.append("  timestamp=$(date +%Y%m%d_%H%M%S)")
        lines.append(
            '  artifact_dir="$HOME/.cache/idun/artifacts/${timestamp}_${script_name}_${duration_str}"'
        )
        lines.append('  mkdir -p "$artifact_dir"')
        lines.append(
            '  if [ -d "$overlay_upper" ] && [ "$(ls -A "$overlay_upper")" ]; then '
            'cp -r "$overlay_upper"/. "$artifact_dir"/; fi'
        )
        lines.append('  echo "Artifacts saved to: $artifact_dir"')
    lines.append('  rm -rf "$overlay_base" 2>/dev/null || true')
    lines.append("}")
    lines.append("trap cleanup EXIT INT TERM")

    # Run from merged directory, but use shared venv outside overlay
    # Use flock to prevent concurrent uv sync from corrupting venv
    lines.append('cd "$overlay_merged/code/python"')
    lines.append('export UV_PROJECT_ENVIRONMENT="$HOME/.cache/idun/venv"')
    lines.append(
        'flock "$HOME/.cache/idun/venv.lock" -c \'if [ -f "uv.lock" ]; then uv sync --frozen; else uv sync; fi\''
    )
    lines.append("export PYTHONUNBUFFERED=1")
    lines.append("set +e")
    python_cmd = f"uv run python -u -m {run_module}"
    if script_args:
        python_cmd = f"{python_cmd} {script_args}"
    lines.append(python_cmd)
    lines.append("script_status=$?")
    lines.append("set -e")
    lines.append('mkdir -p "$HOME/.cache/idun"')
    lines.append('touch "$HOME/.cache/idun/done_${SLURM_JOB_ID}"')
    lines.append('echo "Script exit code: $script_status"')
    if keep_alive:
        lines.append(
            "echo 'Script finished. Allocation kept alive; attach with: srun --jobid ${SLURM_JOB_ID} --pty bash -l'"
        )
        lines.append("sleep infinity")
    lines.append("exit $script_status")
    return "\n".join(lines) + "\n"


def run_on_existing_job(
    config: IdunConfig,
    job_id: str,
    run_module: str,
    overlay_path: str | None,
    repo_path: str,
    background: bool = False,
    script_name: str | None = None,
    script_args: str | None = None,
    host: str | None = None,
    save_artifacts: bool = False,
) -> tuple[str | None, str]:
    """Run a script on an existing RUNNING job using fuse-overlayfs isolation.

    Each run gets its own isolated overlay mount, allowing multiple scripts
    to run concurrently without interfering with each other or the base repo.

    For background mode, SSHs directly to compute node via ProxyJump.
    For foreground mode, uses srun --jobid from jump host.

    Returns tuple of (log_path, run_id). log_path is None for foreground runs.
    """
    # Generate unique run ID for this execution
    run_id = generate_run_id()
    paths = get_overlay_paths(run_id)

    # Clean script name for artifact directory (remove .py, replace spaces)
    clean_script_name = (
        script_name.replace(".py", "").replace(" ", "_")
        if script_name
        else run_module.split(".")[-1]
    )

    # Build the command to run on the compute node with fuse-overlayfs isolation
    commands: list[str] = [
        "set -euo pipefail",
        f'repo_dir="{repo_path}"',
        'if [ ! -d "$repo_dir/.git" ]; then echo "Repo not found at $repo_dir" >&2; exit 1; fi',
        # Track start time for duration calculation
        "start_time=$(date +%s)",
        f'script_name="{clean_script_name}"',
    ]

    # Setup fuse-overlayfs mount
    commands.extend(build_mount_commands(run_id, repo_path, overlay_path))

    # Setup cleanup trap to unmount and optionally save artifacts
    # Define cleanup as a function to avoid quoting issues with trap
    commands.append("cleanup() {")
    commands.append(f"  fusermount3 -u {paths['merged']} 2>/dev/null || true")
    if save_artifacts:
        # Calculate duration and format artifact directory name
        commands.append("  end_time=$(date +%s)")
        commands.append("  duration=$((end_time - start_time))")
        commands.append("  hours=$((duration / 3600))")
        commands.append("  minutes=$(((duration % 3600) / 60))")
        commands.append("  seconds=$((duration % 60))")
        commands.append("  if [ $hours -gt 0 ]; then")
        commands.append('    duration_str="${hours}h${minutes}m${seconds}s"')
        commands.append("  elif [ $minutes -gt 0 ]; then")
        commands.append('    duration_str="${minutes}m${seconds}s"')
        commands.append("  else")
        commands.append('    duration_str="${seconds}s"')
        commands.append("  fi")
        commands.append("  timestamp=$(date +%Y%m%d_%H%M%S)")
        commands.append(
            '  artifact_dir="$HOME/.cache/idun/artifacts/${timestamp}_${script_name}_${duration_str}"'
        )
        commands.append('  mkdir -p "$artifact_dir"')
        commands.append(
            f'  if [ -d {paths["upper"]} ] && [ "$(ls -A {paths["upper"]})" ]; then '
            f'cp -r {paths["upper"]}/. "$artifact_dir"/; fi'
        )
        commands.append('  echo "Artifacts saved to: $artifact_dir"')
    commands.append(f"  rm -rf {paths['base_dir']} 2>/dev/null || true")
    commands.append("}")
    commands.append("trap cleanup EXIT INT TERM")

    # Build the python command with optional args
    python_cmd = f"uv run python -u -m {run_module}"
    if script_args:
        python_cmd = f"{python_cmd} {script_args}"

    # Run from the merged (overlay) directory, but use shared venv outside overlay
    # Use flock to prevent concurrent uv sync from corrupting venv
    commands.extend(
        [
            f"cd {paths['merged']}/code/python",
            "export UV_PROJECT_ENVIRONMENT=$HOME/.cache/idun/venv",
            "flock $HOME/.cache/idun/venv.lock -c 'if [ -f \"uv.lock\" ]; then uv sync --frozen; else uv sync; fi'",
            "export PYTHONUNBUFFERED=1",
            python_cmd,
        ]
    )

    script_body = "\n".join(commands)
    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )

    if background:
        if not host:
            raise RuntimeError("Host required for background execution")

        # SSH directly to compute node via ProxyJump for reliable background execution
        compute_target = f"{config.ssh_user}@{host}" if config.ssh_user else host
        log_dir = "$HOME/.cache/idun/logs"
        # Use script name (without .py) or module name for log file
        name_part = (
            script_name.replace(".py", "") if script_name else run_module.split(".")[-1]
        )
        # Generate timestamp, create log file, run script in background with nohup
        # Use subshell ( ... & ) so SSH can exit cleanly without waiting for child
        bg_cmd = (
            f"mkdir -p {log_dir} && "
            f"ts=$(date +%Y%m%d_%H%M%S) && "
            f"log_file={log_dir}/${{ts}}_{name_part}.log && "
            f"echo $log_file && "
            f"( nohup bash -c {shlex.quote(script_body)} </dev/null >$log_file 2>&1 & )"
        )

        console.print(f"Running script on job {job_id} ({host}) in background...")
        console.print(f"Using isolated overlay: {paths['merged']}")
        result = subprocess.run(
            ["ssh", "-n", "-J", jump_target, compute_target, bg_cmd],
            text=True,
            capture_output=True,
        )
        # Return the actual log path from the command output
        log_path = (result.stdout or "").strip()
        if not log_path:
            log_path = f"~/.cache/idun/logs/<timestamp>_{name_part}.log"
        return log_path, run_id
    else:
        # Run interactively in foreground using srun
        srun_cmd = f"srun --jobid {shlex.quote(job_id)} --pty bash -c {shlex.quote(script_body)}"
        console.print(f"Running script on job {job_id}...")
        console.print(f"Using isolated overlay: {paths['merged']}")
        result = subprocess.run(["ssh", "-tt", jump_target, srun_cmd])
        if result.returncode != 0:
            raise RuntimeError(
                f"Script execution failed with exit code {result.returncode}"
            )
        return None, run_id


def submit_slurm(config: IdunConfig, slurm_text: str) -> str:
    """Submit SLURM script and return job ID."""
    remote_dir = "~/.cache/idun/slurm"
    _ = run_ssh(config, f"mkdir -p {remote_dir}")
    with tempfile.NamedTemporaryFile("w", suffix=".slurm", delete=False) as handle:
        _ = handle.write(slurm_text)
        local_path = Path(handle.name)

    remote_path = f"{remote_dir}/{local_path.name}"
    try:
        scp_to_jump(config, local_path, remote_path)
        result = run_ssh(config, f"sbatch {remote_path}")
    finally:
        try:
            local_path.unlink()
        except FileNotFoundError:
            pass

    output = (result.stdout or result.stderr or "").strip()
    job_id = parse_job_id(output)
    if not job_id:
        raise RuntimeError(f"Failed to parse job id from sbatch output: {output}")
    return job_id
