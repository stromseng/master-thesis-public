"""SSH utilities for IDUN cluster access."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from idun.config import IdunConfig


@dataclass
class PortForward:
    local: int
    remote: int


def ssh_target(host: str, user: str) -> str:
    """Build SSH target string (user@host or just host)."""
    return f"{user}@{host}" if user else host


def run_ssh(
    config: IdunConfig,
    command: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on jump host and capture output."""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "LogLevel=ERROR",
            ssh_target(config.jump_host, config.ssh_user),
            command,
        ],
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "SSH command failed"
        )
    return result


def run_ssh_stream(
    config: IdunConfig,
    command: str,
    check: bool = False,
    tty: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command with output streamed to terminal.

    Args:
        config: IDUN configuration
        command: Command to run on remote
        check: Raise exception on non-zero exit
        tty: Allocate TTY for interactive programs (less, vim, etc.)
    """
    cmd = ["ssh"]
    if tty:
        cmd.append("-t")
    cmd.extend([ssh_target(config.jump_host, config.ssh_user), command])
    result = subprocess.run(cmd, check=check, text=True)
    if check and result.returncode != 0:
        raise RuntimeError("SSH command failed")
    return result


def run_ssh_via_jump(
    config: IdunConfig,
    host: str,
    command: str,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on a compute node via jump host."""
    user_at_host = f"{config.ssh_user}@{host}" if config.ssh_user else host
    return subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "LogLevel=ERROR",
            "-J",
            ssh_target(config.jump_host, config.ssh_user),
            user_at_host,
            command,
        ],
        text=True,
        capture_output=True,
    )


def scp_to_jump(config: IdunConfig, local_path: Path, remote_path: str) -> None:
    """Copy file to jump host via SCP."""
    target = f"{ssh_target(config.jump_host, config.ssh_user)}:{remote_path}"
    _ = subprocess.run(["scp", str(local_path), target], check=True)


def open_ssh_shell(
    config: IdunConfig,
    host: str,
    forwards: list[PortForward] | None = None,
    shell: bool = True,
    job_id: str | None = None,
) -> None:
    """Open an interactive SSH shell to a compute node."""
    jump_target = ssh_target(config.jump_host, config.ssh_user)
    cmd = ["ssh", "-tt"]
    if forwards:
        if not host:
            raise RuntimeError("Host required for port forwarding")
        for forward in forwards:
            cmd += ["-L", f"{forward.local}:{host}:{forward.remote}"]
    cmd.append(jump_target)
    if shell and job_id:
        cmd.append(f"srun --jobid {shlex.quote(job_id)} --pty bash -l")
    elif shell:
        cmd.append("exec bash -l -i")
    _ = subprocess.run(cmd)


def open_direct_ssh(
    config: IdunConfig,
    host: str,
    forwards: list[PortForward] | None = None,
) -> None:
    """Open direct SSH to compute node via ProxyJump (faster than srun)."""
    jump_target = ssh_target(config.jump_host, config.ssh_user)
    compute_target = ssh_target(host, config.ssh_user)
    cmd = ["ssh", "-tt", "-J", jump_target]
    if forwards:
        for forward in forwards:
            cmd += ["-L", f"{forward.local}:localhost:{forward.remote}"]
    cmd.append(compute_target)
    _ = subprocess.run(cmd)


@dataclass
class ProcessInfo:
    pid: int
    command: str
    start_time: str = ""


def _extract_script_name(command: str) -> str:
    """Extract a readable script/module name from a command line."""
    import re

    # Check for -m module pattern (e.g., "python -m scripts.idun.convert_pdfs")
    module_match = re.search(r"-m\s+(\S+)", command)
    if module_match:
        module = module_match.group(1)
        # Get last part of module path (e.g., "convert_pdfs" from "scripts.idun.convert_pdfs")
        return module.split(".")[-1]

    # Check for .py file in command
    py_match = re.search(r"(\S+\.py)", command)
    if py_match:
        # Get just the filename
        path = py_match.group(1)
        return path.split("/")[-1]

    # Fall back to first meaningful word
    parts = command.split()
    if parts:
        # Return the executable name
        return parts[0].split("/")[-1]
    return command


def list_python_processes(
    config: IdunConfig,
    host: str,
) -> list[ProcessInfo]:
    """List Python processes on compute node, filtering out system processes."""
    import re

    # Use ps with start time and full command line
    # lstart format: "Wed Jan 29 15:58:00 2026"
    result = run_ssh_via_jump(
        config,
        host,
        "ps -u $USER -o pid=,lstart=,args= | grep -E 'python|uv' | grep -v grep",
    )
    processes: list[ProcessInfo] = []
    if result.returncode != 0:
        return processes
    # Filter out srun, slurmstepd, and bare "python" (slurm batch wrapper)
    skip_patterns = {"srun", "slurmstepd"}
    # Pattern to parse: "PID Day Mon DD HH:MM:SS YYYY command..."
    # e.g., "2518469 Wed Jan 29 15:58:00 2026 uv run python -m ..."
    line_pattern = re.compile(
        r"^\s*(\d+)\s+\w+\s+\w+\s+\d+\s+(\d{2}:\d{2}):\d{2}\s+\d{4}\s+(.+)$"
    )
    for line in result.stdout.strip().splitlines():
        match = line_pattern.match(line)
        if not match:
            continue
        pid_str, start_time, command = match.groups()
        # Skip system processes
        if any(skip in command for skip in skip_patterns):
            continue
        # Skip bare python (slurm batch wrapper that can't be killed)
        if command.strip() == "python":
            continue
        try:
            # Extract readable name from command
            display_name = _extract_script_name(command)
            processes.append(
                ProcessInfo(
                    pid=int(pid_str), command=display_name, start_time=start_time
                )
            )
        except ValueError:
            continue
    return processes


def kill_processes(
    config: IdunConfig,
    host: str,
    pids: list[int],
) -> bool:
    """Kill processes on compute node by PID."""
    if not pids:
        return True
    pid_args = " ".join(str(pid) for pid in pids)
    result = run_ssh_via_jump(config, host, f"kill {pid_args}")
    return result.returncode == 0


def parse_port_forwards(raw: str) -> list[PortForward]:
    """Parse port forward specification (e.g., '8888,6006' or '8888:6006')."""
    import re

    tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
    forwards: list[PortForward] = []
    for token in tokens:
        if ":" in token:
            left, right = token.split(":", 1)
            if not left or not right:
                raise ValueError(f"Invalid port mapping: {token}")
            local = int(left)
            remote = int(right)
        else:
            local = int(token)
            remote = local
        forwards.append(PortForward(local=local, remote=remote))
    return forwards
