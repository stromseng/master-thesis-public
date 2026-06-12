"""Job listing and display utilities."""

from __future__ import annotations

import re
import shlex
import subprocess
import time

from rich.console import Console
from rich.table import Table

from idun.config import IdunConfig, TERMINAL_STATES, QUEUED_STATES
from idun.ssh import run_ssh, ssh_target
from idun.slurm import (
    parse_estimated_start,
    format_eta,
    get_job_logs,
    is_script_done,
    get_job_state_and_host,
)

console = Console()


def fetch_jobs(config: IdunConfig) -> list[dict[str, str]]:
    """Fetch currently queued/running jobs."""
    fmt = "%i|%T|%M|%D|%C|%m|%R|%j|%N|%S"
    user = config.ssh_user or "$USER"
    result = run_ssh(
        config,
        f"squeue -u {shlex.quote(user)} -h -o '{fmt}'",
        check=False,
    )
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return []
    jobs: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) != 10:
            continue
        job_id, state, time_used, nodes, cpus, mem, reason, name, node, start_time = (
            parts
        )
        jobs.append(
            {
                "id": job_id,
                "state": state,
                "time": time_used,
                "nodes": nodes,
                "cpus": cpus,
                "mem": mem,
                "reason": reason,
                "name": name,
                "node": node,
                "eta": "-",
                "start_time": start_time,
            }
        )

    # Fetch ETA for pending jobs
    pending_ids = [job["id"] for job in jobs if job.get("state") in QUEUED_STATES]
    start_times: dict[str, str] = {}
    if pending_ids:
        ids = ",".join(pending_ids)
        result = run_ssh(
            config,
            f"squeue --start -h -o '%i|%S' -j {shlex.quote(ids)}",
            check=False,
        )
        output = (result.stdout or result.stderr or "").strip()
        if output:
            for line in output.splitlines():
                if "|" not in line:
                    continue
                job_id, start_time = line.split("|", 1)
                if job_id:
                    start_times[job_id] = start_time

    for job in jobs:
        if job.get("state") in QUEUED_STATES:
            job_id = job.get("id", "")
            start_time = start_times.get(job_id) or job.get("start_time", "")
            eta_value = format_eta(parse_estimated_start(config, start_time))
            if eta_value:
                job["eta"] = eta_value
    return jobs


def get_active_nodes(config: IdunConfig) -> list[str]:
    """Return unique node names from RUNNING jobs for the current user."""
    jobs = fetch_jobs(config)
    nodes: list[str] = []
    seen: set[str] = set()
    for job in jobs:
        if job.get("state") != "RUNNING":
            continue
        node = job.get("node", "").strip()
        if node and node not in seen:
            seen.add(node)
            nodes.append(node)
    return nodes


def normalize_sacct_state(state: str) -> str:
    """Normalize sacct job state (remove suffixes like +, :)."""
    if not state:
        return ""
    base = state.split("+", 1)[0].split(":", 1)[0]
    return base


def normalize_job_id(job_id_raw: str) -> str | None:
    """Validate and normalize job ID."""
    job_id = job_id_raw.strip()
    if not job_id:
        return None
    if re.fullmatch(r"\d+(?:_\d+)?", job_id):
        return job_id
    return None


def fetch_completed_jobs(config: IdunConfig, hours: int = 24) -> list[dict[str, str]]:
    """Fetch completed jobs from sacct."""
    user = config.ssh_user or "$USER"
    states = ",".join(sorted(TERMINAL_STATES))
    start_expr = f"$(date -d '{hours} hours ago' '+%Y-%m-%dT%H:%M:%S')"
    command = (
        "sacct -X -P -n "
        f"-u {shlex.quote(user)} "
        f"-S {start_expr} -E now "
        f"-s {states} "
        "-o JobIDRaw,State,Start,End,Elapsed,AllocCPUS,ReqMem,JobName,NodeList"
    )
    result = run_ssh(config, command, check=False)
    if result.returncode != 0:
        return []
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return []
    jobs: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 9:
            continue
        job_id_raw, state_raw, start_time, end_time, elapsed, cpus, mem, name, node = (
            parts[:9]
        )
        job_id = normalize_job_id(job_id_raw)
        if not job_id:
            continue
        state = normalize_sacct_state(state_raw)
        if state and state not in TERMINAL_STATES:
            continue
        jobs.append(
            {
                "id": job_id,
                "state": state or state_raw or "-",
                "time": elapsed,
                "nodes": "-",
                "cpus": cpus,
                "mem": mem,
                "reason": "",
                "name": name,
                "node": node,
                "eta": "-",
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return jobs


def format_start_time(config: IdunConfig, raw: str | None) -> str | None:
    """Format start time for display."""
    if not raw:
        return None
    parsed = parse_estimated_start(config, raw)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d %H:%M")


def job_start_ts(config: IdunConfig, raw: str | None) -> float | None:
    """Get start time as timestamp for sorting."""
    if not raw:
        return None
    parsed = parse_estimated_start(config, raw)
    if parsed is None:
        return None
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def sort_jobs_by_start_time(
    config: IdunConfig, jobs: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Sort jobs by start time (newest first)."""

    def sort_key(job: dict[str, str]) -> tuple[bool, float]:
        ts = job_start_ts(config, job.get("start_time"))
        return (ts is not None, ts or 0.0)

    return sorted(jobs, key=sort_key, reverse=True)


def render_jobs(jobs: list[dict[str, str]]) -> None:
    """Render jobs as a table."""
    table = Table(title="IDUN Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("ETA")
    table.add_column("Time")
    table.add_column("Nodes")
    table.add_column("CPUs")
    table.add_column("Mem")
    table.add_column("Node")
    table.add_column("Name", style="green")
    for job in jobs:
        table.add_row(
            job.get("id", "-"),
            job.get("state", "-"),
            job.get("eta", "-"),
            job.get("time", "-"),
            job.get("nodes", "-"),
            job.get("cpus", "-"),
            job.get("mem", "-"),
            job.get("node", "-"),
            job.get("name", "-"),
        )
    console.print(table)


def job_choice_label(
    config: IdunConfig, job: dict[str, str], include_start: bool = False
) -> str:
    """Format job for interactive selection."""
    parts = [job.get("id", "-"), job.get("state", "-")]
    name = job.get("name")
    if name:
        parts.append(name)
    node = job.get("node")
    if node and node.lower() not in {"(null)", "(none)", "unknown", "n/a"}:
        parts.append(node)
    eta = job.get("eta")
    if eta and eta != "-":
        parts.append(f"eta {eta}")
    if include_start:
        start_display = format_start_time(config, job.get("start_time"))
        if start_display:
            parts.append(f"start {start_display}")
    return " | ".join(parts)


def tail_logs(config: IdunConfig, paths: list[str]) -> None:
    """Tail log files."""
    quoted = " ".join(shlex.quote(path) for path in paths)
    try:
        from idun.ssh import run_ssh_stream

        run_ssh_stream(config, f"tail -n 200 -f {quoted}", check=False)
    except KeyboardInterrupt:
        return


def tail_logs_until_done(
    config: IdunConfig,
    paths: list[str],
    job_id: str,
    poll: int = 5,
    stop_on_done: bool = False,
) -> None:
    """Tail logs until job completes."""
    quoted = " ".join(shlex.quote(path) for path in paths)
    proc = subprocess.Popen(
        [
            "ssh",
            ssh_target(config.jump_host, config.ssh_user),
            f"tail -n 200 -f {quoted}",
        ]
    )
    try:
        while True:
            time.sleep(poll)
            state, _ = get_job_state_and_host(config, job_id)
            if stop_on_done:
                if is_script_done(config, job_id) or state in TERMINAL_STATES:
                    proc.terminate()
                    try:
                        _ = proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            elif state in TERMINAL_STATES:
                proc.terminate()
                try:
                    _ = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
    except KeyboardInterrupt:
        proc.terminate()
        try:
            _ = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                _ = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def tail_reuse_log(config: IdunConfig, log_path: str, mode: str = "view") -> None:
    """View or tail a reuse log file.

    Args:
        config: IDUN configuration
        log_path: Path to log file on remote
        mode: "view" (less, scrollable), "follow" (tail -f), or "tail" (last 200 lines)
    """
    from idun.ssh import run_ssh_stream

    try:
        if mode == "view":
            # Use less for full scrollable view (q to quit)
            # - tty=True allocates a TTY for interactive use
            # - sed converts carriage returns to newlines (for progress bars)
            # - awk filters to show only completed (100%) progress bars
            # - less -R handles ANSI color codes
            run_ssh_stream(
                config,
                f"sed 's/\\r/\\n/g' {shlex.quote(log_path)} | awk '!/[0-9]+%\\|/ || /100%/' | less -R",
                check=False,
                tty=True,
            )
        elif mode == "follow":
            run_ssh_stream(
                config, f"tail -n 200 -f {shlex.quote(log_path)}", check=False
            )
        else:  # tail
            run_ssh_stream(config, f"tail -n 200 {shlex.quote(log_path)}", check=False)
    except KeyboardInterrupt:
        return


def fetch_reuse_logs(config: IdunConfig, limit: int = 20) -> list[dict[str, str]]:
    """Fetch background reuse logs from ~/.cache/idun/logs/."""
    # List log files sorted by modification time (newest first)
    cmd = (
        "ls -t ~/.cache/idun/logs/*.log 2>/dev/null | head -n "
        + str(limit)
        + " | while read f; do "
        + "stat -c '%Y|%n' \"$f\" 2>/dev/null || stat -f '%m|%N' \"$f\" 2>/dev/null; "
        + "done"
    )
    result = run_ssh(config, cmd, check=False)
    output = (result.stdout or "").strip()
    if not output:
        return []

    logs: list[dict[str, str]] = []
    for line in output.splitlines():
        if "|" not in line:
            continue
        timestamp_str, path = line.split("|", 1)
        # Extract filename for display
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        # Parse modification time
        try:
            from datetime import datetime

            mtime = int(timestamp_str)
            mtime_dt = datetime.fromtimestamp(mtime)
            time_display = mtime_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            mtime = None
            time_display = "unknown"

        # Calculate duration from filename timestamp (start time) to mtime (last update)
        # Filename format: YYYYMMDD_HHMMSS_scriptname.log
        duration_display = ""
        if mtime is not None:
            try:
                # Extract timestamp from filename: 20250202_102315_scriptname.log
                parts = filename.split("_", 2)
                if len(parts) >= 2:
                    date_part = parts[0]  # YYYYMMDD
                    time_part = parts[1]  # HHMMSS
                    start_dt = datetime.strptime(
                        f"{date_part}_{time_part}", "%Y%m%d_%H%M%S"
                    )
                    duration_secs = mtime - int(start_dt.timestamp())
                    if duration_secs >= 0:
                        hours = duration_secs // 3600
                        minutes = (duration_secs % 3600) // 60
                        seconds = duration_secs % 60
                        if hours > 0:
                            duration_display = f"{hours}h{minutes}m{seconds}s"
                        elif minutes > 0:
                            duration_display = f"{minutes}m{seconds}s"
                        else:
                            duration_display = f"{seconds}s"
            except (ValueError, IndexError):
                pass

        logs.append(
            {
                "path": path,
                "filename": filename,
                "time": time_display,
                "duration": duration_display,
                "type": "reuse",
            }
        )
    return logs


def fetch_slurm_logs(config: IdunConfig, limit: int = 30) -> list[dict[str, str]]:
    """Fetch slurm job log files from the home directory."""
    # List slurm-*.out and slurm-*.err files sorted by modification time (newest first)
    cmd = (
        "ls -t ~/slurm-*.out ~/slurm-*.err 2>/dev/null | head -n "
        + str(limit)
        + " | while read f; do "
        + "stat -c '%Y|%n' \"$f\" 2>/dev/null || stat -f '%m|%N' \"$f\" 2>/dev/null; "
        + "done"
    )
    result = run_ssh(config, cmd, check=False)
    output = (result.stdout or "").strip()
    if not output:
        return []

    # Parse log files and collect job IDs
    logs: list[dict[str, str]] = []
    job_ids: set[str] = set()
    for line in output.splitlines():
        if "|" not in line:
            continue
        mtime_str, path = line.split("|", 1)
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        job_id = ""
        if filename.startswith("slurm-"):
            name_parts = filename.replace("slurm-", "").rsplit(".", 1)
            if name_parts:
                job_id = name_parts[0]
                job_ids.add(job_id)
        try:
            from datetime import datetime

            mtime = int(mtime_str)
            mtime_dt = datetime.fromtimestamp(mtime)
            time_display = mtime_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            time_display = "unknown"

        log_type = "stdout" if filename.endswith(".out") else "stderr"
        logs.append(
            {
                "path": path,
                "filename": filename,
                "time": time_display,
                "duration": "",
                "job_id": job_id,
                "log_type": log_type,
                "type": "slurm",
            }
        )

    # Query sacct for elapsed time of all job IDs in one call
    if job_ids:
        job_id_list = ",".join(sorted(job_ids))
        sacct_cmd = f"sacct -X -P -n -j {job_id_list} -o JobIDRaw,Elapsed 2>/dev/null"
        sacct_result = run_ssh(config, sacct_cmd, check=False)
        sacct_output = (sacct_result.stdout or "").strip()
        elapsed_map: dict[str, str] = {}
        if sacct_output:
            for sacct_line in sacct_output.splitlines():
                if "|" in sacct_line:
                    parts = sacct_line.split("|", 1)
                    if len(parts) == 2:
                        jid, elapsed = parts
                        elapsed_map[jid.strip()] = elapsed.strip()

        # Update logs with elapsed time
        for log in logs:
            jid = log.get("job_id", "")
            if jid and jid in elapsed_map:
                log["duration"] = elapsed_map[jid]

    return logs


def tail_slurm_log(config: IdunConfig, log_path: str, mode: str = "view") -> None:
    """View or tail a slurm log file.

    Args:
        config: IDUN configuration
        log_path: Path to log file on remote
        mode: "view" (less, scrollable), "follow" (tail -f), or "tail" (last 200 lines)
    """
    from idun.ssh import run_ssh_stream

    try:
        if mode == "view":
            # Use less for full scrollable view (q to quit)
            # - tty=True allocates a TTY for interactive use
            # - sed converts carriage returns to newlines (for progress bars)
            # - awk filters to show only completed (100%) progress bars
            # - less -R handles ANSI color codes
            run_ssh_stream(
                config,
                f"sed 's/\\r/\\n/g' {shlex.quote(log_path)} | awk '!/[0-9]+%\\|/ || /100%/' | less -R",
                check=False,
                tty=True,
            )
        elif mode == "follow":
            run_ssh_stream(
                config, f"tail -n 200 -f {shlex.quote(log_path)}", check=False
            )
        else:  # tail
            run_ssh_stream(config, f"tail -n 200 {shlex.quote(log_path)}", check=False)
    except KeyboardInterrupt:
        return


def follow_job_logs(
    config: IdunConfig,
    job_id: str,
    which: str = "stdout",
    stop_on_done: bool = False,
) -> None:
    """Follow job logs."""
    from idun.ssh import run_ssh_stream

    stdout_path, stderr_path = get_job_logs(config, job_id)
    if not stdout_path and not stderr_path:
        console.print("No log paths found")
        return
    which = which.lower()
    if which == "stderr":
        paths = [stderr_path or stdout_path]
    elif which == "both" and stdout_path and stderr_path:
        paths = [stdout_path, stderr_path]
    else:
        paths = [stdout_path or stderr_path]
    if not paths[0]:
        console.print("No log path available")
        return
    state, _ = get_job_state_and_host(config, job_id)
    if state in TERMINAL_STATES:
        run_ssh_stream(
            config,
            f"tail -n 200 {' '.join(shlex.quote(path) for path in paths if path)}",
            check=False,
        )
        return
    tail_logs_until_done(
        config,
        [path for path in paths if path],
        job_id,
        stop_on_done=stop_on_done,
    )
