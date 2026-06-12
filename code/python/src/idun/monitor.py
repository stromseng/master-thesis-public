"""IDUN resource monitor - CPU/GPU stats for running jobs."""

from __future__ import annotations

import time
from typing import Any, Optional, Union

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from idun.config import IdunConfig
from idun.slurm import parse_estimated_start, format_eta
from idun.ssh import run_ssh, run_ssh_via_jump

console = Console()


def _make_bar(percent: float, width: int = 20) -> Text:
    """Create a text-based progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled
    color = "red" if percent >= 90 else ("yellow" if percent >= 70 else "green")
    bar = Text()
    bar.append("[", style="dim")
    bar.append("=" * filled, style=f"bold {color}")
    bar.append(" " * empty, style="dim")
    bar.append("]", style="dim")
    return bar


def _get_running_jobs(config: IdunConfig) -> list[dict[str, str]]:
    """Get info about all running jobs including node and GPU allocation."""
    result = run_ssh(
        config,
        "squeue -u $USER --noheader -o '%i|%N|%j|%b' --states=RUNNING",
        check=False,
    )
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return []

    jobs: list[dict[str, str]] = []
    for line in output.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        batch_host = parts[1].strip()
        job_name = parts[2].strip() if len(parts) > 2 else "unknown"
        gres = parts[3].strip() if len(parts) > 3 else ""

        gpus = "0"
        gpu_type = ""
        if "gpu:" in gres:
            gres_part = gres.split("gpu:")[-1]
            if ":" in gres_part:
                gpu_parts = gres_part.split(":")
                gpu_type = gpu_parts[0]
                gpus = gpu_parts[1] if len(gpu_parts) > 1 else "1"
            else:
                gpus = gres_part

        jobs.append(
            {
                "job_id": job_id,
                "batch_host": batch_host,
                "job_name": job_name,
                "gpus": gpus,
                "gpu_type": gpu_type,
            }
        )
    return jobs


def _has_gpu(job: dict[str, str]) -> bool:
    """Check if the job has GPU resources allocated."""
    gpus = job.get("gpus", "0")
    gpu_type = job.get("gpu_type", "")
    if gpus in ("0", "", "?"):
        return False
    if gpu_type in ("", "N/A", "(null)"):
        try:
            return int(gpus) > 0
        except ValueError:
            return False
    return True


def _get_pending_jobs(config: IdunConfig) -> list[dict[str, str]]:
    """Get queued/pending jobs with estimated start times in a single SSH call."""
    cmd = (
        "echo '===PENDING==='; "
        "squeue -u $USER --noheader -o '%i|%T|%j|%m|%C|%b|%R|%V' --states=PENDING,CONFIGURING,RESIZING; "
        "echo '===ETA==='; "
        "squeue --start -u $USER --noheader -o '%i|%S' --states=PENDING,CONFIGURING,RESIZING"
    )
    result = run_ssh(config, cmd, check=False)
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return []

    sections = _split_sections(output)

    jobs: list[dict[str, str]] = []
    pending_text = sections.get("PENDING", "")
    for line in pending_text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        job_id, state, name, mem, cpus, gres, reason = parts[:7]
        submit_time = parts[7].strip() if len(parts) > 7 else ""

        # Parse GPU info from GRES (e.g. "gpu:a100:2")
        gpu_info = ""
        gres = gres.strip()
        if "gpu:" in gres:
            gres_part = gres.split("gpu:")[-1]
            gpu_info = gres_part

        jobs.append(
            {
                "id": job_id.strip(),
                "state": state.strip(),
                "name": name.strip(),
                "mem": mem.strip(),
                "cpus": cpus.strip(),
                "gpu_info": gpu_info,
                "reason": reason.strip(),
                "submit_time": submit_time,
                "eta": "-",
            }
        )

    # Parse ETAs
    eta_text = sections.get("ETA", "")
    start_times: dict[str, str] = {}
    for line in eta_text.strip().split("\n"):
        if "|" not in line:
            continue
        job_id, start_time = line.split("|", 1)
        if job_id.strip():
            start_times[job_id.strip()] = start_time.strip()

    for job in jobs:
        start_time = start_times.get(job["id"], "")
        if start_time:
            eta_value = format_eta(parse_estimated_start(config, start_time))
            if eta_value:
                job["eta"] = eta_value

    return jobs


def _render_pending_jobs_panel(jobs: list[dict[str, str]]) -> Panel:
    """Render pending/queued jobs as a panel with a compact table."""
    table = Table(border_style="dim", header_style="bold", box=None, padding=(0, 1))
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("State", style="yellow")
    table.add_column("ETA", justify="right")
    table.add_column("Resources", style="dim")
    table.add_column("Reason", style="bright_black")

    for job in jobs:
        eta = job.get("eta", "-")
        eta_style = "bold white" if eta != "-" else "dim"

        # Build compact resource string
        resources: list[str] = []
        gpu_info = job.get("gpu_info", "")
        if gpu_info:
            resources.append(f"gpu:{gpu_info}")
        cpus = job.get("cpus", "")
        if cpus and cpus != "0":
            resources.append(f"{cpus}cpu")
        mem = job.get("mem", "")
        if mem and mem != "0":
            resources.append(mem)

        table.add_row(
            job.get("id", "-"),
            job.get("name", "-"),
            job.get("state", "-"),
            Text(eta, style=eta_style),
            " ".join(resources) if resources else "-",
            job.get("reason", "-"),
        )

    count = len(jobs)
    title = f"[bold yellow]Queued[/] [dim]({count} job{'s' if count != 1 else ''})[/]"
    return Panel(table, title=title, border_style="yellow")


def _build_host_command(has_gpu: bool) -> str:
    """Build a single shell command that collects all stats from a compute node.

    Combines system, GPU, network, disk, and process stats into one SSH call
    with section markers for parsing.
    """
    cmd = """echo '===SYSTEM==='
echo "CPU_COUNT:$(nproc)"
echo "CPU_USAGE:$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')"
echo "LOAD_AVG:$(cat /proc/loadavg | awk '{print $1, $2, $3}')"
free -b | grep Mem | awk '{print "MEM_TOTAL:" $2 " MEM_USED:" $3 " MEM_FREE:" $4 " MEM_AVAILABLE:" $7}'
"""

    if has_gpu:
        cmd += """echo '===GPU==='
nvidia-smi --query-gpu=index,name,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
echo '===GPU_PROCS==='
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
"""

    cmd += """echo '===NET==='
awk 'NR>2 {
    split($0, a, ":");
    if (a[1] !~ /lo/) {
        split($2, b, " ");
        rx += b[1];
        tx += b[9];
    }
} END {
    print "RX_BYTES:" rx;
    print "TX_BYTES:" tx;
}' /proc/net/dev
echo '===DISK==='
awk '{
    if ($3 !~ /^(loop|ram|dm-)/ && $3 !~ /[0-9]$/) {
        read += $6;
        written += $10;
    }
} END {
    print "READ_BYTES:" read * 512;
    print "WRITE_BYTES:" written * 512;
}' /proc/diskstats
"""

    cmd += r"""echo '===PROCS==='
for pid in $(pgrep -f "python.*scripts\.idun" 2>/dev/null | head -5); do
    if [ -f /proc/$pid/stat ]; then
        cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null | awk '{print $1}')
        mem_usage=$(ps -p $pid -o %mem --no-headers 2>/dev/null | awk '{print $1}')
        state=$(ps -p $pid -o state --no-headers 2>/dev/null | awk '{print $1}')
        threads=$(ps -p $pid -o nlwp --no-headers 2>/dev/null | awk '{print $1}')
        cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | cut -c1-60)
        echo "PID:$pid"
        echo "CPU:$cpu_usage"
        echo "MEM:$mem_usage"
        echo "STATE:$state"
        echo "THREADS:$threads"
        echo "CMD:$cmdline"
        break
    fi
done
"""
    return cmd


def _split_sections(output: str) -> dict[str, str]:
    """Split batched SSH output into named sections."""
    sections: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for line in output.split("\n"):
        if line.startswith("===") and line.endswith("===") and len(line) > 6:
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines)
            current_name = line.strip("=")
            current_lines = []
        else:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_lines)

    return sections


def _parse_system_section(text: str) -> dict[str, Union[str, int, float]]:
    """Parse system stats from a section of batched output."""
    stats: dict[str, Union[str, int, float]] = {
        "cpu_count": "?",
        "cpu_usage": 0.0,
        "load_avg": "?",
        "mem_total": 0,
        "mem_available": 0,
    }
    for line in text.strip().split("\n"):
        if line.startswith("CPU_COUNT:"):
            stats["cpu_count"] = line.split(":")[1]
        elif line.startswith("CPU_USAGE:"):
            try:
                stats["cpu_usage"] = float(line.split(":")[1].replace(",", "."))
            except ValueError:
                pass
        elif line.startswith("LOAD_AVG:"):
            stats["load_avg"] = line.split(":", 1)[1]
        elif line.startswith("MEM_TOTAL:"):
            parts = line.split()
            for part in parts:
                if part.startswith("MEM_TOTAL:"):
                    stats["mem_total"] = int(part.split(":")[1])
                elif part.startswith("MEM_AVAILABLE:"):
                    stats["mem_available"] = int(part.split(":")[1])
    return stats


def _parse_net_section(text: str) -> dict[str, int]:
    """Parse network stats from a section of batched output."""
    stats: dict[str, int] = {"rx_bytes": 0, "tx_bytes": 0}
    for line in text.strip().split("\n"):
        if line.startswith("RX_BYTES:"):
            try:
                stats["rx_bytes"] = int(line.split(":")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("TX_BYTES:"):
            try:
                stats["tx_bytes"] = int(line.split(":")[1])
            except (ValueError, IndexError):
                pass
    return stats


def _parse_disk_section(text: str) -> dict[str, int]:
    """Parse disk stats from a section of batched output."""
    stats: dict[str, int] = {"read_bytes": 0, "write_bytes": 0}
    for line in text.strip().split("\n"):
        if line.startswith("READ_BYTES:"):
            try:
                stats["read_bytes"] = int(line.split(":")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("WRITE_BYTES:"):
            try:
                stats["write_bytes"] = int(line.split(":")[1])
            except (ValueError, IndexError):
                pass
    return stats


def _parse_proc_section(text: str) -> Optional[dict[str, Union[str, float, int]]]:
    """Parse process stats from a section of batched output."""
    stats: dict[str, Union[str, float, int]] = {}
    for line in text.strip().split("\n"):
        if line.startswith("PID:"):
            try:
                stats["pid"] = int(line.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("CPU:"):
            try:
                stats["cpu_percent"] = float(line.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("MEM:"):
            try:
                stats["mem_percent"] = float(line.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("STATE:"):
            state_char = line.split(":", 1)[1].strip()
            state_map = {
                "R": "Running",
                "S": "Sleeping",
                "D": "Disk Wait",
                "Z": "Zombie",
                "T": "Stopped",
            }
            stats["state"] = state_map.get(state_char, state_char)
        elif line.startswith("THREADS:"):
            try:
                stats["threads"] = int(line.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("CMD:"):
            stats["cmdline"] = line.split(":", 1)[1].strip()
    return stats if stats else None


def _collect_host_stats(config: IdunConfig, host: str, has_gpu: bool) -> dict[str, Any]:
    """Collect all stats from a compute node in a single SSH call.

    Returns dict with keys: system, gpu, gpu_procs, network, disk, process.
    Previously this required 6 separate SSH calls; now it's 1.
    """
    cmd = _build_host_command(has_gpu)
    result = run_ssh_via_jump(config, host, cmd)
    if result.returncode != 0:
        return {}

    sections = _split_sections(result.stdout or "")
    stats: dict[str, Any] = {}

    if "SYSTEM" in sections:
        stats["system"] = _parse_system_section(sections["SYSTEM"])

    if has_gpu and "GPU" in sections:
        gpu_text = sections["GPU"].strip()
        stats["gpu"] = gpu_text if gpu_text else None

    if has_gpu and "GPU_PROCS" in sections:
        gpu_procs_text = sections["GPU_PROCS"].strip()
        stats["gpu_procs"] = gpu_procs_text if gpu_procs_text else None

    if "NET" in sections:
        stats["network"] = _parse_net_section(sections["NET"])

    if "DISK" in sections:
        stats["disk"] = _parse_disk_section(sections["DISK"])

    if "PROCS" in sections:
        stats["process"] = _parse_proc_section(sections["PROCS"])

    return stats


def _get_all_job_timings(
    config: IdunConfig, job_ids: list[str]
) -> dict[str, dict[str, str]]:
    """Get timing for all jobs in a single SSH call to the login node."""
    if not job_ids:
        return {}

    cmd = "; ".join(
        f'echo "===JOB:{jid}==="; scontrol show job {jid} 2>/dev/null | grep -E "RunTime|TimeLimit"'
        for jid in job_ids
    )
    result = run_ssh(config, cmd, check=False)
    if result.returncode != 0 or not result.stdout:
        return {}

    timings: dict[str, dict[str, str]] = {}
    current_job: Optional[str] = None
    current_text = ""

    for line in (result.stdout or "").split("\n"):
        if line.startswith("===JOB:") and line.endswith("==="):
            if current_job and current_text:
                timings[current_job] = _parse_timing_text(current_text)
            current_job = line[len("===JOB:") : -len("===")]
            current_text = ""
        else:
            current_text += line + "\n"

    if current_job and current_text:
        timings[current_job] = _parse_timing_text(current_text)

    return timings


def _parse_timing_text(text: str) -> dict[str, str]:
    """Parse timing info from scontrol output text."""
    timing: dict[str, str] = {}
    if "RunTime=" in text:
        try:
            timing["runtime"] = text.split("RunTime=")[1].split()[0]
        except (IndexError, ValueError):
            pass
    if "TimeLimit=" in text:
        try:
            timing["time_limit"] = text.split("TimeLimit=")[1].split()[0]
        except (IndexError, ValueError):
            pass
    return timing


def _render_gpu_table(gpu_output: str) -> Table:
    """Render GPU stats as a table."""
    gpu_table = Table(border_style="dim", header_style="bold", box=None)
    gpu_table.add_column("GPU", style="cyan", justify="center")
    gpu_table.add_column("Name", style="white")
    gpu_table.add_column("Power", justify="right")
    gpu_table.add_column("GPU Load", justify="left", min_width=28)
    gpu_table.add_column("VRAM", justify="left", min_width=28)
    gpu_table.add_column("Temp", justify="right")

    for line in gpu_output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 9:
            (
                idx,
                name,
                power,
                power_limit,
                gpu_util,
                mem_util,
                mem_used,
                mem_total,
                temp,
            ) = parts[:9]
            power_pct = float(power) / float(power_limit) * 100
            power_color = (
                "green" if power_pct < 70 else ("yellow" if power_pct < 90 else "red")
            )
            power_str = Text(
                f"{float(power):.0f}/{float(power_limit):.0f}W", style=power_color
            )

            gpu_pct = float(gpu_util)
            gpu_bar = _make_bar(gpu_pct)
            gpu_text = Text()
            gpu_text.append_text(gpu_bar)
            gpu_text.append(f" {gpu_pct:5.1f}%", style="bold")

            mem_pct = (
                float(mem_used) / float(mem_total) * 100 if float(mem_total) > 0 else 0
            )
            mem_bar = _make_bar(mem_pct)
            mem_text = Text()
            mem_text.append_text(mem_bar)
            mem_text.append(
                f" {float(mem_used) / 1024:.1f}/{float(mem_total) / 1024:.0f}G",
                style="bold",
            )

            temp_val = float(temp)
            temp_color = (
                "green" if temp_val < 70 else ("yellow" if temp_val < 85 else "red")
            )
            temp_str = Text(f"{temp_val:.0f}C", style=temp_color)

            gpu_table.add_row(idx, name[:18], power_str, gpu_text, mem_text, temp_str)
    return gpu_table


def _render_cpu_table(sys_stats: dict[str, Union[str, int, float]]) -> Table:
    """Render CPU/memory stats as a table."""
    cpu_table = Table.grid(padding=(0, 2))
    cpu_table.add_column(justify="right", style="bold")
    cpu_table.add_column(justify="left", min_width=35)
    cpu_table.add_column(justify="left")

    raw_cpu_usage = sys_stats.get("cpu_usage", 0)
    cpu_usage = float(raw_cpu_usage) if isinstance(raw_cpu_usage, (int, float)) else 0.0
    cpu_count = sys_stats.get("cpu_count", "?")
    load_avg = sys_stats.get("load_avg", "?")
    cpu_bar = _make_bar(cpu_usage)
    cpu_text = Text()
    cpu_text.append_text(cpu_bar)
    cpu_text.append(f" {cpu_usage:5.1f}%", style="bold")
    cpu_table.add_row("CPU", cpu_text, f"[dim]{cpu_count} cores | Load: {load_avg}[/]")

    raw_mem_total = sys_stats.get("mem_total", 0)
    raw_mem_available = sys_stats.get("mem_available", 0)
    mem_total = int(raw_mem_total) if isinstance(raw_mem_total, (int, float)) else 0
    mem_available = (
        int(raw_mem_available) if isinstance(raw_mem_available, (int, float)) else 0
    )
    if mem_total > 0:
        mem_used_actual = mem_total - mem_available
        mem_pct = (mem_used_actual / mem_total) * 100
        mem_bar = _make_bar(mem_pct)
        mem_text = Text()
        mem_text.append_text(mem_bar)
        mem_used_gb = mem_used_actual / (1024**3)
        mem_total_gb = mem_total / (1024**3)
        mem_text.append(f" {mem_used_gb:.1f}/{mem_total_gb:.0f}GB", style="bold")
        cpu_table.add_row("RAM", mem_text, f"[dim]{mem_pct:.1f}% used[/]")

    return cpu_table


def _render_network_table(
    net_stats: Optional[dict[str, int]],
    prev_net_stats: Optional[dict[str, int]],
    interval: float,
) -> Optional[Table]:
    """Render network I/O stats as a table.

    Args:
        net_stats: Current network stats (rx_bytes, tx_bytes)
        prev_net_stats: Previous network stats for rate calculation
        interval: Time interval in seconds between measurements

    Returns:
        Table with network stats or None if no data
    """
    if not net_stats:
        return None

    net_table = Table.grid(padding=(0, 2))
    net_table.add_column(justify="right", style="bold")
    net_table.add_column(justify="left", min_width=35)
    net_table.add_column(justify="left")

    # Format bytes to human-readable
    def _format_bytes(bytes_val: int) -> str:
        if bytes_val >= 1024**3:
            return f"{bytes_val / (1024**3):.2f}GB"
        elif bytes_val >= 1024**2:
            return f"{bytes_val / (1024**2):.1f}MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.1f}KB"
        else:
            return f"{bytes_val}B"

    # Calculate rates if we have previous stats
    rx_rate_str = "?"
    tx_rate_str = "?"
    activity_indicator = ""

    if prev_net_stats and interval > 0:
        rx_diff = net_stats["rx_bytes"] - prev_net_stats.get("rx_bytes", 0)
        tx_diff = net_stats["tx_bytes"] - prev_net_stats.get("tx_bytes", 0)

        # Convert to rate per second
        rx_rate = rx_diff / interval
        tx_rate = tx_diff / interval

        rx_rate_str = f"{_format_bytes(int(rx_rate))}/s"
        tx_rate_str = f"{_format_bytes(int(tx_rate))}/s"

        # Show activity indicator
        total_rate = rx_rate + tx_rate
        if total_rate > 1024 * 1024:  # > 1 MB/s
            activity_indicator = " [bold green]●[/] ACTIVE"
        elif total_rate > 1024:  # > 1 KB/s
            activity_indicator = " [yellow]●[/] active"
        else:
            activity_indicator = " [dim]○[/] idle"

    # Display RX (download)
    rx_text = Text()
    rx_text.append(f"↓ {rx_rate_str:>12}", style="cyan")
    rx_total = _format_bytes(net_stats["rx_bytes"])
    net_table.add_row("NET RX", rx_text, f"[dim]Total: {rx_total}[/]")

    # Display TX (upload)
    tx_text = Text()
    tx_text.append(f"↑ {tx_rate_str:>12}", style="magenta")
    tx_total = _format_bytes(net_stats["tx_bytes"])
    net_table.add_row(
        "NET TX", tx_text, f"[dim]Total: {tx_total}[/]{activity_indicator}"
    )

    return net_table


def _render_disk_table(
    disk_stats: Optional[dict[str, int]],
    prev_disk_stats: Optional[dict[str, int]],
    interval: float,
) -> Optional[Table]:
    """Render disk I/O stats as a table.

    Args:
        disk_stats: Current disk stats (read_bytes, write_bytes)
        prev_disk_stats: Previous disk stats for rate calculation
        interval: Time interval in seconds between measurements

    Returns:
        Table with disk I/O stats or None if no data
    """
    if not disk_stats:
        return None

    disk_table = Table.grid(padding=(0, 2))
    disk_table.add_column(justify="right", style="bold")
    disk_table.add_column(justify="left", min_width=35)
    disk_table.add_column(justify="left")

    # Format bytes to human-readable
    def _format_bytes(bytes_val: int) -> str:
        if bytes_val >= 1024**3:
            return f"{bytes_val / (1024**3):.2f}GB"
        elif bytes_val >= 1024**2:
            return f"{bytes_val / (1024**2):.1f}MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.1f}KB"
        else:
            return f"{bytes_val}B"

    # Calculate rates if we have previous stats
    read_rate_str = "?"
    write_rate_str = "?"
    activity_indicator = ""

    if prev_disk_stats and interval > 0:
        read_diff = disk_stats["read_bytes"] - prev_disk_stats.get("read_bytes", 0)
        write_diff = disk_stats["write_bytes"] - prev_disk_stats.get("write_bytes", 0)

        # Convert to rate per second
        read_rate = read_diff / interval
        write_rate = write_diff / interval

        read_rate_str = f"{_format_bytes(int(read_rate))}/s"
        write_rate_str = f"{_format_bytes(int(write_rate))}/s"

        # Show activity indicator
        total_rate = read_rate + write_rate
        if total_rate > 10 * 1024 * 1024:  # > 10 MB/s
            activity_indicator = " [bold green]●[/] ACTIVE"
        elif total_rate > 1024 * 1024:  # > 1 MB/s
            activity_indicator = " [yellow]●[/] active"
        else:
            activity_indicator = " [dim]○[/] idle"

    # Display READ
    read_text = Text()
    read_text.append(f"↓ {read_rate_str:>12}", style="cyan")
    read_total = _format_bytes(disk_stats["read_bytes"])
    disk_table.add_row("DISK RD", read_text, f"[dim]Total: {read_total}[/]")

    # Display WRITE
    write_text = Text()
    write_text.append(f"↑ {write_rate_str:>12}", style="magenta")
    write_total = _format_bytes(disk_stats["write_bytes"])
    disk_table.add_row(
        "DISK WR", write_text, f"[dim]Total: {write_total}[/]{activity_indicator}"
    )

    return disk_table


def _render_process_table(
    proc_stats: Optional[dict[str, Union[str, float, int]]],
) -> Optional[Table]:
    """Render process-level stats as a table.

    Args:
        proc_stats: Process stats (pid, cpu_percent, mem_percent, state, threads)

    Returns:
        Table with process stats or None if no data
    """
    if not proc_stats or "pid" not in proc_stats:
        return None

    proc_table = Table.grid(padding=(0, 2))
    proc_table.add_column(justify="right", style="bold")
    proc_table.add_column(justify="left", min_width=35)
    proc_table.add_column(justify="left")

    pid = proc_stats.get("pid", "?")
    cpu_pct = float(proc_stats.get("cpu_percent", 0.0))
    mem_pct = float(proc_stats.get("mem_percent", 0.0))
    state = str(proc_stats.get("state", "Unknown"))
    threads = proc_stats.get("threads", 0)

    # Format CPU usage with color
    cpu_text = Text()
    cpu_color = "red" if cpu_pct >= 90 else ("yellow" if cpu_pct >= 70 else "green")
    cpu_text.append(f"{cpu_pct:>5.1f}%", style=cpu_color)

    # Format memory usage with color
    mem_text = Text()
    mem_color = "red" if mem_pct >= 90 else ("yellow" if mem_pct >= 70 else "green")
    mem_text.append(f"{mem_pct:>5.1f}%", style=mem_color)

    # State indicator
    state_color = (
        "green" if state == "Running" else ("yellow" if state == "Disk Wait" else "dim")
    )
    state_text = Text(state, style=state_color)

    proc_table.add_row(
        "PROCESS",
        Text(f"PID: {pid}  |  Threads: {threads}"),
        f"[dim]State: [/]{state_text}",
    )

    cpu_label = Text("CPU: ")
    cpu_label.append_text(cpu_text)
    mem_label = Text("Memory: ")
    mem_label.append_text(mem_text)
    proc_table.add_row("", cpu_label, mem_label)

    return proc_table


def _render_gpu_processes(gpu_proc_output: Optional[str]) -> Optional[Table]:
    """Render GPU process information as a table.

    Args:
        gpu_proc_output: Output from nvidia-smi compute apps query

    Returns:
        Table with GPU processes or None if no data
    """
    if not gpu_proc_output:
        return None

    lines = gpu_proc_output.strip().split("\n")
    if not lines or not lines[0].strip():
        return None

    proc_table = Table(
        border_style="dim", header_style="bold cyan", box=None, show_header=False
    )
    proc_table.add_column("Label", style="bold")
    proc_table.add_column("Info", style="white")

    for line in lines[:3]:  # Show up to 3 processes
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            pid, name, mem = parts[0], parts[1], parts[2]
            # Shorten process name
            proc_name = name.split("/")[-1] if "/" in name else name
            proc_table.add_row(
                "GPU PROC", f"PID {pid}: {proc_name[:30]} ({mem}MB VRAM)"
            )

    return proc_table


def _render_job_panel(
    job: dict[str, str],
    gpu_output: Optional[str],
    sys_stats: Optional[dict[str, Union[str, int, float]]],
    net_stats: Optional[dict[str, int]],
    prev_net_stats: Optional[dict[str, int]],
    disk_stats: Optional[dict[str, int]],
    prev_disk_stats: Optional[dict[str, int]],
    proc_stats: Optional[dict[str, Union[str, float, int]]],
    gpu_proc_output: Optional[str],
    timing: Optional[dict[str, str]],
    interval: float,
) -> Panel:
    """Render a single job's monitoring panel."""
    job_id = job["job_id"]
    job_name = job.get("job_name", "")
    batch_host = job["batch_host"]
    has_gpu = _has_gpu(job)
    gpus = job.get("gpus", "0")
    gpu_type = job.get("gpu_type", "")

    header_parts = [f"Job: [bold cyan]{job_id}[/]"]
    if job_name:
        header_parts.append(f"[bright_black]({job_name})[/]")
    header_parts.append(f"Node: [bold green]{batch_host}[/]")
    if has_gpu:
        header_parts.append(f"GPUs: [bold yellow]{gpu_type}:{gpus}[/]")
    else:
        header_parts.append("[bright_black]CPU only[/]")

    # Add timing info to header
    if timing:
        runtime = timing.get("runtime", "?")
        time_limit = timing.get("time_limit", "?")
        header_parts.append(f"[bright_black]Time: {runtime}/{time_limit}[/]")

    content_parts: list[Table] = []

    # GPU stats
    if has_gpu and gpu_output:
        content_parts.append(_render_gpu_table(gpu_output))

    # GPU process stats
    gpu_proc_table = _render_gpu_processes(gpu_proc_output)
    if gpu_proc_table:
        content_parts.append(gpu_proc_table)

    # CPU/Memory stats
    if sys_stats:
        content_parts.append(_render_cpu_table(sys_stats))

    # Process-level stats
    proc_table = _render_process_table(proc_stats)
    if proc_table:
        content_parts.append(proc_table)

    # Network stats
    net_table = _render_network_table(net_stats, prev_net_stats, interval)
    if net_table:
        content_parts.append(net_table)

    # Disk stats
    disk_table = _render_disk_table(disk_stats, prev_disk_stats, interval)
    if disk_table:
        content_parts.append(disk_table)

    return Panel(
        Group(*content_parts) if content_parts else "[dim]No stats available[/]",
        title="  ".join(header_parts),
        border_style="blue",
    )


def _build_display(
    config: IdunConfig,
    prev_network_stats: Optional[dict[str, dict[str, int]]] = None,
    prev_disk_stats: Optional[dict[str, dict[str, int]]] = None,
    interval: float = 2.0,
) -> tuple[Group, int, dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Build the monitor display.

    Args:
        config: IDUN configuration
        prev_network_stats: Previous network stats per host for rate calculation
        prev_disk_stats: Previous disk stats per host for rate calculation
        interval: Time interval since last measurement (seconds)

    Returns:
        (renderable, job_count, current_network_stats, current_disk_stats)
    """
    jobs = _get_running_jobs(config)
    pending_jobs = _get_pending_jobs(config)

    if not jobs and not pending_jobs:
        return Group("[yellow]No running or queued jobs found[/]"), 0, {}, {}

    if prev_network_stats is None:
        prev_network_stats = {}
    if prev_disk_stats is None:
        prev_disk_stats = {}

    current_network_stats: dict[str, dict[str, int]] = {}
    current_disk_stats: dict[str, dict[str, int]] = {}
    panels: list[Panel] = []

    # Batch all job timing queries into a single SSH call to login node
    valid_jobs = [
        j
        for j in jobs
        if j["batch_host"]
        and j["batch_host"] not in {"(null)", "(none)", "unknown", ""}
    ]
    all_timings = _get_all_job_timings(config, [j["job_id"] for j in valid_jobs])

    for job in valid_jobs:
        host = job["batch_host"]
        job_id = job["job_id"]
        has_gpu = _has_gpu(job)

        # Collect all stats from compute node in a single SSH call
        host_stats = _collect_host_stats(config, host, has_gpu)

        sys_stats = host_stats.get("system")
        gpu_output = host_stats.get("gpu")
        gpu_proc_output = host_stats.get("gpu_procs")
        net_stats = host_stats.get("network")
        disk_stats = host_stats.get("disk")
        proc_stats = host_stats.get("process")
        timing = all_timings.get(job_id)

        # Store current stats for next iteration
        if net_stats:
            current_network_stats[host] = net_stats
        if disk_stats:
            current_disk_stats[host] = disk_stats

        # Get previous stats for this host
        prev_net_stats = prev_network_stats.get(host)
        prev_disk_stat = prev_disk_stats.get(host)

        panels.append(
            _render_job_panel(
                job,
                gpu_output,
                sys_stats,
                net_stats,
                prev_net_stats,
                disk_stats,
                prev_disk_stat,
                proc_stats,
                gpu_proc_output,
                timing,
                interval,
            )
        )

    # Add pending jobs panel below running job panels
    if pending_jobs:
        panels.append(_render_pending_jobs_panel(pending_jobs))

    total_count = len(valid_jobs) + len(pending_jobs)
    if not panels:
        return Group("[yellow]No accessible jobs[/]"), 0, {}, {}

    return Group(*panels), total_count, current_network_stats, current_disk_stats


def show_monitor(config: IdunConfig, watch: int = 2) -> int:
    """Show resource monitor for all running jobs.

    Args:
        config: IDUN configuration with jump_host and ssh_user
        watch: Refresh interval in seconds (default: 2). Set to 0 for single run.

    Returns:
        0 on success, 1 if no jobs found
    """
    if watch == 0:
        display, count, _, _ = _build_display(config)
        console.print(display)
        return 0 if count > 0 else 1

    try:
        prev_network_stats: dict[str, dict[str, int]] = {}
        prev_disk_stats: dict[str, dict[str, int]] = {}
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                display, _, current_network_stats, current_disk_stats = _build_display(
                    config, prev_network_stats, prev_disk_stats, float(watch)
                )
                footer = Text(
                    f"\nRefreshing every {watch}s (Ctrl+C to stop)", style="dim"
                )
                live.update(Group(display, footer))
                prev_network_stats = current_network_stats
                prev_disk_stats = current_disk_stats
                time.sleep(watch)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped[/]")
        return 0
