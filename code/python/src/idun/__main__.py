#!/usr/bin/env python3
"""IDUN cluster CLI commands."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel

from idun.config import (
    IdunConfig,
    load_config,
    REMOTE_REPO_PATH,
    REMOTE_REPO_DISPLAY,
    QUEUED_STATES,
    GPU_VRAM_CONSTRAINTS,
    GPU_TYPE_VRAM_OPTIONS,
)
from idun.ssh import (
    run_ssh,
    open_ssh_shell,
    open_direct_ssh,
    parse_port_forwards,
    PortForward,
    list_python_processes,
    kill_processes,
)
from idun.slurm import (
    wait_for_running,
    get_job_state_and_host,
    get_batch_script,
    build_slurm_script,
    submit_slurm,
    run_on_existing_job,
)
from idun.jobs import (
    fetch_jobs,
    get_active_nodes,
    render_jobs,
    follow_job_logs,
    fetch_reuse_logs,
    fetch_slurm_logs,
    tail_reuse_log,
    tail_slurm_log,
)
from idun.overlay import (
    resolve_repo_root,
    collect_changed_files,
    create_overlay_bundle,
    upload_overlay_bundle,
    reset_remote_repo,
)
from idun.prompts import (
    prompt_choice,
    prompt_text,
    prompt_int,
    prompt_confirm,
    prompt_gpu_spec,
    prompt_idun_script,
    module_from_script,
    script_display_path,
    select_job,
    choose_job,
    normalize_machine,
    normalize_post_action,
    normalize_log_choice,
    normalize_gpu_vram,
    normalize_gpu_type,
    format_vram_label,
)
from idun.monitor import show_monitor
from idun.vllm import (
    VLLM_LOG_DIR,
    DEFAULT_PORT,
    VllmServer,
    check_vllm_setup_on_node,
    check_backend_venv_on_node,
    check_vllm_setup,
    delete_backend_venv_on_node,
    build_setup_commands,
    build_vllm_slurm_script,
    start_vllm_server,
    stop_vllm_server,
    get_all_servers,
    get_backend_venv_dir,
    check_server_health_bulk,
    inspect_server,
    render_servers,
    prompt_model,
    prompt_server,
    list_cached_models,
    download_model,
    delete_cached_model_folders,
)

app = typer.Typer(add_completion=False)
console = Console()

COMMON_FORWARD_PORTS = [
    {"name": "8001 - Python dev server", "value": 8001},
    {"name": "8000 - vLLM", "value": 8000},
]


def _run_setup_on_node(config: IdunConfig, host: str, setup_script: str) -> int:
    """Run vLLM/SGLang setup directly on the compute node via SSH jump."""
    import subprocess

    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )
    compute_target = f"{config.ssh_user}@{host}" if config.ssh_user else host

    result = subprocess.run(
        [
            "ssh",
            "-tt",
            "-J",
            jump_target,
            compute_target,
            f"bash -lc {shlex.quote(setup_script)}",
        ]
    )
    return result.returncode


def _prompt_port_forwards() -> list[PortForward]:
    """Interactively prompt for port forwards with common and custom options."""
    if not prompt_confirm("Set up port forwarding?", default=False):
        return []
    choices = [*COMMON_FORWARD_PORTS, {"name": "Custom ports...", "value": "custom"}]
    selected = inquirer.checkbox(  # pyright: ignore[reportPrivateImportUsage]
        message="Select ports to forward",
        choices=choices,
    ).execute()
    forwards: list[PortForward] = []
    for item in selected:
        if item == "custom":
            entry = prompt_text("Ports (e.g. 8888,6006 or 8888:6006)", "").strip()
            if entry:
                forwards.extend(parse_port_forwards(entry))
        else:
            forwards.append(PortForward(local=item, remote=item))
    return forwards


def _quote_arg(value: str, *, force: bool = False) -> str:
    """Quote argument for shell."""
    if not force:
        return shlex.quote(value)
    if value == "":
        return "''"
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _format_replay_command(
    *,
    mode: str,
    script_display: str | None,
    script_args: str | None,
    machine: str,
    gpus: int,
    gpu_type: str,
    gpu_vram: str,
    gpu_constraint_override: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    job_name: str,
    post_action: str,
    log_choice: str,
    keep_alive: bool,
    notify: bool,
    tunnels: list[str],
    begin_time: str | None = None,
) -> str:
    """Format command to replay this submission."""
    cmd: list[tuple[str, bool]] = [("just", False), ("idun", False), ("submit", False)]
    if mode == "shell":
        cmd.append(("--shell", False))
    if mode == "script" and script_display:
        cmd.extend([("--script", False), (script_display, False)])
    if script_args:
        cmd.extend([("--args", False), (script_args, True)])
    cmd.extend([("--machine", False), (machine.lower(), False)])
    if machine == "GPU":
        cmd.extend([("--gpus", False), (str(gpus), False)])
        if gpu_type:
            cmd.extend([("--gpu-type", False), (gpu_type, False)])
        if gpu_constraint_override:
            cmd.extend([("--gpu-constraint", False), (gpu_constraint_override, False)])
        elif gpu_vram:
            cmd.extend([("--gpu-vram", False), (gpu_vram, False)])
    if begin_time:
        cmd.extend([("--begin", False), (begin_time, False)])
    cmd.extend([("--time-limit", False), (time_limit, True)])
    cmd.extend([("--cpus-per-task", False), (str(cpus_per_task), False)])
    cmd.extend([("--mem", False), (mem, False)])
    cmd.extend([("--job-name", False), (job_name, False)])
    cmd.extend([("--post-action", False), (post_action, False)])
    if post_action == "logs":
        cmd.extend([("--log", False), (log_choice, False)])
    cmd.append(("--keep-alive" if keep_alive else "--no-keep-alive", False))
    cmd.append(("--notify" if notify else "--no-notify", False))
    for tunnel in tunnels:
        cmd.extend([("--tunnel", False), (tunnel, False)])
    return " ".join(_quote_arg(value, force=force) for value, force in cmd)


def _list_actions(config: IdunConfig, jobs: list[dict[str, str]]) -> None:
    """Interactive job action menu."""
    while True:
        job = choose_job(config, jobs)
        if job is None:
            return
        action = prompt_choice(
            "Action",
            ["Cancel", "SSH", "Processes", "Logs", "Script", "Wait", "Back"],
        )
        job_id = job["id"]
        if action == "Back":
            continue
        if action == "Cancel":
            if prompt_confirm(f"Cancel job {job_id}?", default=False):
                _ = run_ssh(config, f"scancel {shlex.quote(job_id)}")
                console.print(f"Cancelled job {job_id}")
                jobs[:] = fetch_jobs(config)
                render_jobs(jobs)
            continue
        if action == "SSH":
            host = job.get("node")
            if not host or host in {"(null)", "(none)", "unknown"}:
                host = wait_for_running(config, job_id)
            forwards = _prompt_port_forwards() if sys.stdin.isatty() else []
            console.print(f"Connecting to {host}...")
            open_direct_ssh(config, host, forwards=forwards)
            continue
        if action == "Processes":
            host = job.get("node")
            if not host or host in {"(null)", "(none)", "unknown"}:
                console.print("Job not running yet, cannot list processes")
                continue
            console.print(f"Fetching processes on {host}...")
            processes = list_python_processes(config, host)
            if not processes:
                console.print("No Python processes found")
                continue
            choices = [
                {"name": f"{p.pid} | {p.start_time} | {p.command}", "value": p.pid}
                for p in processes
            ]
            choices.insert(0, {"name": "Kill all", "value": "all"})
            choices.append({"name": "Back", "value": None})
            selected = inquirer.select(
                message=f"Python processes ({len(processes)})",
                choices=choices,
            ).execute()
            if selected is None:
                continue
            if selected == "all":
                if prompt_confirm(
                    f"Kill all {len(processes)} processes?", default=False
                ):
                    pids = [p.pid for p in processes]
                    if kill_processes(config, host, pids):
                        console.print(f"Killed {len(pids)} processes")
                    else:
                        console.print("Failed to kill some processes")
            else:
                proc = next((p for p in processes if p.pid == selected), None)
                if proc and prompt_confirm(
                    f"Kill {proc.pid} ({proc.command})?", default=False
                ):
                    if kill_processes(config, host, [selected]):
                        console.print(f"Killed process {selected}")
                    else:
                        console.print(f"Failed to kill process {selected}")
            continue
        if action == "Logs":
            which = prompt_choice(
                "Which log?", ["stdout", "stderr", "both"], default="stdout"
            )
            if job.get("state") in QUEUED_STATES:
                _ = wait_for_running(config, job_id)
            follow_job_logs(config, job_id, which=which)
            continue
        if action == "Script":
            content = get_batch_script(config, job_id)
            if content is None:
                console.print(f"Could not retrieve batch script for job {job_id}")
                console.print(
                    "Note: Scripts are only available for pending/running jobs"
                )
            else:
                console.print(
                    Panel(
                        content,
                        title=f"Batch Script for Job {job_id}",
                        border_style="blue",
                    )
                )
            continue
        if action == "Wait":
            host = wait_for_running(config, job_id)
            open_ssh_shell(config, host, job_id=job_id)


@app.command()
def submit(
    script: Path | None = typer.Option(
        None,
        "--script",
        help="Python script under code/python/scripts/idun to run as a module",
    ),
    script_args: str | None = typer.Option(
        None,
        "--args",
        help="Arguments to pass to the script (e.g. '--test --verbose')",
    ),
    shell: bool = typer.Option(
        False, "--shell", help="Use interactive shell job (sleep 30d)"
    ),
    wait: bool = typer.Option(
        False, "--wait", help="Wait for RUNNING and open a shell"
    ),
    machine: str | None = typer.Option(
        None, "--machine", help="Machine type: GPU or CPU"
    ),
    gpus: int | None = typer.Option(None, "--gpus", help="GPU count"),
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="GPU type (leave empty for any)"
    ),
    gpu_vram: str | None = typer.Option(
        None, "--gpu-vram", help="GPU memory size: 16g, 32g, 40g, 80g"
    ),
    gpu_constraint: str | None = typer.Option(
        None,
        "--gpu-constraint",
        help="Slurm constraint override (e.g. v100|a100, gpu80g, a100&sxm4)",
    ),
    partition: str | None = typer.Option(None, "--partition", help="Partition"),
    account: str | None = typer.Option(None, "--account", help="Account"),
    time_limit: str | None = typer.Option(
        None, "--time-limit", help="Time limit (D-HH:MM:SS)"
    ),
    cpus_per_task: int | None = typer.Option(
        None, "--cpus-per-task", "--cpus", help="CPUs per task"
    ),
    mem: str | None = typer.Option(None, "--mem", help="Memory (e.g. 32G)"),
    job_name: str | None = typer.Option(None, "--job-name", help="Job name"),
    tunnel: list[str] = typer.Option(
        None, "--tunnel", help="Port forward (port or local:remote)"
    ),
    post_action: str | None = typer.Option(
        None, "--post-action", help="Post-submit action: background, logs, shell"
    ),
    log_choice: str | None = typer.Option(
        None, "--log", help="Which log to stream: stdout, stderr, or both"
    ),
    keep_alive: bool | None = typer.Option(
        None,
        "--keep-alive/--no-keep-alive",
        help="Keep allocation alive after script finishes",
    ),
    notify: bool | None = typer.Option(
        None,
        "--notify/--no-notify",
        help="Send email notification when job starts and ends",
    ),
    begin: str | None = typer.Option(
        None,
        "--begin",
        help="Defer job start (e.g. 2026-01-30T10:00:00 or 'tomorrow')",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Comma-separated list of nodes to exclude (e.g. idun-04-01,idun-04-02)",
    ),
    no_exclude: bool = typer.Option(
        False,
        "--no-exclude",
        help="Skip the exclude-nodes prompt",
    ),
) -> None:
    """Submit a job to IDUN."""
    config = load_config()
    if shell and script is not None:
        raise typer.BadParameter("Use either --script or --shell, not both")

    mode = "shell" if shell else "script"
    if script is None and not shell:
        mode = prompt_choice(
            "Job mode",
            ["Run a Python script (module)", "Interactive shell (sleep 30d)"],
            default="Run a Python script (module)",
        )
        mode = "script" if mode.startswith("Run") else "shell"

    repo_root: Path | None = None
    run_module: str | None = None
    overlay_path: str | None = None
    script_display: str | None = None

    if mode == "script":
        repo_root = resolve_repo_root()
        if script is None:
            script = prompt_idun_script(repo_root)
        script = script.expanduser()
        if not script.exists():
            raise typer.BadParameter(f"Script not found: {script}")
        run_module = module_from_script(repo_root, script)
        script_display = script_display_path(repo_root, script)

        # Prompt for script arguments if not provided
        if script_args is None and sys.stdin.isatty():
            script_args = prompt_text("Script arguments (optional)", "").strip()
            if not script_args:
                script_args = None

        # Check for running jobs FIRST - reuse flow uses isolated overlays
        # so we don't need to reset the base repo
        if sys.stdin.isatty():
            running_jobs = [
                j for j in fetch_jobs(config) if j.get("state") == "RUNNING"
            ]
            if running_jobs and prompt_confirm(
                "Run on an existing RUNNING instance?", default=False
            ):
                job = choose_job(config, running_jobs)
                if job is not None:
                    # Collect changes and create overlay (no repo reset needed)
                    changed, deleted = collect_changed_files(repo_root)
                    bundle = create_overlay_bundle(repo_root, changed, deleted)
                    if bundle:
                        try:
                            overlay_path = upload_overlay_bundle(config, bundle)
                        finally:
                            try:
                                bundle.unlink()
                            except FileNotFoundError:
                                pass

                    job_id = job["id"]
                    job_host = job.get("node")
                    run_mode = prompt_choice(
                        "Run mode",
                        ["Run in background", "Run in foreground (stream output)"],
                        default="Run in background",
                    )
                    run_in_background = run_mode.startswith("Run in background")
                    save_artifacts = prompt_confirm(
                        "Save artifacts (changed/created files) after run?",
                        default=True,
                    )
                    # Extract just the filename from script_display (e.g., "test.py")
                    script_filename = (
                        Path(script_display).name if script_display else None
                    )
                    log_path, run_id = run_on_existing_job(
                        config,
                        job_id,
                        run_module,
                        overlay_path,
                        REMOTE_REPO_PATH,
                        background=run_in_background,
                        script_name=script_filename,
                        script_args=script_args,
                        host=job_host,
                        save_artifacts=save_artifacts,
                    )
                    if log_path:
                        console.print(f"Logs: {log_path}")
                        console.print(
                            f"Tail with: ssh {config.jump_host} 'tail -f {log_path}'"
                        )
                    if save_artifacts:
                        console.print(
                            f"Artifacts will be saved to: ~/.cache/idun/artifacts/{run_id}/"
                        )
                        console.print(
                            f"Download with: just idun artifacts --run-id {run_id}"
                        )
                    # Show replay command
                    replay_parts = [
                        "just",
                        "idun",
                        "submit",
                        "--script",
                        script_display,
                    ]
                    if script_args:
                        replay_parts.extend(
                            ["--args", _quote_arg(script_args, force=True)]
                        )
                    typer.echo("Run again with:")
                    typer.echo(" ".join(replay_parts))
                    return

        # New job submission - reset repo and create overlay
        if not prompt_confirm(
            f"This will reset {REMOTE_REPO_DISPLAY} to origin/main. Continue?",
            default=False,
        ):
            console.print("Submission cancelled.")
            return
        reset_remote_repo(config)
        changed, deleted = collect_changed_files(repo_root)
        bundle = create_overlay_bundle(repo_root, changed, deleted)
        if bundle:
            try:
                overlay_path = upload_overlay_bundle(config, bundle)
            finally:
                try:
                    bundle.unlink()
                except FileNotFoundError:
                    pass

    if machine is None:
        machine = prompt_choice("Machine type", ["GPU", "CPU"], default="GPU")
    else:
        machine = normalize_machine(machine)

    gpu_type_value = ""
    gpu_vram_choice = ""
    gpu_constraint_override = ""

    if machine == "GPU":
        if gpus is None:
            gpus = prompt_int("GPU count", 1)
        if gpus < 1:
            raise typer.BadParameter("GPU count must be at least 1")
        if gpu_type is None and gpu_vram is None and gpu_constraint is None:
            gpu_type_value, gpu_vram_choice, gpu_constraint_override = prompt_gpu_spec()
        else:
            if gpu_type is not None:
                gpu_type_value = normalize_gpu_type(gpu_type)
            if gpu_vram is not None:
                gpu_vram_choice = normalize_gpu_vram(gpu_vram)
            if gpu_constraint is not None:
                gpu_constraint_override = gpu_constraint.strip()
            if gpu_constraint_override and gpu_vram_choice:
                raise typer.BadParameter(
                    "Use either --gpu-vram or --gpu-constraint, not both"
                )
            if gpu_type_value:
                options = GPU_TYPE_VRAM_OPTIONS.get(gpu_type_value)
                if options and gpu_vram_choice and gpu_vram_choice not in options:
                    readable = ", ".join(
                        f"{gpu_type_value} {format_vram_label(v)}" for v in options
                    )
                    raise typer.BadParameter(
                        f"GPU VRAM not available for {gpu_type_value}. Options: {readable}"
                    )
    else:
        gpus = 0
        if gpu_type not in {None, "", "any", "Any"}:
            raise typer.BadParameter("GPU type is only valid for GPU jobs")
        if gpu_vram not in {None, "", "any", "Any"}:
            raise typer.BadParameter("GPU VRAM is only valid for GPU jobs")
        if gpu_constraint not in {None, "", "any", "Any"}:
            raise typer.BadParameter("GPU constraint is only valid for GPU jobs")

    gpu_type = gpu_type_value
    gpu_constraint_resolved = (
        gpu_constraint_override
        if gpu_constraint_override
        else GPU_VRAM_CONSTRAINTS.get(gpu_vram_choice, "")
    )

    # Exclude nodes already in use
    exclude_nodes: list[str] = []
    if exclude is not None:
        exclude_nodes = [n.strip() for n in exclude.split(",") if n.strip()]
    elif not no_exclude and sys.stdin.isatty():
        active_nodes = get_active_nodes(config)
        if active_nodes:
            node_list = ", ".join(active_nodes)
            if prompt_confirm(
                f"Exclude nodes already in use ({node_list})?",
                default=True,
            ):
                exclude_nodes = active_nodes

    partition_default = (
        config.partition_gpu if machine == "GPU" else config.partition_cpu
    )

    if partition is None:
        partition = partition_default
    else:
        partition = partition.strip()
    if account is None:
        account = config.account
    else:
        account = account.strip()

    # Handle scheduling (--begin or tomorrow working hours)
    begin_time: str | None = None
    if begin is not None:
        if begin.lower() == "tomorrow":
            tomorrow = date.today() + timedelta(days=1)
            begin_time = f"{tomorrow.isoformat()}T10:00:00"
            if time_limit is None:
                time_limit = "0-08:00:00"
        else:
            begin_time = begin.strip()
    elif sys.stdin.isatty() and time_limit is None:
        schedule_choice = prompt_choice(
            "Schedule",
            ["Run now", "Tomorrow working hours (10:00-18:00)"],
            default="Run now",
        )
        if schedule_choice.startswith("Tomorrow"):
            tomorrow = date.today() + timedelta(days=1)
            begin_time = f"{tomorrow.isoformat()}T10:00:00"
            time_limit = "0-08:00:00"

    if time_limit is None:
        time_limit = prompt_text("Time limit (D-HH:MM:SS)", config.time_limit).strip()
    else:
        time_limit = time_limit.strip()
    if cpus_per_task is None:
        cpus_per_task = prompt_int("CPUs per task", config.cpus_per_task)
    if mem is None:
        mem = prompt_text("Memory (e.g. 32G)", config.mem).strip()
    else:
        mem = mem.strip()

    job_name_default = f"{config.job_name_prefix}-{machine.lower()}"
    if job_name is None:
        job_name = prompt_text("Job name", job_name_default).strip()
    else:
        job_name = job_name.strip()

    notify_value = notify
    if notify_value is None:
        if sys.stdin.isatty():
            notify_value = prompt_confirm(
                "Send email notification when job starts and ends?", default=False
            )
        else:
            notify_value = False

    forwards: list[PortForward] = []
    tunnel_raw: list[str] = tunnel or []
    post_action_value = "background"
    log_choice_value: str | None = None
    keep_alive_value = keep_alive

    if mode == "script":
        if post_action is not None:
            post_action_value = normalize_post_action(post_action)
        elif wait or tunnel_raw:
            post_action_value = "shell"
        elif sys.stdin.isatty():
            choice = prompt_choice(
                "After submit",
                ["Run in background", "Wait and stream logs", "Wait and open shell"],
                default="Run in background",
            )
            if choice.startswith("Wait and stream"):
                post_action_value = "logs"
                log_choice_value = prompt_choice(
                    "Which log?", ["stdout", "stderr", "both"], default="both"
                )
            elif choice.startswith("Wait and open"):
                post_action_value = "shell"
            else:
                post_action_value = "background"
        else:
            post_action_value = "background"
        if keep_alive_value is None:
            if sys.stdin.isatty():
                if post_action_value == "logs":
                    keep_alive_value = prompt_confirm(
                        "Keep allocation alive after script finishes and open a shell?",
                        default=False,
                    )
                else:
                    keep_alive_value = prompt_confirm(
                        "Keep allocation alive after script finishes?",
                        default=post_action_value == "shell",
                    )
            else:
                keep_alive_value = False
    else:
        if post_action is not None:
            post_action_value = normalize_post_action(post_action)
        elif wait or tunnel_raw:
            post_action_value = "shell"
        elif sys.stdin.isatty():
            wait_shell = prompt_confirm(
                "Wait for RUNNING and open a shell?", default=bool(tunnel_raw)
            )
            post_action_value = "shell" if wait_shell else "background"
        else:
            post_action_value = "background"
        keep_alive_value = False

    if post_action_value == "logs":
        if log_choice is not None:
            log_choice_value = normalize_log_choice(log_choice)
        if log_choice_value is None:
            log_choice_value = "both"
    if log_choice_value is None:
        log_choice_value = "stdout"

    if post_action_value != "shell" and tunnel_raw:
        console.print("Port forwarding requires an interactive shell. Forcing shell.")
        post_action_value = "shell"

    needs_shell = post_action_value == "shell" or (
        post_action_value == "logs" and bool(keep_alive_value)
    )
    if needs_shell:
        if not tunnel_raw:
            forwards = _prompt_port_forwards()
        else:
            try:
                forwards = parse_port_forwards(" ".join(tunnel_raw))
            except ValueError as exc:
                raise typer.BadParameter(str(exc))

    # Prompt for artifact saving (only for script mode)
    save_artifacts_value = False
    if mode == "script" and sys.stdin.isatty():
        save_artifacts_value = prompt_confirm(
            "Save artifacts (changed/created files) after run?",
            default=True,
        )

    slurm_text = build_slurm_script(
        job_name=job_name,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        gpus=gpus,
        gpu_type=gpu_type,
        constraint=gpu_constraint_resolved,
        sleep_command=config.sleep_command,
        run_module=run_module,
        overlay_path=overlay_path,
        repo_path=REMOTE_REPO_PATH,
        keep_alive=bool(keep_alive_value),
        notify=bool(notify_value),
        user_email=config.user_email,
        script_args=script_args,
        begin_time=begin_time,
        save_artifacts=save_artifacts_value,
        exclude_nodes=exclude_nodes or None,
    )

    job_id = submit_slurm(config, slurm_text)
    console.print(f"Submitted job {job_id}")
    replay_command = _format_replay_command(
        mode=mode,
        script_display=script_display,
        script_args=script_args,
        machine=machine,
        gpus=gpus,
        gpu_type=gpu_type,
        gpu_vram=gpu_vram_choice,
        gpu_constraint_override=gpu_constraint_override,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        job_name=job_name,
        post_action=post_action_value,
        log_choice=log_choice_value,
        keep_alive=bool(keep_alive_value),
        notify=bool(notify_value),
        tunnels=tunnel_raw,
        begin_time=begin_time,
    )
    typer.echo("Run again with:")
    typer.echo(replay_command)

    if save_artifacts_value:
        console.print("Artifacts will be saved to: ~/.cache/idun/artifacts/<run_id>/")
        console.print("Download with: just idun artifacts")

    if post_action_value == "shell":
        host = wait_for_running(config, job_id)
        open_ssh_shell(config, host, forwards, job_id=job_id)
    elif post_action_value == "logs":
        host = wait_for_running(config, job_id)
        follow_job_logs(
            config, job_id, which=log_choice_value, stop_on_done=bool(keep_alive_value)
        )
        if keep_alive_value:
            open_ssh_shell(config, host, forwards, job_id=job_id)


@app.command(name="list")
def list_jobs(
    interactive: bool = typer.Option(True, "--interactive/--no-interactive"),
) -> None:
    """List current jobs."""
    config = load_config()
    jobs = fetch_jobs(config)
    if not jobs:
        console.print("No jobs found")
        return
    render_jobs(jobs)
    if interactive and sys.stdin.isatty():
        _list_actions(config, jobs)


@app.command()
def cancel(job_id: str | None = None) -> None:
    """Cancel a job."""
    config = load_config()
    if not job_id:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Job ID is required")
        job = select_job(config, message="Select a job to cancel")
        if job is None:
            return
        job_id = job.get("id")
    if not job_id:
        raise typer.BadParameter("Job ID is required")
    _ = run_ssh(config, f"scancel {shlex.quote(job_id)}")
    console.print(f"Cancelled job {job_id}")


@app.command(name="cancel-all")
def cancel_all() -> None:
    """Cancel all running jobs."""
    config = load_config()
    jobs = fetch_jobs(config)
    if not jobs:
        console.print("No jobs to cancel")
        return
    render_jobs(jobs)
    if not prompt_confirm(f"Cancel all {len(jobs)} job(s)?", default=False):
        console.print("Cancelled")
        return
    for job in jobs:
        job_id = job.get("id")
        if job_id:
            _ = run_ssh(config, f"scancel {shlex.quote(job_id)}")
            console.print(f"Cancelled job {job_id}")
    console.print(f"All {len(jobs)} job(s) cancelled")


@app.command()
def ssh(
    job_id: str | None = None,
    tunnel: list[str] = typer.Option(
        None, "--tunnel", help="Port forward (port or local:remote)"
    ),
) -> None:
    """SSH into a job's compute node."""
    config = load_config()
    if not job_id:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Job ID is required")
        job = select_job(config, message="Select a job to connect")
        if job is None:
            return
        job_id = job.get("id")
    if not job_id:
        raise typer.BadParameter("Job ID is required")
    forwards = parse_port_forwards(" ".join(tunnel or [])) if tunnel else []
    if not forwards and sys.stdin.isatty():
        forwards = _prompt_port_forwards()
    host = None
    if job_id:
        state, host = get_job_state_and_host(config, job_id)
        if state != "RUNNING" or not host:
            host = wait_for_running(config, job_id)
    if not host:
        raise typer.BadParameter("Unable to resolve job host")
    open_ssh_shell(config, host, forwards, job_id=job_id)


@app.command(name="wait")
def wait_cmd(
    job_id: str | None = None,
    poll: int = typer.Option(10, "--poll", help="Polling interval in seconds"),
    tunnel: list[str] = typer.Option(None, "--tunnel", help="Port forward"),
    shell: bool = typer.Option(
        True, "--shell/--no-shell", help="Open shell when running"
    ),
) -> None:
    """Wait for a job to start running."""
    config = load_config()
    if not job_id:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Job ID is required")
        job = select_job(config, message="Select a job to wait for")
        if job is None:
            return
        job_id = job.get("id")
    if not job_id:
        raise typer.BadParameter("Job ID is required")
    forwards = parse_port_forwards(" ".join(tunnel or [])) if tunnel else []
    if forwards and not shell:
        console.print("Port forwarding requires an interactive shell. Forcing shell.")
        shell = True
    host = wait_for_running(config, job_id, poll=poll)
    if shell:
        open_ssh_shell(config, host, forwards, shell=True, job_id=job_id)


def _select_reuse_log(config: IdunConfig) -> str | None:
    """Prompt user to select a reuse log file."""
    reuse_logs = fetch_reuse_logs(config)
    if not reuse_logs:
        console.print("No reuse logs found")
        return None
    choices = []
    for log in reuse_logs:
        # Include duration if available
        duration = log.get("duration", "")
        if duration:
            name = f"{log['filename']} | {log['time']} | {duration}"
        else:
            name = f"{log['filename']} | {log['time']}"
        choices.append({"name": name, "value": log["path"]})
    choices.append({"name": "Exit", "value": None})
    return inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message="Select a log file",
        choices=choices,
    ).execute()


def _select_slurm_log(config: IdunConfig) -> str | None:
    """Prompt user to select a slurm log file."""
    slurm_logs = fetch_slurm_logs(config)
    if not slurm_logs:
        console.print("No slurm log files found")
        return None
    choices = []
    for log in slurm_logs:
        # Include duration if available
        duration = log.get("duration", "")
        if duration:
            name = f"{log['filename']} | {log['time']} | {duration}"
        else:
            name = f"{log['filename']} | {log['time']}"
        choices.append({"name": name, "value": log["path"]})
    choices.append({"name": "Exit", "value": None})
    return inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message="Select a log file",
        choices=choices,
    ).execute()


def _prompt_log_mode() -> str:
    """Prompt user for log view mode."""
    choice = prompt_choice(
        "View mode",
        ["View full log (scrollable, press q to exit)", "Follow log (live updates)"],
        default="View full log (scrollable, press q to exit)",
    )
    return "view" if choice.startswith("View") else "follow"


@app.command()
def logs(
    job_id: str | None = None,
    which: str = typer.Option("stdout", "--which", help="stdout, stderr, or both"),
    reuse: bool = typer.Option(
        False, "--reuse", help="View reuse logs instead of job logs"
    ),
) -> None:
    """Tail job logs."""
    config = load_config()

    # If --reuse flag or interactive selection
    if reuse or (not job_id and sys.stdin.isatty()):
        if not job_id and sys.stdin.isatty() and not reuse:
            # Check if there are reuse logs and offer choice
            reuse_logs = fetch_reuse_logs(config, limit=5)
            if reuse_logs:
                log_type = prompt_choice(
                    "Log type",
                    ["Slurm job logs", "Reuse logs (background runs)"],
                    default="Slurm job logs",
                )
                if log_type.startswith("Reuse"):
                    reuse = True
                else:
                    # Show slurm log files directly
                    log_path = _select_slurm_log(config)
                    if log_path:
                        mode = _prompt_log_mode()
                        tail_slurm_log(config, log_path, mode=mode)
                    return

        if reuse:
            log_path = _select_reuse_log(config)
            if log_path:
                mode = _prompt_log_mode()
                tail_reuse_log(config, log_path, mode=mode)
            return

    # Direct job_id provided via CLI
    if not job_id:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Job ID is required")
        # Fallback: show slurm log files
        log_path = _select_slurm_log(config)
        if log_path:
            mode = _prompt_log_mode()
            tail_slurm_log(config, log_path, mode=mode)
        return
    # Job ID provided - use the original flow
    which = which.lower()
    state, _ = get_job_state_and_host(config, job_id)
    if state in QUEUED_STATES:
        _ = wait_for_running(config, job_id)
    follow_job_logs(config, job_id, which=which)


@app.command()
def script(job_id: str | None = None) -> None:
    """Show the batch script for a job."""
    config = load_config()
    if not job_id:
        if not sys.stdin.isatty():
            raise typer.BadParameter("Job ID is required")
        job = select_job(config, message="Select a job to view script")
        if job is None:
            return
        job_id = job.get("id")
    if not job_id:
        raise typer.BadParameter("Job ID is required")
    content = get_batch_script(config, job_id)
    if content is None:
        console.print(f"Could not retrieve batch script for job {job_id}")
        console.print("Note: Scripts are only available for pending/running jobs")
        raise typer.Exit(1)
    console.print(
        Panel(content, title=f"Batch Script for Job {job_id}", border_style="blue")
    )


@app.command()
def monitor(
    watch: int = typer.Option(
        2, "--watch", "-w", help="Refresh interval (0 for single run)"
    ),
) -> None:
    """Monitor CPU/GPU resources for running jobs."""
    config = load_config()
    raise typer.Exit(show_monitor(config, watch=watch))


@app.command()
def artifacts(
    run_id: str | None = typer.Option(None, "--run-id", help="Run ID to download"),
    output: Path = typer.Option(
        Path("./artifacts"), "--output", "-o", help="Local output directory"
    ),
    list_only: bool = typer.Option(
        False, "--list", "-l", help="List available artifacts without downloading"
    ),
) -> None:
    """Download artifacts from completed runs."""
    config = load_config()
    artifact_base = "$HOME/.cache/idun/artifacts"

    if list_only or run_id is None:
        # List available artifacts
        result = run_ssh(
            config,
            f"ls -lt {artifact_base} 2>/dev/null | head -20",
            check=False,
        )
        output_text = (result.stdout or "").strip()
        if not output_text or "No such file" in (result.stderr or ""):
            console.print("No artifacts found")
            return

        console.print("Available artifacts (most recent first):")
        console.print(output_text)

        if list_only:
            return

        # Prompt to select one
        result = run_ssh(
            config,
            f"ls -t {artifact_base} 2>/dev/null",
            check=False,
        )
        dirs = [d for d in (result.stdout or "").strip().split("\n") if d]
        if not dirs:
            console.print("No artifacts found")
            return

        choices = [{"name": d, "value": d} for d in dirs[:20]]
        choices.append({"name": "Exit", "value": None})
        run_id = inquirer.select(
            message="Select artifact to download",
            choices=choices,
        ).execute()

        if run_id is None:
            return

    # Download artifacts
    remote_path = f"~/.cache/idun/artifacts/{run_id}"

    # Check if artifacts exist and get size
    check_result = run_ssh(
        config,
        f"test -d {remote_path} && du -sh {remote_path} && find {remote_path} -type f | wc -l",
        check=False,
    )
    if check_result.returncode != 0:
        console.print(f"No artifacts found for run {run_id}")
        return

    # Parse size info
    output_lines = (check_result.stdout or "").strip().split("\n")
    if len(output_lines) >= 2:
        size_info = output_lines[0].split()[0] if output_lines[0] else "unknown"
        file_count = output_lines[1].strip() if len(output_lines) > 1 else "0"
        console.print(f"Remote artifacts: {size_info}, {file_count} file(s)")

        if file_count == "0":
            console.print(
                "No files in artifacts (script may have crashed before creating output)"
            )
            return

    # Create local output directory (resolve to absolute path)
    local_dir = (output / run_id).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download using rsync
    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )

    console.print(f"Downloading artifacts from {remote_path}...")
    import subprocess

    rsync_result = subprocess.run(
        [
            "rsync",
            "-avz",
            "--progress",
            "-e",
            "ssh",
            f"{jump_target}:{remote_path}/",
            str(local_dir) + "/",
        ],
        capture_output=False,
    )

    if rsync_result.returncode == 0:
        console.print(f"Artifacts downloaded to: {local_dir}")
    else:
        console.print("Failed to download artifacts")
        raise typer.Exit(1)


# ============================================================================
# vLLM Commands
# ============================================================================


def _vllm_download(model: str | None) -> None:
    """Download a HuggingFace model to the IDUN cache."""
    config = load_config()

    if model is None:
        model_name, _, _, _ = prompt_model()
    else:
        model_name = model

    console.print(f"\n[bold]Downloading [cyan]{model_name}[/cyan]...[/bold]")
    console.print("[dim]Press Ctrl-C to cancel[/dim]\n")
    download_model(config, model_name)
    console.print(f"\n[green]Done.[/green] {model_name}")


def _vllm_delete() -> None:
    """Interactively delete cached HuggingFace models."""
    from InquirerPy import inquirer

    config = load_config()
    models = list_cached_models(config)

    if not models:
        console.print("No cached models found in ~/.cache/huggingface/hub/")
        return

    # Build select choices: individual models + "Delete all"
    choices: list[dict[str, str]] = []
    for m in models:
        choices.append({"name": f"{m.display_name} ({m.size})", "value": m.folder})
    if len(models) > 1:
        choices.append(
            {"name": f"Delete all ({len(models)} models)", "value": "__all__"}
        )

    selected = inquirer.select(
        message="Select model to delete",
        choices=choices,
    ).execute()

    if selected == "__all__":
        folders = [m.folder for m in models]
        display = "all cached models"
    else:
        folders = [selected]
        display = next(m.display_name for m in models if m.folder == selected)

    if not prompt_confirm(f"Delete {display}?", default=False):
        return

    delete_cached_model_folders(config, folders)
    console.print(f"[green]Deleted.[/green]")


def _run_vllm_menu() -> bool:
    """Run the vLLM interactive sub-menu.

    Returns True if user selected 'back' (to return to main menu).
    """
    vllm_choice = inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
        message="vLLM action",
        choices=[
            {"name": "serve - start vLLM server", "value": "serve"},
            {"name": "download - pre-download a model", "value": "download"},
            {"name": "delete - manage cached models", "value": "delete"},
            {"name": "connect - port-forward to a server", "value": "connect"},
            {"name": "status - show running servers", "value": "status"},
            {"name": "logs - view server logs", "value": "logs"},
            {"name": "stop - stop a server", "value": "stop"},
            {"name": "setup - one-time installation", "value": "setup"},
            {"name": "back", "value": None},
        ],
        default="serve",
    ).execute()
    if vllm_choice is None:
        return True  # Go back to main menu
    if vllm_choice == "serve":
        _vllm_serve(
            model=None,
            job_id=None,
            port=DEFAULT_PORT,
            gpus=None,
            gpu_type=None,
            extra_args=None,
        )
    elif vllm_choice == "download":
        _vllm_download(model=None)
    elif vllm_choice == "delete":
        _vllm_delete()
    elif vllm_choice == "connect":
        _vllm_connect(job_id=None, local_port=None)
    elif vllm_choice == "status":
        _vllm_status()
    elif vllm_choice == "logs":
        _vllm_logs(job_id=None, follow=True)
    elif vllm_choice == "stop":
        _vllm_stop(job_id=None)
    elif vllm_choice == "setup":
        _vllm_setup(backend_arg=None)
    return False


def _vllm_setup(backend_arg: str | None) -> None:
    """One-time setup: create venv and install vLLM or SGLang on a compute node."""
    config = load_config()

    # Interactive backend selection if not specified
    if backend_arg is None and sys.stdin.isatty():
        backend_choice = prompt_choice(
            "Which backend to set up?",
            ["vLLM (stable, mature APIs)", "SGLang (faster multi-turn)", "Both"],
            default="vLLM (stable, mature APIs)",
        )
        if "Both" in backend_choice:
            backend_arg = "both"
        elif "SGLang" in backend_choice:
            backend_arg = "sglang"
        else:
            backend_arg = "vllm"

    backends_to_setup = (
        ["vllm", "sglang"] if backend_arg == "both" else [backend_arg or "vllm"]
    )

    for backend in backends_to_setup:
        console.print(f"\n[bold]Setting up {backend}...[/bold]")
        _setup_backend(config, backend)


def _run_setup_on_login_node(config: "IdunConfig", setup_script: str) -> int:
    """Run setup on the login node (which has internet access).

    Compute nodes have no internet, so pip install must run on the login node.
    The venvs live on shared storage so they're accessible from compute nodes too.
    """
    import subprocess

    target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )

    result = subprocess.run(
        ["ssh", "-tt", target, f"bash -lc {shlex.quote(setup_script)}"]
    )
    return result.returncode


def _setup_backend(config: "IdunConfig", backend: str) -> None:
    """Set up a specific backend (vllm or sglang) on the login node."""
    venv_dir = get_backend_venv_dir(backend)

    # Check for an existing backend venv and offer a clean recreate.
    if check_vllm_setup(config, backend):
        console.print(f"Found existing {backend} venv at {venv_dir}")
        if not prompt_confirm(
            f"Delete the old venv at {venv_dir} and recreate it?", default=False
        ):
            return
        # Delete venv on login node (shared storage)
        import subprocess

        target = (
            f"{config.ssh_user}@{config.jump_host}"
            if config.ssh_user
            else config.jump_host
        )
        subprocess.run(["ssh", target, f"rm -rf {venv_dir}"])

    # Run setup on the login node (has internet access)
    console.print(f"Running {backend} setup on login node ({config.jump_host})...")
    setup_script = build_setup_commands(backend)
    result_code = _run_setup_on_login_node(config, setup_script)

    if result_code == 0:
        console.print(f"{backend} setup complete!")
    else:
        console.print("Setup may have failed. Check the output above.")
        raise typer.Exit(1)


def _vllm_serve(
    model: str | None,
    job_id: str | None,
    port: int,
    gpus: int | None,
    gpu_type: str | None,
    extra_args: str | None,
    sglang: bool = False,
    backend_arg: str | None = None,
) -> None:
    """Start vLLM or SGLang server on a compute node."""
    config = load_config()

    # Interactive model selection if not provided
    saved_extra_args: str | None = None
    if model is None:
        model, recommended_gpus, recommended_gpu_type, saved_extra_args = prompt_model()
        if gpus is None:
            gpus = recommended_gpus
        if gpu_type is None and recommended_gpu_type:
            gpu_type = recommended_gpu_type

    if backend_arg not in (None, "vllm", "sglang"):
        console.print("Serve backend must be 'vllm' or 'sglang'")
        raise typer.Exit(1)

    # Resolve backend for serve.
    # Explicit --backend wins, then legacy --sglang, then interactive prompt, else vLLM.
    backend = backend_arg or ("sglang" if sglang else "vllm")
    if backend_arg is None and sys.stdin.isatty() and not sglang:
        backend_choice = prompt_choice(
            "Backend",
            [
                "SGLang (faster multi-turn, better KV cache)",
                "vLLM (stable, mature APIs)",
            ],
            default="vLLM (stable, mature APIs)",
        )
        backend = "sglang" if "SGLang" in backend_choice else "vllm"

    # Prompt for extra args if interactive and not provided
    # Use saved args as default if available
    if extra_args is None and sys.stdin.isatty():
        default_args = saved_extra_args or ""
        prompt_label = f"Extra {backend} args (e.g. --max-model-len 4096)"
        extra_args = (
            prompt_text(
                prompt_label,
                default_args,
            ).strip()
            or None
        )

    # Default GPU settings
    if gpus is None:
        gpus = 1

    # Find or submit a job
    host: str | None = None
    if job_id is not None:
        # Use specified job
        state, host = get_job_state_and_host(config, job_id)
        if state != "RUNNING":
            console.print(f"Job {job_id} is not running (state: {state})")
            console.print("Waiting for job to start...")
            host = wait_for_running(config, job_id)
        # Always detect GPU count from existing job (override model preset)
        from idun.slurm import get_job_gpu_count

        gpus = get_job_gpu_count(config, job_id)
        console.print(f"Detected {gpus} GPU(s) allocated to job {job_id}")
    else:
        # Check for running jobs
        running_jobs = [j for j in fetch_jobs(config) if j.get("state") == "RUNNING"]
        if running_jobs:
            if prompt_confirm("Use an existing running job?", default=True):
                job = choose_job(config, running_jobs, message="Select job for vLLM")
                if job is not None:
                    job_id = job.get("id")
                    host = job.get("node")
                    # Always detect GPU count from selected job (override model preset)
                    if job_id:
                        from idun.slurm import get_job_gpu_count

                        gpus = get_job_gpu_count(config, job_id)
                        console.print(
                            f"Detected {gpus} GPU(s) allocated to job {job_id}"
                        )

        if job_id is None:
            # Submit new job with interactive prompts (like submit command)

            # Interactive GPU spec selection
            gpu_type_value, gpu_vram_choice, gpu_constraint_override = prompt_gpu_spec()
            if gpus is None:
                gpus = prompt_int("GPU count", 1)

            # Build constraint
            constraint = (
                gpu_constraint_override
                if gpu_constraint_override
                else GPU_VRAM_CONSTRAINTS.get(gpu_vram_choice, "")
            )

            # Interactive resource selection
            time_limit = prompt_text("Time limit (D-HH:MM:SS)", "0-12:00:00").strip()
            cpus_per_task = prompt_int("CPUs per task", config.cpus_per_task)
            mem = prompt_text("Memory (e.g. 64G)", "64G").strip()
            job_name = prompt_text("Job name", "vllm").strip()

            # Post-submit action choice
            post_action = prompt_choice(
                "After submit",
                ["Run in background", "Wait and stream logs", "Wait and open shell"],
                default="Run in background",
            )

            # Build SLURM script with server startup
            slurm_text = build_vllm_slurm_script(
                job_name=job_name,
                partition=config.partition_gpu,
                account=config.account,
                time_limit=time_limit,
                cpus_per_task=cpus_per_task,
                mem=mem,
                gpus=gpus,
                gpu_type=gpu_type_value or gpu_type or "",
                constraint=constraint,
                sleep_command=config.sleep_command,
                model=model,
                port=port,
                extra_args=extra_args,
                backend=backend,
            )
            job_id = submit_slurm(config, slurm_text)
            console.print(f"Submitted job {job_id}")
            console.print(f"{backend} will start automatically with model: {model}")
            console.print(f"Port: {port}")

            # Save custom model with working args for next time
            if extra_args and model:
                from idun.vllm import _save_custom_model

                _save_custom_model(model, gpus, gpu_type or "", extra_args)

            if post_action.startswith("Run in background"):
                console.print()
                console.print("Check status with:")
                console.print("  just idun vllm status")
                console.print("Check logs with:")
                console.print(f"  just idun vllm logs --job-id {job_id}")
                return

            console.print()
            console.print("Waiting for job to start...")
            host = wait_for_running(config, job_id)

            # Job started - server is starting automatically
            console.print()
            console.print(f"Job running on {host}")
            console.print(
                f"[green]✓[/green] {backend} server starting at http://localhost:{port}/v1"
            )
            console.print()
            console.print(
                f"Connect with:  [cyan]just idun vllm connect --job-id {job_id}[/cyan]"
            )
            console.print(
                f"Check logs:    [cyan]just idun vllm logs --job-id {job_id}[/cyan]"
            )
            console.print("Check status:  [cyan]just idun vllm status[/cyan]")
            console.print()

            if post_action.startswith("Wait and stream"):
                console.print()
                console.print("Streaming vLLM logs (Ctrl+C to stop)...")
                _vllm_logs(job_id, follow=True)
            elif post_action.startswith("Wait and open"):
                console.print()
                open_ssh_shell(config, host, job_id=job_id)
            return

    if not job_id or not host:
        console.print("No job available")
        return

    # Using existing job - need to start the backend manually.
    if not check_vllm_setup_on_node(config, host, backend):
        console.print(f"{backend} is not set up on this node.")
        if prompt_confirm("Run setup now?"):
            setup_script = build_setup_commands(backend)
            if _run_setup_on_node(config, host, setup_script) != 0:
                console.print("Setup failed")
                raise typer.Exit(1)
        else:
            console.print(f"Run 'just idun vllm setup --backend {backend}' first")
            return

    # Start the server on existing job
    console.print(f"Starting {backend} server with {model} on {host}...")
    if backend == "vllm" and extra_args:
        console.print(
            f"Using {gpus} GPU(s) with explicit vLLM override args (auto parallelism still applies unless overridden)"
        )
    else:
        console.print(f"Using {gpus} GPU(s) with tensor-parallel-size={gpus}")
    success, message = start_vllm_server(
        config, host, job_id, model, port, gpus, extra_args, backend
    )

    if success:
        console.print(message)
        console.print()
        console.print(f"[green]✓[/green] Server starting at http://localhost:{port}/v1")
        console.print()
        console.print("Connect with:  [cyan]just idun vllm connect[/cyan]")
        console.print(
            f"Check logs:    [cyan]just idun vllm logs --job-id {job_id}[/cyan]"
        )
        console.print("Check status:  [cyan]just idun vllm status[/cyan]")

        # Save custom model with working args for next time
        if extra_args and model:
            from idun.vllm import _save_custom_model

            _save_custom_model(model, gpus, gpu_type or "", extra_args)
    else:
        console.print(f"Failed to start server: {message}")
        raise typer.Exit(1)


def _vllm_stop(job_id: str | None, all_servers: bool = False) -> None:
    """Stop a running vLLM server."""
    from InquirerPy import inquirer

    config = load_config()

    servers = get_all_servers(config)
    if not servers:
        console.print("No vLLM servers tracked")
        return

    to_stop: list[VllmServer] = []

    if all_servers:
        to_stop = list(servers.values())
    elif job_id is not None:
        server = servers.get(job_id)
        if not server:
            console.print(f"No server found for job {job_id}")
            return
        to_stop = [server]
    elif len(servers) == 1:
        # Single server — use simple select
        server = prompt_server(config, servers)
        if server is None:
            return
        to_stop = [server]
    else:
        # Multiple servers — checkbox with "Stop all"
        choices = [{"name": f"Stop all ({len(servers)} servers)", "value": "__all__"}]
        for jid, srv in servers.items():
            label = f"{jid} | {srv.model} | {srv.host}:{srv.port}"
            choices.append({"name": label, "value": jid})

        selected = inquirer.checkbox(
            message="Select servers to stop",
            choices=choices,
        ).execute()

        if not selected:
            console.print("Nothing selected")
            return

        if "__all__" in selected:
            to_stop = list(servers.values())
        else:
            to_stop = [servers[jid] for jid in selected]

    count = len(to_stop)
    if count == 1:
        prompt_msg = f"Stop vLLM server on job {to_stop[0].job_id}?"
    else:
        prompt_msg = f"Stop {count} vLLM servers?"

    if not prompt_confirm(prompt_msg, default=True):
        return

    failed = 0
    for server in to_stop:
        console.print(f"Stopping vLLM server on {server.host} (job {server.job_id})...")
        success, message = stop_vllm_server(config, server.host, server.job_id)
        if success:
            console.print(message)
        else:
            console.print(f"Failed to stop server {server.job_id}: {message}")
            failed += 1

    if count > 1:
        stopped = count - failed
        console.print(f"Stopped {stopped}/{count} servers")


def _vllm_status() -> None:
    """Show status of all vLLM servers."""
    config = load_config()

    servers = get_all_servers(config)
    if not servers:
        console.print("No vLLM servers tracked")
        return

    console.print("Checking server health...")
    health = check_server_health_bulk(config, servers)
    render_servers(servers, health)


def _vllm_inspect(job_id: str | None, json_output: bool) -> None:
    """Inspect a tracked vLLM/SGLang server for a specific job."""
    if not job_id:
        console.print("--job-id is required for inspect")
        raise typer.Exit(1)

    config = load_config()
    info = inspect_server(config, job_id)

    if json_output:
        print(json.dumps(info))
        return

    if not info.get("found"):
        console.print(f"No server found for job {job_id}")
        return

    console.print(f"Job ID:  {info['job_id']}")
    console.print(f"Backend: {info['backend']}")
    console.print(f"Model:   {info['model']}")
    console.print(f"Host:    {info['host']}")
    console.print(f"Port:    {info['port']}")
    console.print(f"Status:  {info['status']}")


def _vllm_logs(job_id: str | None, follow: bool) -> None:
    """View vLLM server logs."""
    config = load_config()

    servers = get_all_servers(config)

    if job_id is None and servers:
        # Interactive selection
        server = prompt_server(config, servers)
        if server is None:
            return
        job_id = server.job_id

    if not job_id:
        console.print("No job ID specified and no servers tracked")
        return

    log_file = f"{VLLM_LOG_DIR}/{job_id}_serve.log"

    # Get the host for this job
    server = servers.get(job_id)
    if server:
        host = server.host
    else:
        # Try to get from SLURM
        state, host = get_job_state_and_host(config, job_id)
        if not host:
            console.print(f"Cannot find host for job {job_id}")
            return

    from idun.ssh import run_ssh_stream

    console.print(f"Viewing logs from {host}:{log_file}")
    try:
        # less +F: opens full file scrollable, starts in follow mode (live tail)
        # Ctrl-C to stop following and scroll, Shift-F to resume, q to quit
        run_ssh_stream(config, f"less +F {log_file}", check=False, tty=True)
    except KeyboardInterrupt:
        pass


def _vllm_connect(job_id: str | None, local_port: int | None) -> None:
    """Set up SSH port-forward tunnel to a vLLM server."""
    import subprocess

    config = load_config()

    servers = get_all_servers(config)

    server = None
    if job_id and job_id in servers:
        server = servers[job_id]
    elif servers:
        server = prompt_server(config, servers)
    if server is None:
        console.print("No server selected")
        return

    if local_port is None:
        local_port_str = inquirer.text(  # pyright: ignore[reportPrivateImportUsage]
            message="Local port to forward to",
            default="8000",
            validate=lambda x: x.isdigit() and 1024 <= int(x) <= 65535,
        ).execute()
        local_port = int(local_port_str)

    host = server.host
    remote_port = server.port

    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )

    console.print(f"Forwarding localhost:{local_port} → {host}:{remote_port}")
    console.print(f"Model: {server.model}")
    console.print(f"Test with: curl http://localhost:{local_port}/v1/models")
    console.print("Press Ctrl+C to stop")

    # Build compute node target
    compute_target = f"{config.ssh_user}@{host}" if config.ssh_user else host

    try:
        # Use -J flag to jump through login node to compute node
        # Use 127.0.0.1 explicitly to avoid IPv4/IPv6 resolution issues
        subprocess.run(
            [
                "ssh",
                "-4",  # Force IPv4
                "-N",  # No remote command
                "-T",  # Disable pseudo-terminal
                "-J",
                jump_target,
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                f"{local_port}:127.0.0.1:{remote_port}",
                compute_target,
            ],
            stdin=subprocess.DEVNULL,
        )
    except KeyboardInterrupt:
        console.print("\nTunnel closed")


@app.command(name="vllm")
def vllm(
    action: str | None = typer.Argument(
        None,
        help="Action: serve, download, delete, connect, status, inspect, stop, logs, setup",
    ),
    model: str | None = typer.Argument(None, help="Model for serve action"),
    job_id: str | None = typer.Option(None, "--job-id", help="Target job ID"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Server port (for serve)"),
    local_port: int | None = typer.Option(
        None, "--local-port", help="Local port (for connect, default: prompt or 8000)"
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs"),
    gpus: int | None = typer.Option(None, "--gpus", help="Number of GPUs for serve"),
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="GPU type (a100, h100, etc)"
    ),
    extra_args: str | None = typer.Option(
        None, "--extra", help="Extra vLLM arguments for serve"
    ),
    sglang: bool = typer.Option(False, "--sglang", help="Use SGLang instead of vLLM"),
    backend: str | None = typer.Option(
        None, "--backend", help="Backend for setup (vllm, sglang, both)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON (for inspect action)"
    ),
    all_servers: bool = typer.Option(
        False, "--all", help="Stop all tracked servers (for stop action)"
    ),
) -> None:
    """Manage vLLM and SGLang inference servers."""
    if action is None:
        if not sys.stdin.isatty():
            # Non-interactive: show help
            raise typer.Exit()
        # Show interactive menu
        _run_vllm_menu()
        return

    if action == "serve":
        _vllm_serve(model, job_id, port, gpus, gpu_type, extra_args, sglang, backend)
    elif action == "download":
        _vllm_download(model)
    elif action == "delete":
        _vllm_delete()
    elif action == "connect":
        _vllm_connect(job_id, local_port)
    elif action == "status":
        _vllm_status()
    elif action == "inspect":
        _vllm_inspect(job_id, json_output)
    elif action == "stop":
        _vllm_stop(job_id, all_servers=all_servers)
    elif action == "logs":
        _vllm_logs(job_id, follow)
    elif action == "setup":
        _vllm_setup(backend)
    else:
        console.print(f"Unknown action: {action}")
        console.print(
            "Valid actions: serve, download, delete, connect, status, inspect, stop, logs, setup"
        )
        raise typer.Exit(1)


def _run_main_menu_action(choice: str) -> bool:
    """Run the selected main menu action.

    Returns True if should return to main menu, False to exit.
    """
    if choice == "submit":
        submit(
            script=None,
            script_args=None,
            shell=False,
            wait=False,
            machine=None,
            gpus=None,
            gpu_type=None,
            gpu_vram=None,
            gpu_constraint=None,
            partition=None,
            account=None,
            time_limit=None,
            cpus_per_task=None,
            mem=None,
            job_name=None,
            tunnel=[],
            post_action=None,
            log_choice=None,
            keep_alive=None,
            notify=None,
            begin=None,
            exclude=None,
            no_exclude=False,
        )
        return False
    if choice == "list":
        list_jobs(interactive=True)
        return False
    if choice == "monitor":
        monitor(watch=2)
        return False
    if choice == "artifacts":
        artifacts(run_id=None, output=Path("./artifacts"), list_only=False)
        return False
    if choice == "cancel":
        config = load_config()
        job = select_job(config, message="Select a job to cancel")
        if job is None:
            return False
        job_id = job.get("id")
        if not job_id:
            console.print("Job ID is required")
            return False
        if prompt_confirm(f"Cancel job {job_id}?", default=False):
            cancel(job_id)
        return False
    if choice == "logs":
        config = load_config()
        # Check if there are reuse logs and offer choice
        reuse_logs = fetch_reuse_logs(config, limit=5)
        view_reuse = False
        if reuse_logs:
            log_type = prompt_choice(
                "Log type",
                ["Slurm job logs", "Reuse logs (background runs)"],
                default="Slurm job logs",
            )
            if log_type.startswith("Reuse"):
                view_reuse = True

        if view_reuse:
            log_path = _select_reuse_log(config)
            if log_path:
                mode = _prompt_log_mode()
                tail_reuse_log(config, log_path, mode=mode)
            return False

        # Show slurm log files directly
        log_path = _select_slurm_log(config)
        if log_path:
            mode = _prompt_log_mode()
            tail_slurm_log(config, log_path, mode=mode)
        return False
    if choice == "script":
        config = load_config()
        job = select_job(config, message="Select a job to view script")
        if job is None:
            return False
        job_id = job.get("id")
        if not job_id:
            console.print("Job ID is required")
            return False
        script(job_id)
        return False
    if choice == "ssh":
        config = load_config()
        job = select_job(config, message="Select a job to connect")
        if job is None:
            return False
        job_id = job.get("id")
        if not job_id:
            console.print("Job ID is required")
            return False
        ssh(job_id, tunnel=[])
        return False
    if choice == "wait":
        config = load_config()
        job = select_job(config, message="Select a job to wait for")
        if job is None:
            return False
        job_id = job.get("id")
        if not job_id:
            console.print("Job ID is required")
            return False
        poll = prompt_int("Polling interval (seconds)", 10)
        tunnel_list: list[str] = []
        if prompt_confirm("Set up port forwarding?", default=False):
            entry = prompt_text("Ports (e.g. 8888,6006 or 8888:6006)", "").strip()
            if entry:
                tunnel_list = [entry]
        wait_cmd(job_id, poll=poll, tunnel=tunnel_list, shell=True)
        return False
    if choice == "cancel-all":
        cancel_all()
        return False
    if choice == "vllm":
        # Show vLLM sub-menu; returns True if user selected "back"
        return _run_vllm_menu()
    return False


@app.callback(invoke_without_command=True)
def cli(ctx: typer.Context) -> None:
    """IDUN cluster management CLI."""
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        typer.echo(ctx.get_help())
        raise typer.Exit()

    while True:
        choice = inquirer.select(  # pyright: ignore[reportPrivateImportUsage]
            message="Select command",
            choices=[
                {"name": "submit - interactive job wizard", "value": "submit"},
                {"name": "list - list jobs", "value": "list"},
                {"name": "vllm - manage vLLM servers", "value": "vllm"},
                {"name": "monitor - CPU/GPU resource monitor", "value": "monitor"},
                {"name": "artifacts - download run artifacts", "value": "artifacts"},
                {"name": "wait - wait and attach", "value": "wait"},
                {"name": "ssh - open shell", "value": "ssh"},
                {"name": "logs - tail stdout/stderr", "value": "logs"},
                {"name": "script - show batch script", "value": "script"},
                {"name": "cancel - cancel a job", "value": "cancel"},
                {"name": "cancel-all - cancel all jobs", "value": "cancel-all"},
                {"name": "exit", "value": None},
            ],
            default="submit",
        ).execute()

        if choice is None:
            raise typer.Exit()

        # Run the action; if it returns True, loop back to menu
        if not _run_main_menu_action(choice):
            break


def main() -> None:
    app()


if __name__ == "__main__":
    main()
