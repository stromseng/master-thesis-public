"""vLLM server management for IDUN cluster."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from idun.config import IdunConfig
from idun.ssh import run_ssh, run_ssh_stream, run_ssh_via_jump

console = Console()

# Installation directories for inference backends (separate from thesis project)
VLLM_DIR = "$HOME/repos/vllm"
SGLANG_DIR = "$HOME/repos/sglang"

# State file tracking running servers (keyed by job_id)
VLLM_STATE_FILE = "$HOME/.cache/idun/vllm/servers.json"

# Log and PID file locations
VLLM_LOG_DIR = "$HOME/.cache/idun/vllm"

# Default vLLM serve port
DEFAULT_PORT = 8001

# Model presets with VRAM estimates and GPU recommendations
# vram_gb is approximate VRAM needed for BF16 inference
MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "vram_gb": 4,
        "gpus": 1,
        "gpu_type": "",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "vram_gb": 16,
        "gpus": 1,
        "gpu_type": "a100",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "vram_gb": 18,
        "gpus": 1,
        "gpu_type": "a100",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "vram_gb": 70,
        "gpus": 1,
        "gpu_type": "a100",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "vram_gb": 70,
        "gpus": 1,
        "gpu_type": "a100",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "vram_gb": 150,
        "gpus": 2,
        "gpu_type": "a100",
    },
}


@dataclass
class VllmServer:
    """Represents a running vLLM or SGLang server."""

    job_id: str
    model: str
    port: int
    host: str
    pid: int
    started_at: str
    backend: str = "vllm"  # "vllm" or "sglang"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "model": self.model,
            "port": self.port,
            "host": self.host,
            "pid": self.pid,
            "started_at": self.started_at,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, job_id: str, data: dict[str, str | int]) -> "VllmServer":
        return cls(
            job_id=job_id,
            model=str(data.get("model", "")),
            port=int(data.get("port", DEFAULT_PORT)),
            host=str(data.get("host", "")),
            pid=int(data.get("pid", 0)),
            started_at=str(data.get("started_at", "")),
            backend=str(
                data.get("backend", "vllm")
            ),  # Default to "vllm" for backward compatibility
        )


def get_backend_dir(backend: str) -> str:
    """Return the installation directory for a backend."""
    return SGLANG_DIR if backend == "sglang" else VLLM_DIR


def get_backend_venv_dir(backend: str) -> str:
    """Return the venv directory for a backend."""
    return f"{get_backend_dir(backend)}/.venv"


def get_backend_python(backend: str) -> str:
    """Return the backend Python executable path."""
    return f"{get_backend_venv_dir(backend)}/bin/python"


def get_backend_import_name(backend: str) -> str:
    """Return the import name used to verify a backend install."""
    return "sglang" if backend == "sglang" else "vllm"


def check_backend_venv_on_node(config: IdunConfig, host: str, backend: str) -> bool:
    """Check if a backend venv exists on a compute node."""
    venv_dir = get_backend_venv_dir(backend)
    result = run_ssh_via_jump(config, host, f"test -d {venv_dir}")
    return result.returncode == 0


def delete_backend_venv_on_node(config: IdunConfig, host: str, backend: str) -> bool:
    """Delete a backend venv on a compute node."""
    venv_dir = get_backend_venv_dir(backend)
    result = run_ssh_via_jump(config, host, f"rm -rf {venv_dir}")
    return result.returncode == 0


def check_vllm_setup(config: IdunConfig, backend: str = "vllm") -> bool:
    """Check if a backend is set up on the jump host."""
    venv_dir = get_backend_venv_dir(backend)
    import_name = get_backend_import_name(backend)
    result = run_ssh(
        config,
        f"test -d {venv_dir} && {get_backend_python(backend)} -c 'import {import_name}' 2>/dev/null",
        check=False,
    )
    return result.returncode == 0


def check_vllm_setup_on_node(
    config: IdunConfig, host: str, backend: str = "vllm"
) -> bool:
    """Check if a backend is set up on a compute node."""
    venv_dir = get_backend_venv_dir(backend)
    import_name = get_backend_import_name(backend)
    result = run_ssh_via_jump(
        config,
        host,
        f"test -d {venv_dir} && {get_backend_python(backend)} -c 'import {import_name}' 2>/dev/null",
    )
    return result.returncode == 0


def build_setup_commands(backend: str = "vllm") -> str:
    """Build shell commands to set up vLLM or SGLang environment."""
    backend_dir = get_backend_dir(backend)
    backend_venv_dir = get_backend_venv_dir(backend)
    if backend == "sglang":
        # GCC/13.3.0 — nvcc 12.6 rejects GCC 14+ for FlashInfer/DeepGEMM JIT.
        return f"""
set -e
mkdir -p {backend_dir}
cd {backend_dir}
module load GCC/13.3.0
module load CUDA/12.9.1
if [ ! -d .venv ]; then
    echo "Creating venv..."
    uv venv --python 3.12 --seed
fi
source {backend_venv_dir}/bin/activate
echo "Installing SGLang from source (latest main branch)..."
uv pip install --torch-backend=cu126 'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python'
echo "Upgrading transformers, huggingface_hub, and mistral-common..."
uv pip install --upgrade 'transformers>=5.5.0' huggingface_hub mistral-common
echo "SGLang setup complete!"
python -c "import sglang; print(f'sglang {{sglang.__version__}}')"
python -c "import transformers; print(f'transformers {{transformers.__version__}}')"
echo "Verifying sgl_kernel..."
python -c "import sgl_kernel; print('sgl_kernel OK')"
"""
    else:
        return f"""
set -e
mkdir -p {backend_dir}
cd {backend_dir}
module load GCC/13.3.0
module load CUDA/12.9.1
if [ ! -d .venv ]; then
    echo "Creating venv..."
    uv venv --python 3.12 --seed
fi
source {backend_venv_dir}/bin/activate

echo "Installing vLLM 0.21.0 (cu129)..."
uv pip install 'https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl'
echo "Pinning torch to 2.11.0+cu126 (IDUN driver only supports CUDA 12.9)..."
uv pip install --reinstall 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' --torch-backend=cu126

echo "Upgrading transformers and huggingface_hub..."
uv pip install --upgrade transformers huggingface_hub

rm -rf $HOME/.cache/flashinfer/

echo "vLLM setup complete!"
python -c "import vllm; print(f'vllm {{vllm.__version__}}')"
python -c "import transformers; print(f'transformers {{transformers.__version__}}')"
"""


def build_serve_command(
    model: str,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    override_args: str | None = None,
    backend: str = "vllm",
) -> str:
    """Build the vLLM or SGLang serve command.

    Override args are treated as authoritative for backend-specific serve flags,
    except that auto-added parallelism is still applied when those args do not
    specify parallelism explicitly.
    """
    if backend == "sglang":
        # SGLang command structure
        cmd = f"python -m sglang.launch_server --model-path {shlex.quote(model)} --host 0.0.0.0 --port {port}"

        # SGLang uses --tp-size for tensor parallelism
        if _has_any_cli_flag(override_args, "--tp", "--tp-size"):
            # User specified explicit parallelism, don't auto-add
            pass
        cmd = f"{cmd} --tp-size {tensor_parallel_size}"
    else:
        # vLLM command structure
        cmd = f"vllm serve {shlex.quote(model)} --host 0.0.0.0 --port {port}"

        has_explicit_parallelism = _has_any_cli_flag(
            override_args,
            "--pipeline-parallel-size",
            "--tensor-parallel-size",
        )

        if tensor_parallel_size > 1 and not has_explicit_parallelism:
            # Auto-detect parallelism type based on model size
            # Large models (70B+) work better with pipeline parallelism
            size_match = re.search(r"(\d+)B", model, re.IGNORECASE)
            model_size = int(size_match.group(1)) if size_match else 0

            # Check if tensor parallel size is a power of 2 (safer for most models)
            is_power_of_2 = (tensor_parallel_size & (tensor_parallel_size - 1)) == 0

            if model_size >= 80:
                # Use pipeline parallelism for very large models (80B+)
                cmd = f"{cmd} --pipeline-parallel-size {tensor_parallel_size}"
                if not override_args:
                    # Add memory-efficient defaults for large models only when
                    # the caller did not provide custom serve tuning.
                    cmd = f"{cmd} --gpu-memory-utilization 0.85 --max-model-len 16384 --enable-chunked-prefill --enforce-eager --kv-cache-dtype fp8"
                    # Fix max_num_batched_tokens to be divisible by parallelism size
                    batched_tokens = (
                        (8192 // tensor_parallel_size) + 1
                    ) * tensor_parallel_size
                    cmd = f"{cmd} --max-num-batched-tokens {batched_tokens}"
            elif not is_power_of_2:
                # Non-power-of-2 GPU counts often cause divisibility issues with tensor parallelism
                # Use pipeline parallelism as safer default
                cmd = f"{cmd} --pipeline-parallel-size {tensor_parallel_size}"
                console.print(
                    f"[yellow]ℹ Using pipeline parallelism (GPU count {tensor_parallel_size} is not a power of 2)[/yellow]"
                )
            else:
                # Use tensor parallelism for smaller models with power-of-2 GPU counts
                cmd = f"{cmd} --tensor-parallel-size {tensor_parallel_size}"

    if override_args:
        cmd = f"{cmd} {override_args}"
    return cmd


def _has_any_cli_flag(args: str | None, *flags: str) -> bool:
    """Return True when args contain any flag as a standalone CLI option."""
    if not args:
        return False

    return any(
        re.search(rf"(?<!\S){re.escape(flag)}(?:\s|=|$)", args) for flag in flags
    )


def _build_hf_snapshot_resolver(model: str) -> str:
    """Shell snippet that resolves ``$MODEL_REF`` to a local HF snapshot dir
    containing weights, falling back to the original model id.

    Some HF repos (Qwen3.5-122B-A10B-FP8 is the live example) have a metadata-only
    snapshot pointed to by ``refs/main`` plus a separate weights-only snapshot.
    If the loader follows ``refs/main`` it gets a snapshot with no .safetensors
    and dies with "Cannot find any model weights". This snippet picks the most
    recently modified snapshot that actually contains weight shards.
    """
    quoted = shlex.quote(model)
    return f"""# Resolve to a local HF snapshot containing weight shards if one exists.
MODEL_ID={quoted}
MODEL_REF="$MODEL_ID"
HF_CACHE_DIR="$HOME/.cache/huggingface/hub/models--${{MODEL_ID//\\//--}}"
if [ -d "$HF_CACHE_DIR/snapshots" ]; then
    # Prefer the newest snapshot that has at least one weight shard.
    for snap in $(ls -1dt "$HF_CACHE_DIR/snapshots"/*/ 2>/dev/null); do
        if compgen -G "${{snap}}*.safetensors" > /dev/null 2>&1 \\
           || compgen -G "${{snap}}*.bin" > /dev/null 2>&1 \\
           || compgen -G "${{snap}}*.pt" > /dev/null 2>&1; then
            MODEL_REF="${{snap%/}}"
            echo "Resolved $MODEL_ID -> $MODEL_REF"
            break
        fi
    done
fi
"""


def _wrap_serve_cmd_with_resolved_model(
    serve_cmd: str, model: str, backend: str, override_args: str | None
) -> str:
    """Substitute the literal model arg in ``serve_cmd`` with ``"$MODEL_REF"``
    and (if the caller didn't specify one) append a ``--served-model-name``
    flag so OpenAI clients keep seeing the original model id even when we
    serve from a local snapshot directory.
    """
    quoted_model = shlex.quote(model)
    if quoted_model in serve_cmd:
        serve_cmd = serve_cmd.replace(quoted_model, '"$MODEL_REF"', 1)
    if not _has_any_cli_flag(override_args, "--served-model-name"):
        serve_cmd = f"{serve_cmd} --served-model-name {quoted_model}"
    return serve_cmd


def build_serve_script(
    job_id: str,
    model: str,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    extra_args: str | None = None,
    backend: str = "vllm",
) -> str:
    """Build the full script to start vLLM or SGLang server in background."""
    backend_dir = get_backend_dir(backend)
    backend_venv_dir = get_backend_venv_dir(backend)
    serve_cmd = build_serve_command(
        model, port, tensor_parallel_size, override_args=extra_args, backend=backend
    )
    serve_cmd = _wrap_serve_cmd_with_resolved_model(
        serve_cmd, model, backend, extra_args
    )
    snapshot_resolver = _build_hf_snapshot_resolver(model)
    log_file = f"{VLLM_LOG_DIR}/{job_id}_serve.log"
    pid_file = f"{VLLM_LOG_DIR}/{job_id}.pid"

    return f"""
set -e
cd {backend_dir}
module load GCC/13.3.0
module load CUDA/12.9.1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CC=$(which gcc)
export CXX=$(which g++)
source {backend_venv_dir}/bin/activate
mkdir -p {VLLM_LOG_DIR}

# Kill any existing server for this job
if [ -f {pid_file} ]; then
    old_pid=$(cat {pid_file})
    kill $old_pid 2>/dev/null || true
    rm -f {pid_file}
fi

# Pass HF token if available
export HF_TOKEN="${{HF_TOKEN:-$HUGGINGFACE_TOKEN}}"

# Use only local HF cache — compute nodes have no internet
export HF_HUB_OFFLINE=1

# Disable CuDNN check for SGLang (PyTorch 2.9.1 compatibility)
{"export SGLANG_DISABLE_CUDNN_CHECK=1" if backend == "sglang" else ""}

# FlashInfer FP8 block-scale GEMM crashes on certain batch shapes (SM90)
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0

{snapshot_resolver}
# Start {backend} in background
echo "Running command: {serve_cmd}"
nohup {serve_cmd} > {log_file} 2>&1 &
echo $! > {pid_file}
echo "Started vLLM server (PID: $!)"
echo "Log file: {log_file}"

# Wait a moment and check if it's still running
sleep 3
if kill -0 $(cat {pid_file}) 2>/dev/null; then
    echo "Server is running"
else
    echo "Server failed to start. Check logs:"
    tail -20 {log_file}
    exit 1
fi
"""


def build_vllm_slurm_script(
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
    model: str,
    port: int = DEFAULT_PORT,
    extra_args: str | None = None,
    backend: str = "vllm",
) -> str:
    """Build SLURM batch script that starts vLLM or SGLang server automatically."""
    backend_dir = get_backend_dir(backend)
    backend_venv_dir = get_backend_venv_dir(backend)
    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}-{backend}")
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
    lines.append("")

    lines.append("set -e")
    lines.append(f"cd {backend_dir}")
    lines.append("module load GCC/13.3.0")
    lines.append("module load CUDA/12.9.1")
    lines.append("export CUDA_HOME=$(dirname $(dirname $(which nvcc)))")
    lines.append("export CC=$(which gcc)")
    lines.append("export CXX=$(which g++)")
    lines.append(f"source {backend_venv_dir}/bin/activate")
    lines.append(f"mkdir -p {VLLM_LOG_DIR}")
    lines.append('export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"')
    if backend == "sglang":
        lines.append("export SGLANG_DISABLE_CUDNN_CHECK=1")
    lines.append(
        "# FlashInfer FP8 block-scale GEMM crashes on certain batch shapes (SM90)"
    )
    lines.append("export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0")
    lines.append("")

    # Resolve to a local HF snapshot directory that contains weights — this
    # guards against repos whose refs/main points to a metadata-only snapshot.
    lines.append(_build_hf_snapshot_resolver(model))

    # Build serve command (and swap the model literal for "$MODEL_REF").
    serve_cmd = build_serve_command(
        model, port, gpus, override_args=extra_args, backend=backend
    )
    serve_cmd = _wrap_serve_cmd_with_resolved_model(
        serve_cmd, model, backend, extra_args
    )
    log_file = f"{VLLM_LOG_DIR}/${{SLURM_JOB_ID}}_serve.log"
    pid_file = f"{VLLM_LOG_DIR}/${{SLURM_JOB_ID}}.pid"

    # Start server in background
    lines.append(f"# Start {backend} server in background")
    lines.append(f'echo "Running command: {serve_cmd}"')
    lines.append(f"nohup {serve_cmd} > {log_file} 2>&1 &")
    lines.append("VLLM_PID=$!")
    lines.append(f"echo $VLLM_PID > {pid_file}")
    lines.append(f'echo "Started {backend} server (PID: $VLLM_PID)"')
    lines.append(f'echo "Log file: {log_file}"')
    lines.append(f'echo "Backend: {backend}"')
    lines.append(f'echo "Model: {model}"')
    lines.append(f'echo "Port: {port}"')
    lines.append(f'echo "GPUs: {gpus}"')
    lines.append("")

    # Wait and verify startup
    lines.append(f"# Wait for {backend} to initialize")
    lines.append("sleep 5")
    lines.append("if ! kill -0 $VLLM_PID 2>/dev/null; then")
    lines.append(f'    echo "{backend} failed to start. Check logs:"')
    lines.append(f"    tail -30 {log_file}")
    lines.append("    exit 1")
    lines.append("fi")
    lines.append(f'echo "{backend} server is running"')
    lines.append("")

    # Keep allocation alive (sleep will be killed when job ends)
    lines.append("# Keep allocation alive")
    lines.append(sleep_command)

    return "\n".join(lines) + "\n"


def start_vllm_server(
    config: IdunConfig,
    host: str,
    job_id: str,
    model: str,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    extra_args: str | None = None,
    backend: str = "vllm",
) -> tuple[bool, str]:
    """Start vLLM or SGLang server on a compute node.

    Returns (success, message).
    """
    import subprocess

    script = build_serve_script(
        job_id, model, port, tensor_parallel_size, extra_args, backend
    )

    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )
    compute_target = f"{config.ssh_user}@{host}" if config.ssh_user else host

    result = subprocess.run(
        ["ssh", "-J", jump_target, compute_target, f"bash -c {shlex.quote(script)}"],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        return False, result.stderr or result.stdout or "Failed to start server"

    # Update state file
    _update_server_state(
        config,
        job_id,
        VllmServer(
            job_id=job_id,
            model=model,
            port=port,
            host=host,
            pid=0,  # Will be read from PID file
            started_at=datetime.now().isoformat(),
            backend=backend,
        ),
    )

    return True, result.stdout


def stop_vllm_server(config: IdunConfig, host: str, job_id: str) -> tuple[bool, str]:
    """Stop vLLM server on a compute node.

    Returns (success, message).
    """
    import subprocess

    pid_file = f"{VLLM_LOG_DIR}/{job_id}.pid"
    script = f"""
if [ -f {pid_file} ]; then
    pid=$(cat {pid_file})
    if kill -0 $pid 2>/dev/null; then
        kill $pid
        rm -f {pid_file}
        echo "Stopped vLLM server (PID: $pid)"
    else
        rm -f {pid_file}
        echo "Server was not running (stale PID file removed)"
    fi
else
    echo "No PID file found for job {job_id}"
fi
"""

    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )
    compute_target = f"{config.ssh_user}@{host}" if config.ssh_user else host

    result = subprocess.run(
        ["ssh", "-J", jump_target, compute_target, f"bash -c {shlex.quote(script)}"],
        text=True,
        capture_output=True,
    )

    # Remove from state file
    _remove_server_state(config, job_id)

    if result.returncode != 0:
        return False, result.stderr or result.stdout or "Failed to stop server"

    return True, result.stdout


def _update_server_state(config: IdunConfig, job_id: str, server: VllmServer) -> None:
    """Update server state in remote state file."""
    # Read current state
    servers = get_all_servers(config)

    # Update
    servers[job_id] = server

    # Write back
    state_json = json.dumps({jid: s.to_dict() for jid, s in servers.items()}, indent=2)
    run_ssh(
        config,
        f"mkdir -p $(dirname {VLLM_STATE_FILE}) && cat > {VLLM_STATE_FILE} << 'EOF'\n{state_json}\nEOF",
        check=False,
    )


def _remove_server_state(config: IdunConfig, job_id: str) -> None:
    """Remove server from state file."""
    servers = get_all_servers(config)
    if job_id in servers:
        del servers[job_id]
        state_json = json.dumps(
            {jid: s.to_dict() for jid, s in servers.items()}, indent=2
        )
        run_ssh(
            config,
            f"mkdir -p $(dirname {VLLM_STATE_FILE}) && cat > {VLLM_STATE_FILE} << 'EOF'\n{state_json}\nEOF",
            check=False,
        )


def get_all_servers(config: IdunConfig) -> dict[str, VllmServer]:
    """Read all server states from remote state file."""
    result = run_ssh(
        config,
        f"cat {VLLM_STATE_FILE} 2>/dev/null || echo '{{}}'",
        check=False,
    )
    output = (result.stdout or "{}").strip()
    if not output:
        output = "{}"

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {}

    servers: dict[str, VllmServer] = {}
    for job_id, server_data in data.items():
        if isinstance(server_data, dict):
            servers[job_id] = VllmServer.from_dict(job_id, server_data)

    return servers


def get_server_status(config: IdunConfig, server: VllmServer) -> tuple[bool, str]:
    """Check if a vLLM server is running and healthy.

    Returns (is_running, status_message).
    """
    import subprocess

    pid_file = f"{VLLM_LOG_DIR}/{server.job_id}.pid"
    check_script = f"""
if [ -f {pid_file} ]; then
    pid=$(cat {pid_file})
    if kill -0 $pid 2>/dev/null; then
        # Check if API responds
        if curl -s --max-time 5 http://localhost:{server.port}/health > /dev/null 2>&1; then
            echo "HEALTHY"
        else
            echo "STARTING"
        fi
    else
        echo "DEAD"
    fi
else
    echo "NO_PID"
fi
"""

    jump_target = (
        f"{config.ssh_user}@{config.jump_host}" if config.ssh_user else config.jump_host
    )
    compute_target = (
        f"{config.ssh_user}@{server.host}" if config.ssh_user else server.host
    )

    result = subprocess.run(
        [
            "ssh",
            "-J",
            jump_target,
            compute_target,
            f"bash -c {shlex.quote(check_script)}",
        ],
        text=True,
        capture_output=True,
    )

    status = (result.stdout or "").strip()
    if status == "HEALTHY":
        return True, "OK"
    elif status == "STARTING":
        return True, "Starting..."
    elif status == "DEAD":
        return False, "Dead"
    else:
        return False, "Not running"


def inspect_server(config: IdunConfig, job_id: str) -> dict[str, str | int | bool]:
    """Inspect the tracked server for a job ID."""
    server = get_all_servers(config).get(job_id)
    if server is None:
        return {"found": False}

    is_running, status = get_server_status(config, server)
    return {
        "found": True,
        "job_id": server.job_id,
        "model": server.model,
        "backend": server.backend,
        "host": server.host,
        "port": server.port,
        "running": is_running,
        "status": status,
    }


def check_server_health_bulk(
    config: IdunConfig, servers: dict[str, VllmServer]
) -> dict[str, str]:
    """Check health of all servers (returns job_id -> status mapping)."""
    # Group by host to minimize SSH connections
    by_host: dict[str, list[VllmServer]] = {}
    for server in servers.values():
        if server.host not in by_host:
            by_host[server.host] = []
        by_host[server.host].append(server)

    results: dict[str, str] = {}
    for host, host_servers in by_host.items():
        for server in host_servers:
            is_running, status = get_server_status(config, server)
            results[server.job_id] = status

    return results


def render_servers(
    servers: dict[str, VllmServer], health: dict[str, str] | None = None
) -> None:
    """Render vLLM/SGLang servers as a table."""
    if not servers:
        console.print("No vLLM/SGLang servers tracked")
        return

    table = Table(title="Inference Servers")
    table.add_column("Job ID", style="cyan")
    table.add_column("Backend", style="blue")
    table.add_column("Model", style="green")
    table.add_column("Host")
    table.add_column("Port")
    table.add_column("Health", style="yellow")
    table.add_column("Started")

    for job_id, server in servers.items():
        status = health.get(job_id, "?") if health else "?"
        style = (
            "green"
            if status == "OK"
            else "red"
            if status in {"Dead", "Not running"}
            else "yellow"
        )
        table.add_row(
            job_id,
            server.backend,
            server.model,
            server.host,
            str(server.port),
            f"[{style}]{status}[/{style}]",
            server.started_at[:19] if server.started_at else "-",
        )

    console.print(table)


CUSTOM_MODELS_FILE = Path("~/.config/idun/custom_models.json").expanduser()


def _load_custom_models() -> dict[str, dict[str, str | int]]:
    """Load user-saved custom models from disk."""
    if not CUSTOM_MODELS_FILE.exists():
        return {}
    try:
        data = json.loads(CUSTOM_MODELS_FILE.read_text())
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_custom_model(
    model: str, gpus: int, gpu_type: str, extra_args: str | None = None
) -> None:
    """Persist a custom model so it appears in future selections."""
    models = _load_custom_models()
    models[model] = {
        "model": model,
        "gpus": gpus,
        "gpu_type": gpu_type,
        "extra_args": extra_args or "",
    }
    CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_MODELS_FILE.write_text(json.dumps(models, indent=2) + "\n")


def prompt_model() -> tuple[str, int, str, str | None]:
    """Interactive model selection.

    Returns (model_name, recommended_gpus, recommended_gpu_type, extra_args).
    """
    from InquirerPy import inquirer

    choices = []
    for name, preset in MODEL_PRESETS.items():
        vram_gb = preset.get("vram_gb", 0)
        label = f"{name} (~{vram_gb}GB)"
        choices.append({"name": label, "value": name})

    # Add previously used custom models
    custom_models = _load_custom_models()
    for name in custom_models:
        if name not in MODEL_PRESETS:
            choices.append({"name": f"{name} (custom)", "value": name})

    choices.append({"name": "Custom model...", "value": "__custom__"})

    selected = inquirer.select(
        message="Select model",
        choices=choices,
    ).execute()

    if selected == "__custom__":
        model = inquirer.text(
            message="Model name (HuggingFace model ID)",
        ).execute()
        _save_custom_model(model, 1, "")
        return model, 1, "", None

    if selected in MODEL_PRESETS:
        preset = MODEL_PRESETS[selected]
        return (
            str(preset["model"]),
            int(preset.get("gpus", 1)),
            str(preset.get("gpu_type", "")),
            None,
        )

    # Selected a previously saved custom model
    custom = custom_models[selected]
    return (
        str(custom["model"]),
        int(custom.get("gpus", 1)),
        str(custom.get("gpu_type", "")),
        str(custom.get("extra_args", "")) or None,
    )


def prompt_server(
    config: IdunConfig, servers: dict[str, VllmServer]
) -> VllmServer | None:
    """Prompt user to select a running server."""
    from InquirerPy import inquirer

    if not servers:
        console.print("No vLLM servers tracked")
        return None

    choices = []
    for job_id, server in servers.items():
        label = f"{job_id} | {server.model} | {server.host}:{server.port}"
        choices.append({"name": label, "value": server})

    choices.append({"name": "Cancel", "value": None})

    return inquirer.select(
        message="Select server",
        choices=choices,
    ).execute()


# ============================================================================
# Model download / cache management
# ============================================================================


@dataclass
class CachedModel:
    """A HuggingFace model cached on disk."""

    folder: str  # e.g. "models--Qwen--Qwen2.5-7B-Instruct"
    size: str  # human-readable, e.g. "14G"
    display_name: str  # e.g. "Qwen/Qwen2.5-7B-Instruct"


def list_cached_models(config: IdunConfig) -> list[CachedModel]:
    """List HuggingFace models cached in ~/.cache/huggingface/hub/."""
    result = run_ssh(
        config,
        "du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null || true",
        check=False,
    )
    models: list[CachedModel] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        size, path = parts
        folder = path.strip().rstrip("/").split("/")[-1]
        # Convert "models--Qwen--Qwen2.5-7B-Instruct" → "Qwen/Qwen2.5-7B-Instruct"
        display_name = folder.removeprefix("models--").replace("--", "/")
        models.append(CachedModel(folder=folder, size=size, display_name=display_name))
    return models


def download_model(config: IdunConfig, model: str) -> None:
    """Download a HuggingFace model with live-streamed output."""
    python = get_backend_python("vllm")
    script = (
        "import os, sys; "
        "os.environ.setdefault('HF_TOKEN', os.environ.get('HUGGINGFACE_TOKEN', '')); "
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({model!r})"
    )
    cmd = f"{python} -c {shlex.quote(script)}"
    run_ssh_stream(config, cmd)


def delete_cached_model_folders(config: IdunConfig, folders: list[str]) -> None:
    """Delete cached model folders from ~/.cache/huggingface/hub/."""
    for folder in folders:
        safe_folder = shlex.quote(folder)
        run_ssh(config, f"rm -rf ~/.cache/huggingface/hub/{safe_folder}", check=False)
