# IDUN Cluster CLI Context

This document provides context for LLMs working with the IDUN cluster management system.

## Overview

IDUN is NTNU's HPC (High-Performance Computing) cluster with GPU nodes. This codebase provides a custom TUI/CLI to simplify GPU job submission, script execution, and artifact retrieval. The CLI is invoked as a Python package: `python -m idun` (or via `just idun`).

**For deeper understanding of the flow, read:** [\_\_main\_\_.py](__main__.py) (CLI entry point) and [vllm.py](vllm.py) (vLLM/SGLang server management)

## Direct SSH Access

You can SSH directly to the IDUN server if needed:
```bash
ssh idun
```

The repository is located at `~/repos/master-thesis` on the server. Direct SSH access is allowed and can be useful for:
- Debugging issues on the server
- Checking file states manually
- Running ad-hoc commands
- Inspecting logs or artifacts directly

## Architecture

```
code/python/
├── src/idun/
│   ├── __main__.py      # Main CLI entry point (typer-based TUI)
│   ├── vllm.py          # vLLM/SGLang server management (serve, setup, status, logs)
│   ├── config.py        # Configuration management (TOML + env vars)
│   ├── ssh.py           # SSH/SCP utilities for remote operations
│   ├── slurm.py         # SLURM job submission and management
│   ├── jobs.py          # Job listing, log streaming
│   ├── overlay.py       # Git overlay bundle creation/upload
│   ├── fuse_overlay.py  # Fuse-overlayfs isolation utilities
│   ├── prompts.py       # Interactive prompts (InquirerPy)
│   └── monitor.py       # CPU/GPU resource monitoring
└── scripts/idun/            # Scripts to run on IDUN (executed as modules)
    ├── convert_pdfs.py      # PDF to markdown conversion
    └── dedup_mcq.py         # MCQ deduplication pipeline
```

## Key Concepts

### 1. Fuse-Overlayfs Isolation

Each script run uses **fuse-overlayfs** to create an isolated execution environment:

- **lowerdir**: Base repository at `~/repos/master-thesis` (read-only)
- **upperdir**: Contains local changes (from overlay bundle) + any files created during run
- **merged**: Combined view where the script executes

This allows:
- Multiple scripts to run concurrently without interfering
- Local changes to be synced without committing
- Automatic artifact capture (any created/modified files)

### 2. Overlay Bundle Flow

When submitting a job:
1. CLI detects uncommitted changes vs `origin/main`
2. Creates a tarball with changed/deleted files
3. Uploads bundle to `~/.cache/idun/overlays/`
4. SLURM script extracts bundle into upperdir before mounting

### 3. Job Submission Modes

**New Job Submission:**
```
just idun submit
→ Prompts for script, GPU config, time limit, etc.
→ Resets remote repo to origin/main
→ Creates overlay bundle with local changes
→ Submits SLURM job
→ Optionally waits for RUNNING and streams logs
```

**Reuse Existing Job:**
```
just idun submit
→ Select "Run on an existing RUNNING instance"
→ Creates isolated overlay (no base repo reset needed)
→ Runs script via srun or SSH to compute node
→ Collects artifacts after completion
```

### 4. Artifacts

Files created/modified during a run are captured in the overlay's upperdir and saved to:
```
~/.cache/idun/artifacts/<timestamp>_<script>_<duration>/
```

Download artifacts locally:
```bash
just idun artifacts --run-id <id>
# or interactively
just idun artifacts
```

## Configuration

Configuration sources (in order of precedence):
1. Environment variables (`IDUN_*`)
2. TOML config at `~/.config/idun/config.toml`
3. Defaults

Key settings:
- `jump_host`: SSH jump host (default: `idun`)
- `account`: SLURM account (default: `share-ie-idi`)
- `partition_gpu`: GPU partition (default: `GPUQ`)
- `user_email`: For job notifications

## Common Commands

```bash
# Interactive job submission wizard
just idun submit

# Submit with explicit options
just idun submit --script convert_pdfs.py --args "--force" --gpus 2 --gpu-vram 80g

# List jobs and manage them interactively
just idun list

# Stream logs from a job
just idun logs

# Monitor GPU/CPU resources
just idun monitor

# Download artifacts from completed runs
just idun artifacts

# Cancel all jobs
just idun cancel-all
```

## Inference Backend: vLLM and SGLang

The IDUN CLI supports two high-performance inference backends for serving LLMs:

### vLLM (default)
- Mature, stable APIs with broad model support
- PagedAttention for efficient memory management
- Best for general-purpose inference workloads
- Excellent single-sequence efficiency

### SGLang (alternative)
- **29% faster** than vLLM for multi-turn conversations
- RadixAttention for automatic KV cache reuse across requests
- Better for agent workflows and complex serving patterns
- Optimized for chat applications and RAG systems with follow-ups

### Using SGLang

```bash
# Interactive serve with backend selection (prompted)
just idun vllm serve

# Explicit SGLang flag
just idun vllm serve --sglang

# With specific model
just idun vllm serve Qwen/Qwen2.5-7B-Instruct --sglang --gpus 1 --gpu-type a100

# Setup SGLang environment
just idun vllm setup --backend sglang

# Setup both backends
just idun vllm setup --backend both

# Check status (shows backend column)
just idun vllm status
```

### Backend Comparison

| Feature | vLLM | SGLang |
|---------|------|--------|
| Single-sequence efficiency | ✓✓ | ✓ |
| Multi-turn conversations | ✓ | ✓✓ |
| KV cache strategy | PagedAttention | RadixAttention |
| API maturity | High | Medium |
| Best use cases | General inference | Agents, chat, RAG |

### When to Use SGLang
- Multi-turn chat applications (e.g., maritime advisors)
- RAG systems with conversation history
- Agent-based workflows with repeated prefix patterns
- High-throughput evaluation runs with conversation context

### Technical Details
- **vLLM**: Uses `vllm serve` with `--tensor-parallel-size` or `--pipeline-parallel-size`
- **SGLang**: Uses `python -m sglang.launch_server` with `--tp` flag
- Both backends share the same state tracking, port forwarding, and log management
- vLLM env: `~/repos/vllm/.venv`
- SGLang env: `~/repos/sglang/.venv`
- Installation:
  - vLLM: `uv pip install vllm --torch-backend=auto`
  - SGLang: `uv pip install --upgrade "sglang[all]"` (plus the existing `sgl_kernel` / dependency steps)

## GPU Configuration

Available GPU types and VRAM:
- P100: 16GB
- V100: 16GB, 32GB
- A100: 40GB, 80GB
- H100: 80GB
- H200: 80GB

SLURM constraints are automatically set based on VRAM selection.

## Script Execution Details

Scripts in `code/python/scripts/idun/` are run as Python modules:
```bash
uv run python -m scripts.idun.convert_pdfs --force
```

The execution environment:
1. Mounts fuse-overlayfs with local changes
2. Uses shared venv at `$HOME/.cache/idun/venv` (locked with flock to prevent concurrent corruption)
3. Runs `uv sync --frozen` (or `uv sync` if no lockfile)
4. Executes the module with `PYTHONUNBUFFERED=1`
5. On exit: unmounts overlay, saves artifacts, cleans up

## Example Workflow

```bash
# 1. Make local changes to code
vim code/python/scripts/idun/convert_pdfs.py

# 2. Submit to IDUN (changes auto-synced via overlay)
just idun submit
# Select: convert_pdfs.py
# Select: Run on existing RUNNING instance (if available)
# Select: Run in foreground

# 3. Watch output stream
# Output shows: "Using isolated overlay: /tmp/idun-overlay-xxx/merged"

# 4. After completion, download artifacts
just idun artifacts
# Select the run by timestamp
# Files downloaded to ./artifacts/<run-id>/
```

## Remote Paths

- **Repository: `~/repos/master-thesis`** (main working directory)
- Overlays: `~/.cache/idun/overlays/`
- Artifacts: `~/.cache/idun/artifacts/`
- Logs: `~/.cache/idun/logs/`
- Shared venv: `~/.cache/idun/venv`
- SLURM scripts: `~/.cache/idun/slurm/`

## Cluster Resource Investigation

When the user asks what's available/free on IDUN, run the following commands via `ssh idun` to build a full picture. Note that some users schedule jobs with `--begin` (delayed start), so SLURM-reported "free" resources may already be reserved for future jobs — always mention this caveat.

### Step 1: GPU availability per node (free vs allocated GPUs)

```bash
ssh idun "sinfo -p GPUQ -N -O 'NodeList:20,Gres:30,GresUsed:30,StateLong:15'"
```

This shows total GRES (GPUs) and GRES_USED per node. Compare the counts to find free GPUs. For example `gpu:h100:8` total with `gpu:h100:3` used = 5 free H100s. Nodes in `drained` or `drained*` state are unavailable.

### Step 2: CPU availability per node

```bash
ssh idun "sinfo -p GPUQ -N -O 'NodeList:20,Gres:30,CPUsState:20,StateLong:15'"
```

The `CPUS(A/I/O/T)` column shows Allocated/Idle/Other/Total. The "Idle" (I) value is the number of free CPU cores on that node.

### Step 3: RAM availability per node

```bash
ssh idun "sinfo -p GPUQ -N -O 'NodeList:20,Gres:30,Memory:12,AllocMem:12,FreeMem:12,CPUsState:20,StateLong:15'"
```

- `MEMORY`: Total RAM available to SLURM (in MB)
- `ALLOCMEM`: RAM already allocated to jobs (in MB)
- `FREE_MEM`: Actual free physical RAM (in MB)
- Free SLURM RAM = MEMORY - ALLOCMEM

### Step 4: Check pending/running jobs for the user

```bash
ssh idun "squeue -u $USER -o '%i %T %C %b %N %r' --noheader"
```

Check pending job reasons. Common reasons:
- `Priority`: Other users' jobs are ahead in the fair-share queue
- `Resources`: Not enough resources available right now
- `BeginTime`: Job is scheduled for a future start time

### Step 5: Check for delayed-start (--begin) jobs blocking resources

```bash
ssh idun "squeue -p GPUQ -o '%i %u %T %C %b %S %r' --noheader | grep -i 'BeginTime\|PENDING'"
```

This reveals jobs that have reserved resources for a future start time. These "phantom" reservations mean that SLURM-reported idle resources may not actually be available.

### How to present results

Build a table per GPU type showing: Node, GPU Type, Total GPUs, Used GPUs, Free GPUs, Idle CPUs, Free RAM (GB), State. Then provide:

1. **Summary table** grouped by GPU type with totals
2. **Recommendation** of which node/GPU type to target based on actual availability
3. **Caveat** that `--begin` jobs may have reserved some of the "free" resources
4. **Tips** to improve scheduling chances:
   - Relax GPU constraints (e.g. `gpu40g|gpu80g` instead of just `gpu80g`)
   - Lower wall-time (shorter jobs schedule faster)
   - Lower CPU count if not strictly needed
   - Try less popular GPU types (P100/V100) if the workload allows

## CRITICAL: NEVER Cancel SLURM Jobs Without Explicit User Confirmation

**NEVER run `scancel` on a SLURM job to kill individual processes.** Long-running GPU allocations (especially multi-GPU, multi-day jobs) are extremely expensive and hard to re-acquire. Cancelling the job kills the ENTIRE allocation, not just the misbehaving process.

### How to kill processes inside a running job WITHOUT cancelling the job:

**Option 1: srun into the job and kill processes**
```bash
# Use srun to execute a command within the job's allocation
ssh idun "srun --jobid=<JOBID> --overlap --pty kill <PID1> <PID2>"
```

**Option 2: SSH to the compute node directly and kill as the job owner**
```bash
# SSH to the node, then kill the process (must be run within the job's cgroup)
ssh idun "ssh <node> 'srun --jobid=<JOBID> --overlap kill <PID1> <PID2>'"
```

**Option 3: Cancel only a specific job step (not the whole job)**
```bash
# List job steps first
ssh idun "scontrol listjobs <JOBID>"
# Cancel only the offending step
ssh idun "scancel <JOBID>.<STEP_ID>"
```

### Rules:
1. **NEVER use `scancel <JOBID>` to kill processes** — this destroys the entire allocation
2. **ALWAYS ask the user before running `scancel`** on any job, even if it seems obvious
3. **Try `kill` via srun/SSH first** to stop individual processes within the job
4. GPU allocations can take days to schedule — losing one is catastrophic

## Troubleshooting

**Job stuck in PENDING:**
- Check queue: `squeue -u $USER`
- View estimated start time in `just idun list`

**Script fails immediately:**
- Check logs: `just idun logs`
- Common issue: missing dependencies in pyproject.toml

**Overlay mount fails:**
- Verify fuse-overlayfs is available: `which fuse-overlayfs`
- Check `/tmp` has space: `df -h /tmp`

**Concurrent runs interfere:**
- Each run gets unique overlay (UUID-based)
- Shared venv is protected by flock
