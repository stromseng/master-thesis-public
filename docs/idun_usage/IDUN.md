# IDUN HPC Cluster

Commands for running workloads on the NTNU IDUN HPC cluster. All commands are run via:

```bash
just idun <subcommand>
```

This runs `python -m idun` from `code/python/`. See [IDUN_CLI_SUBMIT.md](IDUN_CLI_SUBMIT.md) for how the CLI submits scripts, and [SETUP_GITHUB_UV_IDUN.md](SETUP_GITHUB_UV_IDUN.md) for initial cluster setup.

## vLLM Server Management

| Command | Description |
| --- | --- |
| `just idun vllm start` | Start vLLM server on GPU node |
| `just idun vllm stop` | Stop vLLM server and cancel SLURM job |

Example: Start vLLM with a specific model on H200 GPUs, then run evals against it using `EVAL_PROVIDER=vllm`.

## Job Management

| Command | Description |
| --- | --- |
| `just idun submit <script>` | Submit a Python script as a SLURM job |
| `just idun shell` | Open SSH shell to compute node |
| `just idun monitor` | Show GPU/CPU usage across nodes |

The CLI handles:
- SLURM job submission with GPU allocation
- SSH port forwarding (8001 for FastAPI, 8000 for vLLM)
- Local overlay bundles (run uncommitted code without pushing)
- Job monitoring and log tailing

Inference backends are installed into separate environments:
- `~/repos/vllm/.venv`
- `~/repos/sglang/.venv`

## Docling PDF Conversion

GPU-accelerated PDF to markdown conversion using Docling on IDUN. See [the old justfile](../backup.just) for legacy recipes, or use the IDUN CLI directly.

## Monitoring

```bash
just idun monitor           # GPU power, utilization, memory, temp + CPU/RAM
just idun monitor --queue   # Show SLURM job queue
just idun monitor --watch 5 # Auto-refresh every 5 seconds
```

## Related Docs

- [IDUN_CLI_SUBMIT.md](IDUN_CLI_SUBMIT.md) - How the CLI submits and runs scripts remotely
- [SETUP_GITHUB_UV_IDUN.md](SETUP_GITHUB_UV_IDUN.md) - Initial cluster setup (SSH, Git, UV)
- [ocr/paddle.md](ocr/paddle.md) - PaddleOCR on IDUN with vLLM
