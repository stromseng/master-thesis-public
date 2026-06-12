# IDUN CLI script submission

This document describes how the IDUN CLI submits and runs Python scripts.

## Overview

The CLI submits a Slurm batch job that runs a Python module from this repo.
It keeps IDUN in sync with `origin/main`, then overlays your local uncommitted
and unpushed changes so you can run test scripts without committing.

## Step-by-step flow

1) Script selection (local)
- You choose "Run a Python script (module)".
- The CLI lists modules from `code/python/scripts/idun`.
- The script is executed as a module, e.g. `scripts.idun.test`.

2) Safety confirmation (local)
- You must confirm a hard reset on IDUN: `~/repos/master-thesis` will be reset
  to `origin/main` before the job starts.

3) Remote repo reset (login host)
- The CLI runs on the login host:
  - `cd ~/repos/master-thesis`
  - `git fetch origin main`
  - `git reset --hard origin/main`

4) Local overlay bundle (local)
- The CLI builds a tarball containing only files that differ from `origin/main`:
  - Unpushed commits (`git diff origin/main...HEAD`)
  - Staged and unstaged changes (`git diff`, `git diff --cached`)
  - Untracked files (excluding gitignored)
- Deletions are recorded in a `.idun_deleted.txt` manifest.

5) Upload overlay (login host)
- The tarball is uploaded to `~/.cache/idun/overlays/...` via scp.

6) Batch job execution (compute node)
- The Slurm job:
  - Extracts the overlay into `~/repos/master-thesis`
  - Applies deletions from `.idun_deleted.txt`
  - Runs `uv sync --frozen` (or `uv sync` if no lockfile)
  - Runs the script with unbuffered output:
    - `PYTHONUNBUFFERED=1 uv run python -u -m scripts.idun.<module>`

7) Post-submit behavior (local)
You can choose one of the following:
- Run in background (no wait)
- Wait and stream logs (tails stdout/stderr and exits when the job finishes)
- Wait and open shell (interactive `srun --pty bash -l`, supports tunnels)

## Why this avoids committing scripts

The overlay bundle contains local changes only. This lets you run experiments
on IDUN without pushing them to `main`.
