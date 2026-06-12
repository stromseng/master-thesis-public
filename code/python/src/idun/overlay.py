"""Git overlay bundle utilities for syncing local changes to IDUN."""

from __future__ import annotations

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from idun.config import IdunConfig, REMOTE_REPO_PATH
from idun.ssh import run_ssh, scp_to_jump


def run_git(args: list[str], cwd: Path) -> str:
    """Run git command and return output."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "git command failed"
        )
    return (result.stdout or "").strip()


def find_repo_root(start: Path) -> Path | None:
    """Find git repository root from starting path."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(start),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    root = (result.stdout or "").strip()
    return Path(root) if root else None


def resolve_repo_root() -> Path:
    """Resolve git repository root, trying multiple candidates."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in candidates:
        root = find_repo_root(candidate)
        if root:
            return root
    raise RuntimeError("Unable to locate git repository root")


def collect_changed_files(repo_root: Path) -> tuple[set[str], set[str]]:
    """Collect changed and deleted files compared to origin/main."""
    changed: set[str] = set()
    deleted: set[str] = set()
    name_only_cmds = [
        ["diff", "--name-only", "origin/main...HEAD"],
        ["diff", "--name-only"],
        ["diff", "--cached", "--name-only"],
        ["ls-files", "--others", "--exclude-standard"],
    ]
    deleted_cmds = [
        ["diff", "--name-only", "--diff-filter=D", "origin/main...HEAD"],
        ["diff", "--name-only", "--diff-filter=D"],
        ["diff", "--cached", "--name-only", "--diff-filter=D"],
    ]
    for cmd in name_only_cmds:
        output = run_git(cmd, repo_root)
        changed.update(line for line in output.splitlines() if line.strip())
    for cmd in deleted_cmds:
        output = run_git(cmd, repo_root)
        deleted.update(line for line in output.splitlines() if line.strip())
    changed -= deleted
    return changed, deleted


def create_overlay_bundle(
    repo_root: Path, changed: set[str], deleted: set[str]
) -> Path | None:
    """Create a tarball with changed files and deletion manifest."""
    if not changed and not deleted:
        return None
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        bundle_path = Path(handle.name)
    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel_path in sorted(changed):
            abs_path = repo_root / rel_path
            if not abs_path.exists():
                continue
            tar.add(abs_path, arcname=rel_path)
        if deleted:
            manifest = "\n".join(sorted(deleted)).rstrip() + "\n"
            data = manifest.encode("utf-8")
            info = tarfile.TarInfo(name=".idun_deleted.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return bundle_path


def upload_overlay_bundle(config: IdunConfig, bundle_path: Path) -> str:
    """Upload overlay bundle to remote and return path."""
    remote_dir = "~/.cache/idun/overlays"
    _ = run_ssh(config, f"mkdir -p {remote_dir}")
    scp_path = f"{remote_dir}/overlay_{uuid4().hex}.tar.gz"
    scp_to_jump(config, bundle_path, scp_path)
    return scp_path.replace("~/", "$HOME/")


def reset_remote_repo(config: IdunConfig) -> None:
    """Reset remote repository to origin/main."""
    command = (
        f'cd "{REMOTE_REPO_PATH}" && '
        "git fetch origin main && "
        "git reset --hard origin/main"
    )
    _ = run_ssh(config, command)
