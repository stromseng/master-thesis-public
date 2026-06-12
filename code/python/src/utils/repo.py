"""Shared utility for locating the repository root directory."""

from __future__ import annotations

from pathlib import Path

_cached_root: Path | None = None


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from *start* until a directory with pyproject.toml, code/, and data/ is found.

    The result is cached after the first successful lookup.
    """
    global _cached_root  # noqa: PLW0603
    if _cached_root is not None:
        return _cached_root

    origin = (start or Path(__file__)).resolve()
    for candidate in [origin, *origin.parents]:
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "code").exists()
            and (candidate / "data").exists()
        ):
            _cached_root = candidate
            return candidate
    raise RuntimeError(f"Could not locate repo root from {origin}")


REPO_ROOT: Path = find_repo_root()
