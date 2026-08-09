"""Path helpers for explicitly configured A_memorix runtimes."""

from __future__ import annotations

from pathlib import Path


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve a caller-owned data directory without creating it."""

    return Path(data_dir).expanduser().resolve()


def resolve_runtime_path(data_dir: str | Path, path: str | Path) -> Path:
    """Resolve a path and reject traversal outside the runtime data directory."""

    root = resolve_data_dir(data_dir)
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes data directory: {path}") from exc
    return resolved
