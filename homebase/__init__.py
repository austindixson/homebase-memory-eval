"""Home Base: contract tests and helpers for ghost32 services."""

from __future__ import annotations

import os
from pathlib import Path

# Default root on ghost32's 5TB volume; override with HOMEBASE_ROOT.
DEFAULT_HOMEBASE_ROOT = Path("/Volumes/DRIVE/HomeBase")
REQUIRED_DIRS = ("Projects", "Models", "Media", "Services", "Secrets")


def homebase_root() -> Path:
    return Path(os.environ.get("HOMEBASE_ROOT", DEFAULT_HOMEBASE_ROOT))


def required_paths(root: Path | None = None) -> list[Path]:
    base = root or homebase_root()
    return [base / name for name in REQUIRED_DIRS]


def ensure_layout(root: Path | None = None) -> Path:
    """Create the Home Base directory layout; return root."""
    base = root or homebase_root()
    base.mkdir(parents=True, exist_ok=True)
    for p in required_paths(base):
        p.mkdir(parents=True, exist_ok=True)
    secrets = base / "Secrets"
    try:
        os.chmod(secrets, 0o700)
    except OSError:
        pass
    return base
