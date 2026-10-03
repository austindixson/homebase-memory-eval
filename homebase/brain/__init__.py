"""Home Base brain: Jev compaction, Mem-lite, router, AutoMode proxy."""

from __future__ import annotations

__all__ = ["__version__", "create_app"]

__version__ = "0.1.0"


def create_app(*args, **kwargs):  # lazy — avoids importing FastAPI at package import
    from homebase.brain.proxy import create_app as _create_app

    return _create_app(*args, **kwargs)
