"""Feature flags for Lattice Mem (measurement only; defaults leave behavior unchanged)."""

from __future__ import annotations

import os
from typing import Any


def lattice_enabled(reg: Any = None) -> bool:
    """Lattice views (claims/timeline/conflicts/profile observe + ability routes).

    ``HOMEBASE_MEM_LATTICE=0`` (or ``reg.mem_lattice = False``) turns them off for the
    on/off ablation (PRD W9): episodic drawers + fusion retrieval + Score-as-sort stay.
    Default on.
    """
    env = os.environ.get("HOMEBASE_MEM_LATTICE")
    if env is not None and env.strip() != "":
        return env.strip() != "0"
    return bool(getattr(reg, "mem_lattice", True))


def jev_frugal(reg: Any = None) -> bool:
    """Ask fewer billed Jev questions on the write path (measured trade-off).

    - explicit-tier writes ask only the ``preference`` type (the only type the memory
      path consumes; the others feed Nexus page labels),
    - relation checks go to the 5 most lexically similar candidates instead of 10.
    ``HOMEBASE_MEM_JEV_FRUGAL=1`` or ``reg.mem_jev_frugal = True``. Default off.
    """
    env = os.environ.get("HOMEBASE_MEM_JEV_FRUGAL")
    if env is not None and env.strip() != "":
        return env.strip() == "1"
    return bool(getattr(reg, "mem_jev_frugal", False))
