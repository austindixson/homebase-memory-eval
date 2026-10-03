"""Shared evidence limits + per-request stage trace for Lattice retrieve.

``MAX_EVIDENCE_CHARS`` is the single per-passage cap used by the usefulness filter,
the pack formatter and Prove-or-Abstain (PRD W5). Every clip below the source length
is recorded so a truncation can never be silent; callers budget by passage *count*.

The trace (PRD W2) records, per retrieve call, which passages survived each stage
(``retrieved`` -> ``kept`` -> ``packed``) with the exact text at that stage, so the
bench can say *where* gold evidence was lost.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from homebase.brain.mem.timeparse import iso_date

MAX_EVIDENCE_CHARS = int(os.environ.get("HOMEBASE_MEM_MAX_EVIDENCE_CHARS") or "2000")
# Total characters Prove-or-Abstain may read; whole passages are dropped (and logged)
# beyond this rather than silently cutting the tail of one passage.
PROVE_PACK_CHARS = int(os.environ.get("HOMEBASE_MEM_PROVE_PACK_CHARS") or "10000")
TRACE_TEXT_CHARS = 6000


@dataclass
class StageTrace:
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    packed: list[dict[str, Any]] = field(default_factory=list)
    truncations: list[dict[str, Any]] = field(default_factory=list)
    routes: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_truncation(self, stage: str, node_id: str, chars: int, limit: int) -> None:
        with self._lock:
            self.truncations.append(
                {"stage": stage, "id": node_id, "chars": chars, "limit": limit}
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved": self.retrieved,
            "kept": self.kept,
            "packed": self.packed,
            "truncations": self.truncations,
            "routes": self.routes,
        }


_TRACE: StageTrace | None = None


def begin_trace() -> StageTrace:
    global _TRACE
    _TRACE = StageTrace()
    return _TRACE


def current_trace() -> StageTrace | None:
    return _TRACE


def last_trace() -> dict[str, Any] | None:
    return _TRACE.to_dict() if _TRACE is not None else None


def clip(text: str, *, stage: str, node_id: str = "", limit: int | None = None) -> str:
    """Clip ``text`` to ``limit`` (default ``MAX_EVIDENCE_CHARS``), logging any cut."""
    lim = MAX_EVIDENCE_CHARS if limit is None else int(limit)
    text = text or ""
    if len(text) <= lim:
        return text
    tr = _TRACE
    if tr is not None:
        tr.record_truncation(stage, node_id, len(text), lim)
    return text[:lim]


def node_row(node: Any, *, text: str | None = None, **extra: Any) -> dict[str, Any]:
    body = node.content if text is None else text
    row = {
        "id": node.id,
        "chars": len(node.content or ""),
        "text": (body or "")[:TRACE_TEXT_CHARS],
    }
    row.update(extra)
    return row


def evidence_prefix(node: Any) -> str:
    """Use the same source and temporal metadata in proof and answer prompts."""
    parts = [f"source={node.id}"]
    source_position = getattr(node, "source_position", None)
    source_role = getattr(node, "source_role", None)
    if source_position is not None:
        parts.append(f"position={int(source_position)}")
    if source_role:
        parts.append(f"speaker={source_role}")
    if node.valid_to is not None:
        parts.append("superseded")
    # A write-time default is not evidence of an event date.
    if node.valid_from is not None and abs(node.valid_from - (node.ts or 0.0)) > 1.0:
        parts.append(f"date={iso_date(node.valid_from)}")
    for key in ("project", "person", "session"):
        value = getattr(node, f"scope_{key}", None)
        if value:
            parts.append(f"{key}={value}")
    if node.tier and node.tier != "fact":
        parts.append(f"tier={node.tier}")
    return f"[{', '.join(parts)}] "


def record_stage(name: str, nodes: Iterable[Any], **extra: Any) -> None:
    tr = _TRACE
    if tr is None:
        return
    setattr(tr, name, [node_row(n, **extra) for n in nodes])
