"""In-memory activity ring for Home Base TUI status indicators."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class ActivityLog:
    """Ring buffer of recent memory / skill-related events (last N)."""

    def __init__(self, max_events: int = 50) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_events)))
        self._lock = threading.Lock()

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": time.time(),
            "kind": kind,
            **{k: v for k, v in fields.items() if v is not None},
        }
        with self._lock:
            self._events.append(event)
        return event

    def since(self, since_ts: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events if float(e.get("ts") or 0) > since_ts]

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._events:
                return None
            return dict(self._events[-1])

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
