"""Search results -> answer-prompt memory text.

AML keeps "a token-counted prefix of Search candidates in returned rank order" within
117,760 input tokens. Its exact memory rendering is unpublished; this uses the one
published form (CL-Bench's ``- [created_at] text``) for every dataset.
"""
from __future__ import annotations

from typing import Any, Callable

INPUT_TOKEN_BUDGET = 128_000 - 8_192 - 2_048


def render_memories(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        stamp = str(item.get("created_at") or "").strip()
        lines.append(f"- [{stamp}] {text}" if stamp else f"- {text}")
    return "\n".join(lines)


def fit_prefix(items: list[dict[str, Any]], prompt: Callable[[str], str],
               count: Callable[[str], int], budget: int = INPUT_TOKEN_BUDGET) -> tuple[int, str]:
    """Longest rank-order prefix whose full prompt fits ``budget`` tokens."""
    low, high = 0, len(items)
    while low < high:
        middle = (low + high + 1) // 2
        if count(prompt(render_memories(items[:middle]))) <= budget:
            low = middle
        else:
            high = middle - 1
    return low, render_memories(items[:low])
