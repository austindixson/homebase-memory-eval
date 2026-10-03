"""Jev (or heuristic) ability classification for Lattice retrieve routing."""

from __future__ import annotations

import re
from typing import Any

from homebase.brain.jev_client import SystemOneClient

ABILITIES = (
    "abstention",
    "contradiction",
    "event_ordering",
    "temporal",
    "extraction",
    "preference",
    "default",
)

_ABILITY_CRITERIA = {
    "abstention": "Question asks about something likely absent / should withhold answer",
    "contradiction": "Question about conflicting claims or which statement is correct",
    "event_ordering": "Question about sequence / order / which came first",
    "temporal": "Question about dates, durations, before/after relative time",
    "extraction": "Question asks to recall a specific fact/entity/detail",
    "preference": "Question about user preferences, habits, or personal style",
    "default": "General memory question not matching the above",
}


def heuristic_ability(query: str, hint: str = "") -> str:
    h = (hint or "").strip().lower().replace("-", "_").replace(" ", "_")
    # BEAM / LongMemEval style ability names (check before generic substring match).
    if "contradiction" in h:
        return "contradiction"
    if "event_ordering" in h or "eventordering" in h.replace("_", ""):
        return "event_ordering"
    if "temporal" in h:
        return "temporal"
    if "information_extraction" in h or "info_extraction" in h:
        return "extraction"
    if "abstention" in h:
        return "abstention"
    if "preference" in h:
        return "preference"
    if "multi_session" in h or "knowledge_update" in h or "single_session" in h:
        return "extraction"
    for ab in ABILITIES:
        if ab in h or ab.replace("_", "") in h.replace("_", ""):
            return ab

    q = (query or "").lower()
    if re.search(
        r"contradict|which(?:\s+\w+){0,3}\s+is correct|clarify which|inconsistent|you said",
        q,
    ):
        return "contradiction"
    if re.search(r"\border\b|sequence|which came first|timeline|before or after", q):
        return "event_ordering"
    if re.search(r"\bwhen\b|how many days|what date|before .* after|ago\b", q):
        return "temporal"
    if re.search(r"prefer|favorite|always|never|from now on", q):
        return "preference"
    if re.search(r"what (?:is|was|did)|who |which |name of|tell me about my", q):
        return "extraction"
    return "default"


def classify_ability(
    jev: SystemOneClient | None,
    query: str,
    *,
    hint: str = "",
) -> str:
    """Return ability id. Prefer hint/heuristic; optionally confirm with Jev choice."""
    base = heuristic_ability(query, hint)
    if jev is None or hint:
        return base
    try:
        answers = jev.system_one(
            state=f"user question:\n{query[:800]}",
            questions={
                "ability": {
                    "type": "choice",
                    "instructions": (
                        "Which memory ability does this question primarily require?"
                    ),
                    "criteria": dict(_ABILITY_CRITERIA),
                }
            },
        )
        raw = answers.get("ability") or {}
        choice = str(raw.get("choice") or "").strip()
        if choice in ABILITIES:
            return choice
    except Exception:
        pass
    return base


def needs_historical(ability: str) -> bool:
    return ability in ("contradiction", "temporal", "event_ordering", "knowledge_update")


def top_k_for_ability(ability: str, default_k: int) -> int:
    if ability in ("contradiction", "event_ordering", "temporal"):
        return max(default_k, 16)
    if ability == "abstention":
        return min(default_k, 6)
    return default_k
