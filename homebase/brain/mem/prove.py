"""Prove-or-Abstain: pack-level Jev decision after Score-as-sort."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from homebase.brain.jev_client import SystemOneClient
from homebase.brain.mem.evidence import PROVE_PACK_CHARS, clip, current_trace, evidence_prefix
from homebase.brain.mem.store import MemNode


@dataclass
class ProveResult:
    proves: bool
    score: float
    mode: str  # prove | abstain | skipped
    reason: str = ""  # no_evidence | below_floor | abstention_hint | evidence_gap | error_open
    gap: str = ""  # unmet evidence-contract reasons handed over by retrieve


def enforcement_enabled() -> bool:
    """Prove-or-Abstain withholds an abstaining pack from the answerer.

    Off by default so general chat recall on the prod brain is unchanged; the bench
    launcher sets ``HOMEBASE_MEM_PROVE_ENFORCE=1`` on the isolated bench brain.
    """
    return (os.environ.get("HOMEBASE_MEM_PROVE_ENFORCE") or "0").strip() == "1"


def gates_answer(result: ProveResult | None) -> bool:
    """True when the pack must be withheld from the answerer."""
    return bool(
        result is not None
        and enforcement_enabled()
        and result.mode == "abstain"
        and not result.proves
    )


def build_prove_pack(nodes: list[MemNode], *, budget: int | None = None) -> str:
    """Concatenate whole passages (<= MAX_EVIDENCE_CHARS each) within ``budget`` chars.

    Budgeting is by passage count: once the next passage no longer fits it (and any
    after it) is dropped and logged, instead of silently cutting the pack tail.
    """
    limit = max(0, PROVE_PACK_CHARS if budget is None else int(budget))
    parts: list[str] = []
    used = 0
    tr = current_trace()
    included: list[str] = []
    dropped_ids: list[str] = []
    for i, n in enumerate(nodes):
        text = evidence_prefix(n) + clip(n.content or "", stage="prove", node_id=n.id)
        separator_chars = 5 if parts else 0
        if used + len(text) + separator_chars > limit:
            dropped_ids = [dropped.id for dropped in nodes[i:]]
            if tr is not None:
                for dropped in nodes[i:]:
                    tr.record_truncation("prove_budget", dropped.id, len(dropped.content or ""), limit)
            break
        parts.append(text)
        included.append(n.id)
        used += len(text) + separator_chars
    if tr is not None:
        tr.routes["prove_pack"] = {
            "included_ids": included, "dropped_ids": dropped_ids,
            "chars": used, "budget": limit,
        }
    return "\n---\n".join(parts)


def prove_or_abstain(
    jev: SystemOneClient,
    query: str,
    nodes: list[MemNode],
    *,
    floor: float = 4.0,
    ability: str = "default",
    gaps: str = "",
) -> ProveResult:
    """Score whether the whole pack proves an answer (0-10).

    ``gaps`` lists unmet evidence-contract reasons (e.g. only one of two events found);
    they are shown to the scorer and raise the floor by one point.
    """
    if not nodes:
        return ProveResult(proves=False, score=0.0, mode="abstain", reason="no_evidence", gap=gaps)
    if ability == "abstention" and (
        os.environ.get("HOMEBASE_MEM_PROVE_TRUST_ABSTENTION_HINT") or "0"
    ).strip() == "1":
        # Legacy shortcut: trust an "abstention" label. Off by default — a benchmark
        # label is not evidence, and it made abstention accuracy unprovable.
        return ProveResult(proves=False, score=0.0, mode="abstain", reason="abstention_hint", gap=gaps)

    pack = build_prove_pack(nodes)
    if not pack:
        return ProveResult(proves=False, score=0.0, mode="abstain", reason="no_evidence", gap=gaps)
    try:
        answers = jev.system_one(
            state=(
                f"question:\n{query[:500]}\n\nevidence pack:\n{pack}"
                + (f"\n\nKnown evidence gaps: {gaps}" if gaps else "")
            ),
            questions={
                "prove": {
                    "type": "score",
                    "instructions": (
                        "Rate 0-10: does this evidence pack prove a specific answer "
                        "to the question (not merely related topic noise)?"
                    ),
                    # Jev allows ≤10 levels; indices 0-9 map to prove strength.
                    "criteria": [
                        "0 No evidence",
                        "1 Related noise only",
                        "2 Weak / incomplete",
                        "3 Partial but thin",
                        "4 Borderline",
                        "5 Partial but usable",
                        "6 Mostly enough",
                        "7 Strong enough to answer",
                        "8 Very strong",
                        "9 Definitive",
                    ],
                }
            },
        )
        raw = answers.get("prove") or {}
        # Jev score is expected criteria index in [0, 9].
        score = min(9.0, max(0.0, float(raw.get("score", 0.0))))
    except Exception:
        # Fail open: allow answer path if we have any evidence.
        return ProveResult(proves=True, score=5.0, mode="skipped", reason="error_open", gap=gaps)

    eff_floor = float(floor) + (1.0 if gaps else 0.0)
    proves = score >= eff_floor
    return ProveResult(
        proves=proves,
        score=score,
        mode="prove" if proves else "abstain",
        reason="" if proves else ("evidence_gap" if gaps else "below_floor"),
        gap=gaps,
    )


ABSTAIN_DIRECTIVE = (
    "Pack decision: ABSTAIN — memory holds no evidence that answers this question. "
    "Reply that you don't know / that the information is not in memory. Do not guess."
)


def format_prove_hint(result: ProveResult, ability: str) -> str:
    if gates_answer(result):
        return ABSTAIN_DIRECTIVE
    if result.mode == "abstain" or not result.proves:
        if ability == "abstention" or result.mode == "abstain":
            return (
                "Pack decision: ABSTAIN — if evidence is insufficient, say you don't know "
                "or that the information is not in memory."
            )
    if ability == "contradiction":
        return (
            "Pack decision: PROVE — list conflicting claims from evidence and resolve "
            "which appears current."
        )
    if ability == "event_ordering":
        return (
            "Pack decision: PROVE — use the order metadata in each source. For questions about "
            "what the user brought up, use only user turns and order by transcript position, "
            "not source IDs or calendar dates. For ordered-list answers, include exactly one "
            "event per line in the requested sequence; do not combine events on a line and add "
            "no preamble or explanation."
        )
    return "Pack decision: PROVE — answer from the evidence; do not abstain without cause."
