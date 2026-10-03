"""Ability evidence contracts (PRD W7).

Each ability states what a pack must contain before an answer can be trusted:

- ``contradiction`` (CR): both sides of a conflict, each dated; the older marked superseded.
- ``event_ordering`` / ``temporal`` (EO): every event the question names, with real dates.
- ``extraction`` (IE): a verbatim span covering the question's distinctive terms,
  inside the text that actually reaches the answerer (``MAX_EVIDENCE_CHARS``).

``check_contract`` is pure (no model calls). ``retrieve`` runs it after Score-as-sort,
does one widened retry when it fails, and hands any still-unmet contract to
Prove-or-Abstain as an explicit gap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from homebase.brain.mem.evidence import MAX_EVIDENCE_CHARS
from homebase.brain.mem.store import MemNode, MemStore, query_tokens, split_sides, stem
from homebase.brain.mem.temporal import compute_temporal, dated_events

_SPAN_WINDOW = 320


@dataclass
class ContractResult:
    ability: str
    satisfied: bool
    reasons: list[str] = field(default_factory=list)
    widened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ability": self.ability,
            "satisfied": self.satisfied,
            "reasons": list(self.reasons),
            "widened": self.widened,
        }

    def gap_hint(self) -> str:
        return "; ".join(self.reasons)


def _has_date(node: MemNode, dated_ids: set[str]) -> bool:
    return node.id in dated_ids


def _check_cr(evidence: Sequence[MemNode], store: MemStore) -> list[str]:
    reasons: list[str] = []
    pair: tuple[MemNode, MemNode] | None = None
    for i, a in enumerate(evidence):
        for b in evidence[i + 1 :]:
            if store.are_conflicting(a.id, b.id):
                pair = (a, b)
                break
        if pair:
            break
    if pair is None:
        # Superseded body next to a current one on the same topic also counts.
        sup = [n for n in evidence if n.valid_to is not None]
        cur = [n for n in evidence if n.valid_to is None]
        if not (sup and cur):
            return ["cr_missing_second_claim"]
        pair = (sup[0], cur[0])
    dated = {e.node_id for e in dated_events(list(pair), store)}
    if not all(_has_date(n, dated) for n in pair):
        reasons.append("cr_claims_undated")
    older, newer = sorted(pair, key=lambda n: n.valid_from or n.ts or 0.0)
    if older.valid_to is None and newer.valid_to is None and not store.are_conflicting(older.id, newer.id):
        reasons.append("cr_supersession_unmarked")
    return reasons


def _check_eo(question: str, evidence: Sequence[MemNode], store: MemStore) -> list[str]:
    if is_transcript_sequence_question(question):
        required = requested_sequence_length(question) or 2
        ordered_claims = {
            (node.source_position, " ".join(node.content.casefold().split()))
            for node in evidence
            if node.source_position is not None
            and (node.source_role or "").lower() == "user"
        }
        if len(ordered_claims) < required:
            return ["eo_not_enough_ordered_events"]
        return []

    sides = split_sides(question)
    if len(sides) == 2:
        facts = compute_temporal(question, evidence, store).facts
        return [] if facts.get("both_events_dated") else ["eo_named_events_not_both_dated"]
    n_dated = len(dated_events(evidence, store))
    return [] if n_dated >= 2 else ["eo_fewer_than_two_dated_events"]


_SEQUENCE_VERBS = re.compile(r"\b(?:brought\s+up|mentioned|discussed|talked\s+about|asked\s+about)\b", re.I)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COUNTABLE = r"(?:items?|events?|aspects?|steps?|points?|things?)"


def is_transcript_sequence_question(question: str) -> bool:
    """Whether the requested order is mention order in the conversation."""
    return bool(_SEQUENCE_VERBS.search(question or ""))


def requested_sequence_length(question: str) -> int | None:
    """Extract an explicit requested list size without interpreting dates/turn IDs."""
    match = re.search(rf"\b(\d+)\s+{_COUNTABLE}\b", question or "", re.I)
    if match:
        return int(match.group(1))
    for word, count in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+{_COUNTABLE}\b", question or "", re.I):
            return count
    return None


def _densest_hits(text: str, tokens: list[str]) -> int:
    low = text.lower()
    best = 0
    step = _SPAN_WINDOW // 2
    for start in range(0, max(1, len(low) - _SPAN_WINDOW + step), step):
        window = low[start : start + _SPAN_WINDOW]
        best = max(best, sum(1 for t in tokens if stem(t) in window))
    return best


def _check_ie(question: str, evidence: Sequence[MemNode]) -> list[str]:
    tokens = query_tokens(question)[:6]
    if not tokens:
        return []
    need = max(1, math.ceil(0.6 * len(tokens)))
    seen_beyond_cap = False
    for n in evidence:
        text = n.content or ""
        if _densest_hits(text[:MAX_EVIDENCE_CHARS], tokens) >= need:
            return []
        if len(text) > MAX_EVIDENCE_CHARS and _densest_hits(text, tokens) >= need:
            seen_beyond_cap = True
    return ["ie_span_beyond_evidence_cap" if seen_beyond_cap else "ie_span_missing"]


def check_contract(
    ability: str,
    question: str,
    evidence: Sequence[MemNode],
    store: MemStore,
) -> ContractResult:
    if not evidence:
        return ContractResult(ability, False, ["empty_pack"])
    if ability == "contradiction":
        reasons = _check_cr(evidence, store)
    elif ability in ("event_ordering", "temporal"):
        reasons = _check_eo(question, evidence, store)
    elif ability == "extraction":
        reasons = _check_ie(question, evidence)
    else:
        reasons = []
    return ContractResult(ability, not reasons, reasons)
