"""Off-loop Lattice observer: claims + timeline + conflicts + profile (never deletes episodic)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from homebase.brain.config import Registry
from homebase.brain.jev_client import SystemOneClient
from homebase.brain.mem.flags import jev_frugal, lattice_enabled
from homebase.brain.mem.store import MemNode, MemStore
from homebase.brain.mem.timeparse import ParsedTime, extract_time

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class ObserveResult:
    claim_ids: list[str]
    timeline_ids: list[str]
    conflict_ids: list[str]
    profile_key: str | None
    time_kind: str = "none"  # absolute | relative | ordinal | malformed | none
    unparsed_time_label: str | None = None


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


_PREFILTER_BATCH = 10
_KEEP_INSTR = (
    "Should this sentence go to a memory fact extractor? "
    "Yes for lasting facts, preferences, events, entities. "
    "No for filler, greetings, or empty acknowledgements."
)


def prefilter_for_observer(
    jev: SystemOneClient,
    text: str,
    *,
    max_keep: int = 12,
) -> str:
    """Jev Noul keep for observer input only — never deletes stored episodic text.

    Sentences are scored ``_PREFILTER_BATCH`` per System One call (one question per
    sentence, same instruction as before) instead of one HTTP round-trip each.
    """
    sents = _sentences(text)[:40]
    if len(sents) <= 3:
        return text
    # Only sentences the claim extractor could ever select are worth a billed
    # question; the rest can never become claims whatever Jev answers.
    sents = [s for s in sents if _claim_candidate(s)]
    if not sents:
        return text
    kept: list[str] = []
    try:
        for i in range(0, len(sents), _PREFILTER_BATCH):
            batch = sents[i : i + _PREFILTER_BATCH]
            answers = jev.system_one(
                state="Decide for each numbered sentence whether it should be kept.",
                questions={
                    f"keep{j}": {
                        "type": "noul",
                        "instructions": f"{_KEEP_INSTR}\nSentence: {s[:800]}",
                    }
                    for j, s in enumerate(batch)
                },
            )
            for j, s in enumerate(batch):
                if float((answers.get(f"keep{j}") or {}).get("noul", 0.0)) >= 0.45:
                    kept.append(s)
            if len(kept) >= max_keep:
                kept = kept[:max_keep]
                break
    except Exception:
        return text
    return "\n".join(kept) if kept else text


_CLAIM_TRIGGERS = (
    "i ",
    "my ",
    "prefer",
    "always",
    "never",
    "on ",
    "was ",
    "were ",
    "have ",
    "use ",
    "using ",
)


def _claim_candidate(sentence: str) -> bool:
    """Could ``_extract_claims_heuristic`` ever select this sentence?"""
    low = sentence.lower()
    return len(sentence) >= 20 and any(w in low for w in _CLAIM_TRIGGERS)


def _extract_claims_heuristic(text: str, *, max_claims: int = 6) -> list[str]:
    """Cheap claim split when Jev extract is unavailable."""
    out: list[str] = []
    for s in _sentences(text):
        if _claim_candidate(s):
            out.append(s[:580])
        if len(out) >= max_claims:
            break
    return out


def _extract_claims_jev(jev: SystemOneClient, text: str) -> list[str]:
    try:
        answers = jev.system_one(
            state=text[:3500],
            questions={
                # Only has_facts is consumed; is_preference / is_event were asked
                # and discarded (2 billed questions per write).
                "has_facts": {
                    "type": "noul",
                    "instructions": "Does this text contain lasting factual claims worth storing?",
                },
            },
        )
        if float((answers.get("has_facts") or {}).get("noul", 0)) < 0.35:
            return []
    except Exception:
        return _extract_claims_heuristic(text)
    # One claim card per high-signal sentence after prefilter.
    return _extract_claims_heuristic(text, max_claims=8)


def _parse_time_label(text: str, *, anchor_epoch: float | None = None) -> ParsedTime:
    """Real time for ``text``: explicit ``date=``/``@`` label, absolute date, or a
    relative phrase anchored at ``anchor_epoch``. Never hashes a label into a number;
    an unparseable label returns ``epoch=None`` with ``kind="malformed"``.
    """
    return extract_time(text, anchor_epoch=anchor_epoch)


def observe_text(
    text: str,
    store: MemStore,
    jev: SystemOneClient,
    reg: Registry,
    *,
    scope_project: str | None = None,
    scope_person: str | None = None,
    scope_session: str | None = None,
    source_node: MemNode | None = None,
) -> ObserveResult:
    """Extract claims/timeline/conflicts/profile alongside verbatim episodic."""
    if not lattice_enabled(reg):
        return ObserveResult([], [], [], None, time_kind="disabled")
    if any(_claim_candidate(sent) for sent in _sentences(text)):
        filtered = prefilter_for_observer(jev, text)
        claims_text = _extract_claims_jev(jev, filtered)
    else:
        claims_text = []  # nothing could become a claim: skip the billed questions
    claim_ids: list[str] = []
    timeline_ids: list[str] = []
    conflict_ids: list[str] = []
    profile_key: str | None = None

    chunk_time = _parse_time_label(text)
    anchor = chunk_time.epoch if chunk_time.kind == "absolute" else None
    entity = None
    em = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    if em:
        entity = em.group(1)

    for claim in claims_text:
        # Event time: a relative phrase inside the claim ("two weeks ago") resolved
        # against the session date beats the session date itself (mention time).
        claim_time = chunk_time
        if anchor is not None:
            rel = extract_time(claim, anchor_epoch=anchor)
            if rel.kind == "relative" and rel.usable:
                claim_time = rel
        real_time = claim_time.kind in ("absolute", "relative") and claim_time.usable
        node = store.add_node(
            claim,
            types={"semantic": 0.7, "episodic": 0.3},
            tier="claim",
            scope_project=scope_project,
            scope_person=scope_person,
            scope_session=scope_session,
            valid_from=claim_time.epoch if real_time else None,
            source_position=(source_node.source_position if source_node else None),
            source_role=(source_node.source_role if source_node else None),
        )
        claim_ids.append(node.id)
        if source_node is not None:
            try:
                store.add_edge(node.id, source_node.id, "temporal", 0.5)
            except Exception:
                pass
        if claim_time.usable:
            t_sort = claim_time.epoch  # type: ignore[assignment]
        elif source_node is not None and source_node.source_position is not None:
            # Transcript position is a separate coordinate from event/calendar time.
            t_sort = float(source_node.source_position)
            claim_time = ParsedTime(t_sort, f"position={source_node.source_position}", "ordinal")
        elif claim_time.kind == "none":
            t_sort = time.time()  # no label at all: mention time
        else:
            t_sort = None  # malformed label: skip rather than invent an order
        if t_sort is not None:
            tid = store.add_timeline_event(
                node.id,
                t_sort=float(t_sort),
                t_label=claim_time.label,
                entity=entity,
                scope_session=scope_session,
                t_kind=claim_time.kind,
            )
            timeline_ids.append(tid)

        # Supersede / conflict against current same-scope claims.
        cands = store.fts_search(claim[:120], limit=6, current_only=True)
        pairs: list[tuple[MemNode, float]] = []
        for cand in cands:
            if cand.id == node.id or cand.tier not in ("fact", "claim"):
                continue
            if scope_session and cand.scope_session and cand.scope_session != scope_session:
                continue
            overlap = _token_overlap(claim, cand.content)
            if overlap < 0.25:
                continue
            pairs.append((cand, overlap))
        if not pairs:
            continue
        if jev_frugal(reg):
            # 2 billed questions per pair: check only the 3 closest old claims.
            pairs = sorted(pairs, key=lambda p: -p[1])[:3]
        # One System One call for every candidate pair of this claim (was one call each).
        scores: dict[str, tuple[float, float]] = {}
        try:
            questions: dict[str, Any] = {}
            for i, (cand, _ov) in enumerate(pairs):
                pair_txt = f"new:\n{claim[:500]}\nold:\n{cand.content[:500]}"
                questions[f"update{i}"] = {
                    "type": "noul",
                    "instructions": (
                        "Does the new claim update/replace the old claim "
                        f"about the same entity/topic?\n{pair_txt}"
                    ),
                }
                questions[f"conflict{i}"] = {
                    "type": "noul",
                    "instructions": f"Do the two claims contradict each other?\n{pair_txt}",
                }
            ans = jev.system_one(
                state="Compare the new claim with each old claim.", questions=questions
            )
            for i, (cand, _ov) in enumerate(pairs):
                scores[cand.id] = (
                    float((ans.get(f"update{i}") or {}).get("noul", 0)),
                    float((ans.get(f"conflict{i}") or {}).get("noul", 0)),
                )
        except Exception:
            for cand, overlap in pairs:
                scores[cand.id] = (0.0, 0.6 if overlap >= 0.4 else 0.0)
        for cand, _overlap in pairs:
            upd, conf = scores[cand.id]
            if upd >= 0.7:
                newer, older = store.mark_superseded(node, cand, upd)
                pid = store.add_conflict_pair(
                    newer.id, older.id, topic=(entity or claim[:80])
                )
                conflict_ids.append(pid)
            elif conf >= 0.6:
                pid = store.add_conflict_pair(
                    node.id, cand.id, topic=(entity or claim[:80])
                )
                conflict_ids.append(pid)

    # Profile scratchpad for preference-ish text.
    low = text.lower()
    if any(w in low for w in ("prefer", "always", "never", "from now on", "i like")):
        key = scope_person or scope_project or scope_session or "default"
        profile_key = f"person:{key}" if scope_person else f"scope:{key}"
        prev = store.get_profile(profile_key) or ""
        merged = (prev + "\n" + text[:400]).strip()
        # Keep last ~800 chars.
        store.upsert_profile(profile_key, merged[-800:])

    return ObserveResult(
        claim_ids=claim_ids,
        timeline_ids=timeline_ids,
        conflict_ids=conflict_ids,
        profile_key=profile_key,
        time_kind=chunk_time.kind,
        unparsed_time_label=chunk_time.label if chunk_time.kind == "malformed" else None,
    )


def _token_overlap(a: str, b: str) -> float:
    ta = {t.lower() for t in a.split() if len(t) > 2}
    tb = {t.lower() for t in b.split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def continuation_chunk(
    jev: SystemOneClient | None,
    text: str,
    *,
    target_chars: int = 1600,
) -> list[str]:
    """Jev continuation chunking near target size; falls back to char splits."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    sents = _sentences(text)
    if not sents or jev is None:
        return _char_chunks(text, target_chars)

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    try:
        for i, s in enumerate(sents):
            cont = True
            if buf and buf_len >= int(target_chars * 0.6):
                ans = jev.system_one(
                    state=f"prev:\n{buf[-1][:400]}\nnext:\n{s[:400]}",
                    questions={
                        "continues": {
                            "type": "noul",
                            "instructions": (
                                "Does the next sentence continue the previous thought "
                                "(vs start a new topic)?"
                            ),
                        }
                    },
                )
                cont = float((ans.get("continues") or {}).get("noul", 0.5)) >= 0.45
            if not cont and buf and buf_len >= int(target_chars * 0.5):
                chunks.append(" ".join(buf))
                buf, buf_len = [], 0
            buf.append(s)
            buf_len += len(s) + 1
            if buf_len >= target_chars:
                chunks.append(" ".join(buf))
                buf, buf_len = [], 0
            _ = i
        if buf:
            chunks.append(" ".join(buf))
        return chunks or [text]
    except Exception:
        return _char_chunks(text, target_chars)


def _char_chunks(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + max_chars // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end
    return out
