"""Mem v2 write path: type → TopK candidates → relate → optional supersede + embed."""

from __future__ import annotations

import re
import time
from typing import Any

from homebase.brain.config import Registry
from homebase.brain.jev_client import SystemOneClient
from homebase.brain.mem.flags import jev_frugal, lattice_enabled
from homebase.brain.mem.observe import observe_text
from homebase.brain.mem.store import MemNode, MemStore
from homebase.brain.mem.timeparse import extract_time


def _keyword_overlap(a: str, b: str) -> float:
    ta = {t.lower() for t in a.split() if len(t) > 2}
    tb = {t.lower() for t in b.split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def write_observation(
    content: str,
    store: MemStore,
    clm: SystemOneClient,
    reg: Registry,
    *,
    tier: str | None = None,
    scope_project: str | None = None,
    scope_person: str | None = None,
    scope_session: str | None = None,
    valid_from: float | None = None,
) -> MemNode:
    """Preserve observation; annotate types; add edges; supersede updates."""
    types = {"episodic": 0.5, "semantic": 0.5, "procedural": 0.0, "preference": 0.0}
    frugal = jev_frugal(reg)
    type_questions: dict[str, Any] = {
        "episodic": {
            "type": "noul",
            "instructions": "Does observation describe a particular experience or event?",
        },
        "semantic": {
            "type": "noul",
            "instructions": "Does observation state a general fact or knowledge?",
        },
        "procedural": {
            "type": "noul",
            "instructions": "Does observation describe a procedure or how-to?",
        },
        "preference": {
            "type": "noul",
            "instructions": "Does observation express a preference or habit?",
        },
    }
    if frugal and tier in ("fact", "episodic", "claim"):
        # Tier is given, so episodic/semantic/procedural only label Nexus pages;
        # preference gates supersession and is the one memory actually consumes.
        type_questions = {"preference": type_questions["preference"]}
    try:
        answers = clm.system_one(
            state=f"observation:\n{content[:4000]}",
            questions=type_questions,
        )
        for key in types:
            types[key] = float((answers.get(key) or {}).get("noul", types[key]))
    except Exception:
        pass

    tier_v = tier
    if tier_v not in ("fact", "episodic", "claim"):
        # Auto: strong episodic → episodic drawer; else fact card.
        tier_v = "episodic" if types.get("episodic", 0) >= 0.75 and len(content) > 600 else "fact"
        if tier is None and len(content) > 600:
            tier_v = "episodic"

    if valid_from is None:
        # Event time from the session/turn date label; node.ts stays mention time.
        pt = extract_time(content)
        if pt.kind == "absolute" and pt.usable:
            valid_from = pt.epoch

    source_prefix = content[:600]
    position_match = re.search(r"\bposition=(\d+)\b", source_prefix)
    role_match = re.search(r"\bturn=[^\s]+\s+(user|assistant)\b", source_prefix, re.I)
    source_position = int(position_match.group(1)) if position_match else None
    source_role = role_match.group(1).lower() if role_match else None

    node = store.add_node(
        content,
        types=types,
        tier=tier_v or "fact",
        scope_project=scope_project,
        scope_person=scope_person,
        scope_session=scope_session,
        valid_from=valid_from,
        source_position=source_position,
        source_role=source_role,
    )

    # Best-effort embedding (Phase 2).
    try:
        from homebase.brain.mem.embed import embed_enabled, embed_and_store

        if embed_enabled(reg):
            embed_and_store(store, node, reg)
    except Exception:
        pass

    # Candidate discovery: prefer current + same scope.
    candidates = [
        c
        for c in store.fts_search(content[:200], limit=reg.write_candidate_k)
        if not (scope_project and c.scope_project and c.scope_project != scope_project)
    ]
    seen = {c.id for c in candidates}
    for other in store.recent(limit=40, current_only=True):
        if other.id == node.id or other.id in seen:
            continue
        if scope_project and other.scope_project and other.scope_project != scope_project:
            continue
        if _keyword_overlap(content, other.content) >= 0.05:
            candidates.append(other)
            seen.add(other.id)
        if len(candidates) >= reg.write_candidate_k:
            break

    # Drop self / invalidated.
    candidates = [c for c in candidates if c.id != node.id and c.valid_to is None]
    candidate_k = reg.write_candidate_k
    if frugal:
        # Relation checks cost 2-3 billed questions per candidate: keep the closest 5.
        candidates.sort(key=lambda c: -_keyword_overlap(content, c.content))
        candidate_k = min(candidate_k, 5)

    # Supersede can only fire for fact/claim nodes or strong preferences; asking the
    # `update` question otherwise is a billed question whose answer is never used.
    can_supersede = node.tier in ("fact", "claim") or types.get("preference", 0) >= 0.5

    if candidates:
        questions: dict[str, Any] = {}
        pair_map: dict[str, MemNode] = {}
        for i, cand in enumerate(candidates[:candidate_k]):
            pair_map[f"p{i}"] = cand
            questions[f"p{i}_semantic"] = {
                "type": "noul",
                "instructions": (
                    "Would a semantic link between these observations help retrieve a shared topic? "
                    f"new={content[:600]!r} cand={cand.content[:600]!r}"
                ),
            }
            questions[f"p{i}_causal"] = {
                "type": "noul",
                "instructions": (
                    "Does the candidate cause/enable/explain the new observation? "
                    f"new={content[:400]!r} cand={cand.content[:400]!r}"
                ),
            }
            if not can_supersede:
                continue
            questions[f"p{i}_update"] = {
                "type": "noul",
                "instructions": (
                    "Does the new observation update, correct, or replace the candidate fact "
                    "(same entity/topic, newer value or preference)? "
                    f"new={content[:500]!r} cand={cand.content[:500]!r}"
                ),
            }

        try:
            answers = clm.system_one(
                state="Relate new memory to candidates.",
                questions=questions,
            )
            now = time.time()
            for prefix, cand in pair_map.items():
                sem = float((answers.get(f"{prefix}_semantic") or {}).get("noul", 0.0))
                cau = float((answers.get(f"{prefix}_causal") or {}).get("noul", 0.0))
                upd = float((answers.get(f"{prefix}_update") or {}).get("noul", 0.0))
                if sem >= reg.relation_threshold:
                    store.add_edge(node.id, cand.id, "semantic", sem)
                    store.add_edge(cand.id, node.id, "semantic", sem)
                if cau >= reg.relation_threshold:
                    store.add_edge(cand.id, node.id, "causal", cau)
                if (
                    can_supersede
                    and upd >= max(reg.relation_threshold, 0.7)
                    and cand.valid_to is None
                ):
                    newer, older = store.mark_superseded(node, cand, upd)
                    store.add_edge(newer.id, older.id, "temporal", upd)
                    # Keep both bodies readable via ConflictPair (Lattice CR).
                    store.add_conflict_pair(
                        newer.id,
                        older.id,
                        topic=content[:80],
                        ts=now,
                    )
        except Exception:
            pass

    # Off-loop observe for episodic drawers (claims/timeline/conflicts).
    # Fact/claim writes already are atomic; only refresh preference profile.
    if (tier_v or "fact") == "episodic":
        try:
            observe_text(
                content,
                store,
                clm,
                reg,
                scope_project=scope_project,
                scope_person=scope_person,
                scope_session=scope_session,
                source_node=node,
            )
        except Exception:
            pass
    elif lattice_enabled(reg):
        low = content.lower()
        if any(w in low for w in ("prefer", "always", "never", "from now on", "i like")):
            key = scope_person or scope_project or scope_session or "default"
            profile_key = f"person:{key}" if scope_person else f"scope:{key}"
            try:
                prev = store.get_profile(profile_key) or ""
                merged = (prev + "\n" + content[:400]).strip()
                store.upsert_profile(profile_key, merged[-800:])
            except Exception:
                pass

    return node
