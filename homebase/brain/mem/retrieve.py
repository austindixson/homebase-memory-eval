"""Lattice Mem v3 retrieve: ability-route → fusion shortlist → Score-as-sort → prove."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from homebase.brain.config import Registry
from homebase.brain.jev_client import SystemOneClient
from homebase.brain.mem.ability import (
    classify_ability,
    needs_historical,
    top_k_for_ability,
)
from homebase.brain.mem.contracts import (
    ContractResult,
    check_contract,
    is_transcript_sequence_question,
)
from homebase.brain.mem.embed import embed_enabled, search_similar
from homebase.brain.mem.evidence import (
    begin_trace,
    clip,
    current_trace,
    evidence_prefix,
    last_trace,
    record_stage,
)
from homebase.brain.mem.filter import FilterStats, filter_by_usefulness
from homebase.brain.mem.flags import lattice_enabled
from homebase.brain.mem.lexical import select_bm25
from homebase.brain.mem.prove import (
    ProveResult,
    format_prove_hint,
    gates_answer,
    prove_or_abstain,
)
from homebase.brain.mem.store import MemNode, MemStore, query_tokens, split_sides, spread_by_position
from homebase.brain.mem.temporal import compute_temporal, select_pair_evidence

if TYPE_CHECKING:
    from homebase.brain.activity import ActivityLog

_LAST_STATS: FilterStats | None = None
_LAST_PROVE: ProveResult | None = None
_LAST_ABILITY: str = "default"
_LAST_PROFILES: list[str] = []
_LAST_CONTRACT: ContractResult | None = None
_LAST_COMPUTED: list[str] = []
CONTRACT_ABILITIES = ("contradiction", "event_ordering", "temporal", "extraction")


def contract_retry_enabled() -> bool:
    return (os.environ.get("HOMEBASE_MEM_CONTRACT_RETRY") or "1").strip() != "0"


def last_filter_stats() -> FilterStats | None:
    return _LAST_STATS


def last_prove_result() -> ProveResult | None:
    return _LAST_PROVE


def last_ability() -> str:
    return _LAST_ABILITY


def last_profiles() -> list[str]:
    return list(_LAST_PROFILES)


def last_contract() -> ContractResult | None:
    return _LAST_CONTRACT


def last_computed() -> list[str]:
    return list(_LAST_COMPUTED)


def last_stage_trace() -> dict[str, Any] | None:
    """Per-stage evidence trace of the last ``retrieve`` (+ pack) call."""
    return last_trace()


def _merge_unique(*groups: list[MemNode]) -> list[MemNode]:
    out: list[MemNode] = []
    seen: set[str] = set()
    for group in groups:
        for n in group:
            if n.id in seen:
                continue
            seen.add(n.id)
            out.append(n)
    return out


def _current_only(nodes: list[MemNode], *, historical: bool) -> list[MemNode]:
    if historical:
        return nodes
    return [n for n in nodes if n.valid_to is None]


def _scope_boost(
    nodes: list[MemNode],
    *,
    scope_project: str | None,
    scope_person: str | None,
    scope_session: str | None,
) -> list[MemNode]:
    if not (scope_project or scope_person or scope_session):
        return nodes

    # Prefer same session/project, then fill from outside (don't hard-drop).
    def key(n: MemNode) -> tuple[int, float]:
        hits = 0
        if scope_session and n.scope_session == scope_session:
            hits += 5
        if scope_project and n.scope_project == scope_project:
            hits += 3
        if scope_person and n.scope_person == scope_person:
            hits += 2
        # Newest event first; write order only when no event date exists (valid_from == ts).
        return (-hits, -(n.valid_from if n.valid_from is not None else n.ts))

    return sorted(nodes, key=key)


def _fusion_candidates(
    query: str,
    store: MemStore,
    reg: Registry,
    *,
    cand_k: int,
    embed_candidates: list[MemNode] | None,
) -> list[MemNode]:
    """Multi-signal first stage: FTS ∪ BM25-over-recent ∪ embed."""
    fts = store.fts_search(query, limit=cand_k)
    tokens = [t for t in query.replace('"', " ").split() if len(t) > 3][:8]
    if tokens:
        fts = _merge_unique(fts, store.fts_search(" ".join(tokens), limit=max(8, cand_k // 2)))

    recent = store.recent(limit=min(80, cand_k * 2))
    bm25 = select_bm25(query, recent, max_results=min(24, cand_k))

    emb = embed_candidates or []
    if not emb:
        try:
            if embed_enabled(reg):
                emb = search_similar(store, query, reg, limit=cand_k)
        except Exception:
            emb = []

    # Light entity boost: capitalized tokens + quoted phrases as FTS.
    ents = [t for t in query.split() if t[:1].isupper() and len(t) > 2][:4]
    ent_hits: list[MemNode] = []
    for e in ents:
        ent_hits.extend(store.fts_search(e, limit=6))
    for phrase in re.findall(r"'([^']{3,80})'|\"([^\"]{3,80})\"", query):
        p = (phrase[0] or phrase[1]).strip()
        if p:
            ent_hits.extend(store.fts_search(p, limit=6))
    # "which A or B" / "first, the X or the Y" — pull both sides.
    or_parts = re.split(r"\bor\b", query, flags=re.I)
    if len(or_parts) == 2:
        for side in or_parts:
            side_toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", side) if len(t) > 3][-4:]
            if side_toks:
                ent_hits.extend(store.fts_search(" ".join(side_toks), limit=6))

    merged = _merge_unique(fts, bm25, emb, ent_hits)
    if not merged:
        merged = recent[:cand_k]
    return merged[:cand_k]


def _gather(
    q: str,
    ability: str,
    store: MemStore,
    clm: SystemOneClient,
    reg: Registry,
    *,
    k: int,
    cand_k: int,
    hist: bool,
    scope_project: str | None,
    scope_person: str | None,
    scope_session: str | None,
    embed_candidates: list[MemNode] | None,
    widen: bool,
    trace: Any,
    score_cache: dict[str, float],
) -> tuple[list[MemNode], FilterStats]:
    """Route -> fusion shortlist -> Score-as-sort -> one-hop expansion."""
    transcript_order = ability == "event_ordering" and is_transcript_sequence_question(q)
    hops_budget = min(int(getattr(reg, "max_hops", 1) or 1), 1)
    routed: list[MemNode] = []
    if ability == "contradiction":
        for a, b, _topic in store.search_conflicts(q, limit=8 if not widen else 16):
            routed.extend([a, b])
            routed.extend(store.supersede_neighbors(a.id, limit=2))
            routed.extend(store.supersede_neighbors(b.id, limit=2))
        hist = True
    elif ability in ("event_ordering", "temporal"):
        tl = store.search_timeline_ex(
            q,
            limit=cand_k,
            scope_session=scope_session,
            transcript_order=transcript_order,
            speaker_role="user" if transcript_order else None,
        )
        routed.extend(tl.nodes)
        # Dead-route visibility (PRD W4): log zero-hit and A-vs-B coverage.
        key = "timeline_widened" if widen else "timeline"
        trace.routes[key] = {
            "tokens": tl.tokens,
            "sides": tl.sides,
            "side_hits": tl.side_hits,
            "n_hits": len(tl.nodes),
            "zero_hit": tl.zero_hit,
            "both_events_present": tl.both_events_present,
        }
        hist = True
    elif ability in ("extraction", "preference"):
        # Prefer claim/fact tiers + profile injection via format later.
        hybrid = _fusion_candidates(q, store, reg, cand_k=cand_k, embed_candidates=embed_candidates)
        routed.extend([n for n in hybrid if n.tier in ("claim", "fact")] or hybrid)
    else:
        routed.extend(
            _fusion_candidates(q, store, reg, cand_k=cand_k, embed_candidates=embed_candidates)
        )

    # Always union with hybrid shortlist so routing can't starve.
    hybrid = _fusion_candidates(q, store, reg, cand_k=cand_k, embed_candidates=embed_candidates)
    extra: list[MemNode] = []
    if widen:
        # Widened retry: search each named side / the distinctive terms on their own.
        subqueries = [" ".join(side) for side in split_sides(q)]
        subqueries.append(" ".join(query_tokens(q)[:6]))
        for sub in dict.fromkeys(x for x in subqueries if x.strip()):
            extra.extend(
                _fusion_candidates(sub, store, reg, cand_k=cand_k, embed_candidates=None)
            )
    candidates = _merge_unique(routed, hybrid, extra)
    if transcript_order:
        candidates = [
            n for n in candidates
            if n.source_position is not None and (n.source_role or "").lower() == "user"
        ]
    candidates = _current_only(candidates, historical=hist)
    candidates = _scope_boost(
        candidates,
        scope_project=scope_project,
        scope_person=scope_person,
        scope_session=scope_session,
    )
    pair_sources: list[MemNode] = []
    if lattice_enabled(reg) and ability in ("temporal", "event_ordering") and not transcript_order and k >= 2:
        scoped = [n for n in candidates if n.scope_session == scope_session] if scope_session else []
        pair_sources = select_pair_evidence(q, scoped or candidates, store)
        candidates = _merge_unique(pair_sources, candidates)
    candidates = candidates[:cand_k]
    if not widen:
        record_stage("retrieved", candidates)

    kept, stats = filter_by_usefulness(
        clm,
        q,
        candidates,
        # Mention-order questions rank every candidate, then spread the pick across
        # the transcript below; a plain top-k would cluster in one stretch again.
        max_keep=len(candidates) if transcript_order else k,
        concurrency=int(getattr(reg, "mem_jev_concurrency", 4) or 4),
        score_cache=score_cache,
        keyword_relative_threshold=float(
            getattr(reg, "mem_keyword_relative_threshold", 0.5) or 0.5
        ),
    )

    if not kept and candidates:
        kept = list(candidates)[:k]
        stats = FilterStats(
            mode="fts_fallback",
            kept=len(kept),
            dropped=max(0, len(candidates) - len(kept)),
            candidate_n=len(candidates),
        )

    if transcript_order:
        kept = spread_by_position(
            [n for n in kept if (n.source_role or "").lower() == "user"], k
        )
    elif pair_sources:
        kept = _merge_unique(pair_sources, kept)[:k]
        trace.routes["pair_sources_widened" if widen else "pair_sources"] = [n.id for n in pair_sources]
    seen = {n.id for n in kept}
    evidence = list(kept)

    if hops_budget > 0 and evidence:
        frontier: list[MemNode] = []
        for n in evidence:
            for neigh, kind, _score in store.neighbors(n.id, limit=5):
                if neigh.id in seen:
                    continue
                if not hist and neigh.valid_to is not None and kind != "supersedes":
                    continue
                frontier.append(neigh)
                seen.add(neigh.id)
        if frontier:
            extra_nodes, hop_stats = filter_by_usefulness(
                clm,
                q,
                frontier[: min(12, cand_k)],
                max_keep=max(0, k - len(evidence)),
                concurrency=int(getattr(reg, "mem_jev_concurrency", 4) or 4),
                score_cache=score_cache,
            )
            evidence.extend(extra_nodes)
            stats = FilterStats(
                mode=stats.mode,
                kept=len(evidence),
                dropped=stats.dropped + hop_stats.dropped,
                candidate_n=stats.candidate_n + hop_stats.candidate_n,
                scores=list(stats.scores) + list(hop_stats.scores),
            )
    if transcript_order:
        # Retrieval/filtering chooses relevant turns; presentation follows their
        # actual order in the flattened transcript, never source-ID or date order.
        evidence = [
            n for n in evidence
            if n.source_position is not None and (n.source_role or "").lower() == "user"
        ]
        evidence.sort(key=lambda n: (n.source_position, n.ts))
    return evidence[:k], stats


def retrieve(
    query: str,
    store: MemStore,
    clm: SystemOneClient,
    reg: Registry,
    top_k: int | None = None,
    *,
    activity: ActivityLog | None = None,
    scope_project: str | None = None,
    scope_person: str | None = None,
    scope_session: str | None = None,
    historical: bool | None = None,
    embed_candidates: list[MemNode] | None = None,
    ability_hint: str = "",
    run_prove: bool = True,
    as_of: float | None = None,
) -> list[MemNode]:
    """Ability-routed Lattice retrieve: Score-as-sort -> evidence contract -> prove."""
    global _LAST_STATS, _LAST_PROVE, _LAST_ABILITY, _LAST_PROFILES
    global _LAST_CONTRACT, _LAST_COMPUTED
    trace = begin_trace()
    _LAST_CONTRACT = None
    _LAST_COMPUTED = []
    ability = classify_ability(clm, query, hint=ability_hint)
    _LAST_ABILITY = ability
    _LAST_PROFILES = []
    k = top_k_for_ability(ability, top_k or reg.mem_top_k)
    cand_k = getattr(reg, "mem_candidate_k", 48) or 48
    hist = needs_historical(ability) if historical is None else bool(historical)
    lattice = lattice_enabled(reg)
    if not lattice:
        # Ablation: no ability routes / contracts / temporal helper / profiles.
        ability = "default"
        _LAST_ABILITY = ability
        hist = False if historical is None else bool(historical)
    trace.routes["lattice"] = lattice

    q = (query or "").strip()
    if not q:
        _LAST_STATS = FilterStats(mode="empty", kept=0, dropped=0, candidate_n=0)
        _LAST_PROVE = ProveResult(proves=False, score=0.0, mode="abstain", reason="no_evidence")
        return []

    common: dict[str, Any] = {
        "scope_project": scope_project,
        "scope_person": scope_person,
        "scope_session": scope_session,
        "embed_candidates": embed_candidates,
        "trace": trace,
        "score_cache": {},  # one question = one score per passage per retrieve call
    }
    evidence, stats = _gather(
        q, ability, store, clm, reg, k=k, cand_k=cand_k, hist=hist, widen=False, **common
    )

    # Evidence contract: one widened retry, then hand any gap to Prove-or-Abstain.
    contract = check_contract(ability, q, evidence, store)
    if (
        lattice
        and not contract.satisfied
        and ability in CONTRACT_ABILITIES
        and contract_retry_enabled()
    ):
        first_reasons = list(contract.reasons)
        wide_k = min(k + 6, 24)
        wide, wide_stats = _gather(
            q, ability, store, clm, reg, k=wide_k, cand_k=cand_k * 2, hist=True, widen=True, **common
        )
        merged = _merge_unique(evidence, wide)[:wide_k]
        retry = check_contract(ability, q, merged, store)
        retry.widened = True
        trace.routes["contract_first_pass"] = first_reasons
        if retry.satisfied or len(retry.reasons) < len(first_reasons):
            evidence, stats, contract = merged, wide_stats, retry
        else:
            contract = retry
    _LAST_CONTRACT = contract
    trace.routes["contract"] = contract.to_dict()

    record_stage("kept", evidence)
    _LAST_STATS = stats

    transcript_order = ability == "event_ordering" and is_transcript_sequence_question(q)
    if lattice and ability in ("event_ordering", "temporal", "default", "extraction") and not transcript_order:
        try:
            tres = compute_temporal(q, evidence, store, as_of=as_of)
            _LAST_COMPUTED = list(tres.lines)
            trace.routes["temporal"] = tres.facts
        except Exception:
            _LAST_COMPUTED = []

    if lattice and ability in ("preference", "extraction", "default"):
        try:
            _LAST_PROFILES = store.find_profile_for_query(q, limit=2)
            if scope_session:
                p = store.get_profile(f"session:{scope_session}")
                if p and p not in _LAST_PROFILES:
                    _LAST_PROFILES.insert(0, p)
        except Exception:
            _LAST_PROFILES = []

    if run_prove:
        floor = float(getattr(reg, "mem_prove_floor", 4.0) or 4.0)
        gaps = "" if contract.satisfied else contract.gap_hint()
        _LAST_PROVE = prove_or_abstain(
            clm, q, evidence, floor=floor, ability=ability, gaps=gaps
        )
        if transcript_order and not contract.satisfied:
            # An LLM score cannot replace missing source positions or user turns.
            _LAST_PROVE = ProveResult(
                proves=False,
                score=_LAST_PROVE.score,
                mode="abstain",
                reason="evidence_gap",
                gap=contract.gap_hint(),
            )
        trace.routes["prove"] = {
            "mode": _LAST_PROVE.mode,
            "score": _LAST_PROVE.score,
            "reason": _LAST_PROVE.reason,
            "gap": _LAST_PROVE.gap,
        }
    else:
        _LAST_PROVE = ProveResult(proves=True, score=5.0, mode="skipped")

    if activity is not None:
        try:
            activity.record(
                "mem_filtered",
                kept=stats.kept,
                dropped=stats.dropped,
                mode=stats.mode,
                candidate_n=stats.candidate_n,
                ability=ability,
                prove=(_LAST_PROVE.mode if _LAST_PROVE else None),
                prove_score=(_LAST_PROVE.score if _LAST_PROVE else None),
                contract_ok=contract.satisfied,
            )
        except Exception:
            pass
    return evidence


_UNSET: Any = object()


def format_memory_block(
    nodes: list[MemNode],
    *,
    query: str = "",
    ability: str = "",
    prove: Any = _UNSET,
    profiles: list[str] | None = None,
    computed: Any = _UNSET,
) -> str:
    ab = ability or _infer_answer_mode(query)
    prove_res: ProveResult | None
    if prove is _UNSET:
        prove_res = _LAST_PROVE
    else:
        prove_res = prove  # may be None to skip pack decision
    if not nodes and not profiles and not gates_answer(prove_res):
        return ""
    header = "[PIN] Home Base Lattice memory evidence:"
    if prove_res is not None:
        header += "\n" + format_prove_hint(prove_res, ab)
    elif ab == "contradiction":
        header += (
            "\n(If evidence conflicts, list both claims and resolve which is current.)"
        )
    elif ab == "event_ordering":
        header += "\n(Prefer dated events; answer as an ordered timeline.)"
    lines = [header]
    tr = current_trace()
    # Prove-or-Abstain gate (PRD W8): an abstaining pack never reaches the answerer.
    if gates_answer(prove_res):
        if tr is not None:
            tr.packed = []
            tr.routes["prove_gate"] = {
                "withheld": len(nodes),
                "profiles_withheld": len(profiles or []),
                "score": prove_res.score if prove_res else None,
                "reason": prove_res.reason if prove_res else "",
            }
        lines.append(
            f"({len(nodes)} related passage(s) withheld: they do not prove an answer.)"
        )
        return "\n".join(lines)
    if profiles:
        lines.append("Profile scratchpad:")
        for p in profiles[:2]:
            lines.append(f"- {p[:400]}")
    comp = last_computed() if computed is _UNSET else list(computed or [])
    if comp:
        lines.append(
            "Computed temporal facts (deterministic; trust these over mental date arithmetic):"
        )
        lines.extend(f"- {c}" for c in comp)
    packed_rows: list[dict[str, Any]] = []
    for i, n in enumerate(nodes, 1):
        prefix = evidence_prefix(n)
        body = clip(n.content, stage="pack", node_id=n.id)
        lines.append(f"{i}. {prefix}{body}")
        packed_rows.append(
            {"id": n.id, "chars": len(n.content or ""), "packed_chars": len(body), "text": body}
        )
    if tr is not None:
        tr.packed = packed_rows
    return "\n".join(lines)


def _infer_answer_mode(query: str) -> str:
    return classify_ability(None, query)
