"""Jev Score-as-sort over memory candidates (no absolute delete-gate)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence

from homebase.brain.jev_client import SystemOneClient
from homebase.brain.mem.evidence import clip
from homebase.brain.mem.lexical import select_bm25
from homebase.brain.mem.store import MemNode

# Score-as-sort rubric. Jev allows at most 10 score levels (0-9).
USEFULNESS_CRITERIA_10 = [
    "0 Completely unrelated",
    "1 Barely related topic noise",
    "2 Same broad topic, no useful facts",
    "3 Weak supporting detail",
    "4 Partial answer fragment",
    "5 Partially answers with useful facts",
    "6 Solid supporting evidence",
    "7 Mostly answers the question",
    "8 Directly answers with specific facts",
    "9 Definitive answer evidence",
]

# Cap how many candidates we ask Jev to score (BM25 pre-ranks the rest away).
_DEFAULT_SCORE_CAP = 48
_DEFAULT_BATCH = 8


@dataclass
class FilterStats:
    mode: str  # jev_sort | keyword | passthrough | empty | fts_fallback
    kept: int
    dropped: int
    candidate_n: int
    scores: list[float] = field(default_factory=list)


def _parse_score(raw: Any) -> float | None:
    if isinstance(raw, dict) and raw.get("score") is not None:
        try:
            return min(10.0, max(0.0, float(raw["score"])))
        except (TypeError, ValueError):
            return None
    return None


def _score_batch(
    jev: SystemOneClient,
    query: str,
    nodes: Sequence[MemNode],
) -> list[tuple[float, MemNode]]:
    """Score a small batch in one System One call (fewer HTTP round-trips)."""
    if not nodes:
        return []
    questions: dict[str, Any] = {}
    for i, n in enumerate(nodes):
        # Keep the question in shared state only — passage text alone in instructions
        # so scorers/mocks cannot confuse query tokens with passage content.
        questions[f"u{i}"] = {
            "type": "score",
            "instructions": (
                "Rate 0-10 how useful THIS passage is for the question in state. "
                f"Passage:\n{clip(n.content or '', stage='filter', node_id=n.id)}"
            ),
            "criteria": list(USEFULNESS_CRITERIA_10),
        }
    answers = jev.system_one(
        state=f"question:\n{query[:500]}\n\nScore each passage u0..u{len(nodes)-1}.",
        questions=questions,
    )
    out: list[tuple[float, MemNode]] = []
    for i, n in enumerate(nodes):
        s = _parse_score(answers.get(f"u{i}"))
        out.append((0.0 if s is None else s, n))
    return out


def _score_one(
    jev: SystemOneClient,
    query: str,
    node: MemNode,
) -> float:
    """Return usefulness score in [0, 10]. Score only — Noul is not used for keep."""
    batch = _score_batch(jev, query, [node])
    return batch[0][0] if batch else 0.0


def score_nodes(
    jev: SystemOneClient,
    query: str,
    nodes: Sequence[MemNode],
    *,
    concurrency: int = 4,
    batch_size: int = _DEFAULT_BATCH,
    score_cache: dict[str, float] | None = None,
) -> list[tuple[float, MemNode]]:
    """``score_cache`` (node id -> score, same query) skips re-asking Jev about
    passages already scored earlier in the same retrieve call."""
    if not nodes:
        return []
    # BM25 pre-rank so we spend Jev budget on the best lexical candidates.
    score_cap = int(os.environ.get("MEM_JEV_SCORE_CAP") or _DEFAULT_SCORE_CAP)
    ranked = select_bm25(
        query,
        list(nodes),
        relative_threshold=0.0,
        max_results=min(score_cap, len(nodes)),
    )
    if not ranked:
        ranked = list(nodes)[:score_cap]

    cache = score_cache if score_cache is not None else {}
    cached = [(cache[n.id], n) for n in ranked if n.id in cache]
    ranked = [n for n in ranked if n.id not in cache]
    batches = [
        ranked[i : i + max(1, batch_size)]
        for i in range(0, len(ranked), max(1, batch_size))
    ]
    workers = max(1, min(int(concurrency), len(batches)))
    scored: list[tuple[float, MemNode]] = []

    def _run(batch: Sequence[MemNode]) -> list[tuple[float, MemNode]]:
        try:
            return _score_batch(jev, query, batch)
        except Exception:
            # Retry once serially per node.
            out: list[tuple[float, MemNode]] = []
            for n in batch:
                try:
                    out.append((_score_one(jev, query, n), n))
                except Exception:
                    out.append((-1.0, n))
            return out

    if workers == 1:
        for batch in batches:
            scored.extend(_run(batch))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run, b) for b in batches]
            for fut in as_completed(futs):
                try:
                    scored.extend(fut.result())
                except Exception:
                    pass

    for sc, n in scored:
        if sc >= 0:
            cache[n.id] = sc
    scored.extend(cached)
    # Append unscored candidates at the bottom (score 0) so top-k can still fill.
    seen = {n.id for _, n in scored}
    for n in nodes:
        if n.id not in seen:
            scored.append((0.0, n))
    scored.sort(key=lambda t: -t[0])
    return scored


def filter_by_usefulness(
    jev: SystemOneClient,
    query: str,
    nodes: Sequence[MemNode],
    *,
    min_score: float | None = None,  # ignored — kept for call-site compat
    max_keep: int = 8,
    concurrency: int | None = None,
    keyword_relative_threshold: float = 0.5,
    score_cache: dict[str, float] | None = None,
) -> tuple[list[MemNode], FilterStats]:
    """Score-as-sort: rank by Jev usefulness, take top-k. No absolute delete cut."""
    _ = min_score
    cand_n = len(nodes)
    if cand_n == 0:
        return [], FilterStats(mode="empty", kept=0, dropped=0, candidate_n=0)

    conc = concurrency
    if conc is None:
        conc = int(os.environ.get("MEM_JEV_CONCURRENCY") or "4")

    try:
        scored = score_nodes(jev, query, nodes, concurrency=conc, score_cache=score_cache)
    except Exception:
        kept = select_bm25(
            query,
            nodes,
            relative_threshold=keyword_relative_threshold,
            max_results=max_keep,
        )
        return kept, FilterStats(
            mode="keyword",
            kept=len(kept),
            dropped=max(0, cand_n - len(kept)),
            candidate_n=cand_n,
        )

    ok = [(s, n) for s, n in scored if s >= 0]
    if not ok:
        kept = select_bm25(
            query,
            nodes,
            relative_threshold=keyword_relative_threshold,
            max_results=max_keep,
        )
        return kept, FilterStats(
            mode="keyword",
            kept=len(kept),
            dropped=max(0, cand_n - len(kept)),
            candidate_n=cand_n,
        )

    # Pure Score-as-sort: take top-k by score.
    kept_pairs = ok[:max_keep]
    kept = [n for _, n in kept_pairs]
    scores = [s for s, _ in kept_pairs]
    return kept, FilterStats(
        mode="jev_sort",
        kept=len(kept),
        dropped=max(0, cand_n - len(kept)),
        candidate_n=cand_n,
        scores=scores,
    )
