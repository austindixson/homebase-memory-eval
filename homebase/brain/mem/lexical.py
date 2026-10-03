"""BM25 keyword ranking over memory candidates (GPTR-style fallback)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from homebase.brain.mem.store import MemNode

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOP = frozenset(
    "a an the and or but if in on at to for of is are was were be been being "
    "this that these those it its as with by from into about than then so "
    "not no nor do does did doing done have has had having i you he she we "
    "they them my your our their what which who whom how when where why".split()
)


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _TOKEN_RE.finditer((text or "").lower()):
        t = m.group(0)
        if len(t) < 2 or t in _STOP:
            continue
        # Light plural stem.
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        toks.append(t)
    return toks


def bm25_rank(
    query: str,
    nodes: Sequence[MemNode],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, MemNode]]:
    """Score nodes with BM25 over the candidate set; highest first."""
    if not nodes:
        return []
    q_toks = tokenize(query)
    if not q_toks:
        return [(0.0, n) for n in nodes]

    docs = [tokenize(n.content) for n in nodes]
    lengths = [max(1, len(d)) for d in docs]
    avgdl = sum(lengths) / len(lengths)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    n_docs = len(docs)
    idf = {
        t: math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
        for t in set(q_toks)
    }

    scored: list[tuple[float, MemNode]] = []
    for node, doc, dl in zip(nodes, docs, lengths, strict=True):
        tf = Counter(doc)
        score = 0.0
        for t in q_toks:
            if t not in tf:
                continue
            freq = tf[t]
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            score += idf.get(t, 0.0) * (freq * (k1 + 1.0)) / denom
        scored.append((score, node))
    scored.sort(key=lambda x: -x[0])
    return scored


def select_bm25(
    query: str,
    nodes: Sequence[MemNode],
    *,
    relative_threshold: float = 0.5,
    max_results: int = 8,
) -> list[MemNode]:
    ranked = bm25_rank(query, nodes)
    if not ranked:
        return []
    best = ranked[0][0]
    if best <= 0:
        # No lexical match — return opening candidates (stable order).
        return list(nodes)[:max_results]
    floor = best * relative_threshold
    out: list[MemNode] = []
    for score, node in ranked:
        if score < floor:
            break
        out.append(node)
        if len(out) >= max_results:
            break
    return out
