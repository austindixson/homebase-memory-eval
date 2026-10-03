"""Optional embedding expander for Mem v2 (never the primary filter)."""

from __future__ import annotations

import os
import struct
from typing import Any

import httpx

from homebase.brain.config import Registry
from homebase.brain.mem.store import MemNode, MemStore


def embed_url(reg: Registry | None = None) -> str:
    env = (os.environ.get("BRAIN_EMBED_URL") or "").strip()
    if env:
        return env.rstrip("/")
    if reg is not None:
        return str(getattr(reg, "embed_url", "") or "").rstrip("/")
    return ""


def embed_enabled(reg: Registry | None = None) -> bool:
    return bool(embed_url(reg))


def _pack_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob[: n * 4]))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / ((na**0.5) * (nb**0.5))


def embed_texts(texts: list[str], reg: Registry | None = None) -> list[list[float]]:
    base = embed_url(reg)
    if not base or not texts:
        return []
    model = (
        os.environ.get("BRAIN_EMBED_MODEL")
        or (getattr(reg, "embed_model", None) if reg else None)
        or "qwen3-embed"
    )
    with httpx.Client(timeout=float(os.environ.get("BRAIN_EMBED_TIMEOUT", "60"))) as client:
        r = client.post(
            f"{base}/embeddings",
            json={"model": model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
    items = data.get("data") or []
    # Preserve order by index when present.
    items = sorted(items, key=lambda x: int(x.get("index", 0)))
    return [list(map(float, it.get("embedding") or [])) for it in items]


def embed_and_store(store: MemStore, node: MemNode, reg: Registry) -> None:
    vecs = embed_texts([node.content[:4000]], reg)
    if not vecs or not vecs[0]:
        return
    store.set_embedding(node.id, _pack_f32(vecs[0]))


def search_similar(
    store: MemStore,
    query: str,
    reg: Registry,
    *,
    limit: int = 24,
) -> list[MemNode]:
    """Cosine NN over stored embeddings; empty if embed URL unset or index empty."""
    if not embed_enabled(reg):
        return []
    indexed = store.nodes_with_embeddings(limit=5000)
    if not indexed:
        return []
    qvecs = embed_texts([query[:2000]], reg)
    if not qvecs or not qvecs[0]:
        return []
    qv = qvecs[0]
    ranked: list[tuple[float, MemNode]] = []
    for node, blob in indexed:
        try:
            score = _cosine(qv, _unpack_f32(blob))
        except Exception:
            continue
        ranked.append((score, node))
    ranked.sort(key=lambda t: -t[0])
    return [n for _, n in ranked[:limit]]
