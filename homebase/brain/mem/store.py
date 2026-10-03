"""SQLite Mem v2 store: dual-tier nodes + typed edges + FTS5 + optional embeddings."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TIMELINE_STOP = frozenset(
    """a an the and or of to in on at for from with by about as is are was were be been being
    do does did done have has had having i me my mine we our you your he she it its they them
    their this that these those there here which what when where who whom whose how why
    many much more most some any all each every than then so if but not no nor only own same
    too very can will just should would could may might must also into over under again
    first last before after earlier later happened happen order time ago days day weeks week
    months month years year between during while since until once ever never
    tell remember recall mention mentioned said say told""".split()
)
_TIMELINE_CANDIDATE_CAP = 400
_TRANSCRIPT_ORDER_STOP = frozenset(
    "list sequence different aspects brought up talked discussed asked topics topic items item "
    "things thing across throughout conversations conversation mentioned mention only "
    # Requested list sizes ("Mention ONLY and ONLY five items") are not topic words.
    "two three four five six seven eight nine ten".split()
)
_FTS_STOP = frozenset(
    "a an the and or of to in on at for from with by about as is are was were be been am do does did "
    "have has had i me my we our you your he she it its they them their this that these those there "
    "which what when where who whom how why so if but not no than then can will would could should "
    "may might must also into over under again".split()
)


def query_tokens(text: str) -> list[str]:
    """Distinctive lowercase tokens of ``text`` (stop-words and 1-2 char tokens removed)."""
    out: list[str] = []
    for t in re.findall(r"[a-z0-9][a-z0-9'-]*", (text or "").lower()):
        t = t.strip("'-")
        if len(t) < 3 or t in _TIMELINE_STOP or t in out:
            continue
        out.append(t)
    return out


def stem(token: str) -> str:
    """Prefix stem with common verb endings normalized (fixed/fixing, planted/planting)."""
    t = token.lower().replace("'s", "").strip("'-")
    for suffix in ("ing", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            t = t[:-len(suffix)]
            break
    return t[:5] if len(t) > 5 else t


def split_sides(query: str) -> list[list[str]]:
    """Named event groups for comparisons and elapsed-time questions."""
    for pattern in (
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bsince\s+(.+?)\s+when\s+(.+?)(?:\?|$)",
    ):
        match = re.search(pattern, query or "", re.I)
        if match:
            a, b = query_tokens(match.group(1)), query_tokens(match.group(2))
            if a and b:
                return [a, b]
    parts = re.split(r"\bor\b", query or "", flags=re.I)
    if len(parts) == 2:
        # "Which X did I do first in May, the A or the B?": the question frame sits
        # before the last comma/colon; side A is only what follows it.
        head = re.split(r"[,:;]", parts[0])[-1]
        a = query_tokens(head) or query_tokens(parts[0])
        b = query_tokens(parts[1])
        if a and b:
            return [a, b]
    return [query_tokens(query)]


@dataclass
class TimelineSearch:
    nodes: list["MemNode"]
    tokens: list[str]
    sides: list[list[str]]
    side_hits: list[int]
    both_events_present: bool
    zero_hit: bool
    n_candidates: int


def spread_by_position(ranked: list["MemNode"], k: int) -> list["MemNode"]:
    """Pick ``k`` of ``ranked`` (best first) that cover the transcript, in position order.

    Mention-order questions ("list the order in which I brought up …") ask for one item
    per stage of a long conversation. A global top-k of a generic topic clusters in one
    stretch, so the position range is cut into ``k`` equal bins, each bin keeps its
    best-ranked node, and leftover slots go to the next best nodes overall.
    Nodes without a ``source_position`` are dropped.
    """
    placed = [n for n in ranked if n.source_position is not None]
    if k <= 0 or not placed:
        return []
    if len(placed) > k:
        lo = min(n.source_position for n in placed)
        hi = max(n.source_position for n in placed)
        width = (hi - lo + 1) / k
        winners: dict[int, "MemNode"] = {}
        for n in placed:
            winners.setdefault(min(k - 1, int((n.source_position - lo) / width)), n)
        chosen = {n.id for n in winners.values()}
        picked = list(winners.values())
        for n in placed:
            if len(picked) >= k:
                break
            if n.id not in chosen:
                picked.append(n)
                chosen.add(n.id)
        placed = picked
    return sorted(placed, key=lambda n: (n.source_position, n.ts))


EDGE_KINDS = ("semantic", "temporal", "causal", "entity", "supersedes", "conflicts_with")
TIERS = ("fact", "episodic", "claim")


@dataclass
class MemNode:
    id: str
    content: str
    ts: float
    types: dict[str, float]
    tier: str = "fact"
    scope_project: str | None = None
    scope_person: str | None = None
    scope_session: str | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    has_embedding: bool = False
    source_position: int | None = None
    source_role: str | None = None

    @property
    def event_time(self) -> float | None:
        """When the remembered thing happened, or None if only the write time is known.

        ``valid_from`` defaults to the write time (``ts``) when no event date was
        supplied, so a value equal to ``ts`` means "unset", never a real date.
        """
        if self.valid_from is not None and abs(self.valid_from - (self.ts or 0.0)) > 1.0:
            return float(self.valid_from)
        return None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MemNode:
        keys = set(row.keys())
        types = json.loads(row["types_json"] or "{}")
        emb = row["embedding"] if "embedding" in keys else None
        return cls(
            id=row["id"],
            content=row["content"],
            ts=float(row["ts"]),
            types=types,
            tier=str(row["tier"]) if "tier" in keys and row["tier"] else "fact",
            scope_project=(
                str(row["scope_project"])
                if "scope_project" in keys and row["scope_project"]
                else None
            ),
            scope_person=(
                str(row["scope_person"])
                if "scope_person" in keys and row["scope_person"]
                else None
            ),
            scope_session=(
                str(row["scope_session"])
                if "scope_session" in keys and row["scope_session"]
                else None
            ),
            valid_from=(
                float(row["valid_from"])
                if "valid_from" in keys and row["valid_from"] is not None
                else None
            ),
            valid_to=(
                float(row["valid_to"])
                if "valid_to" in keys and row["valid_to"] is not None
                else None
            ),
            has_embedding=bool(emb),
            source_position=(
                int(row["source_position"])
                if "source_position" in keys and row["source_position"] is not None
                else None
            ),
            source_role=(
                str(row["source_role"])
                if "source_role" in keys and row["source_role"]
                else None
            ),
        )


class MemStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _columns(self, table: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r["name"]) for r in rows}

    def _ensure_column(self, table: str, name: str, decl: str) -> None:
        if name not in self._columns(table):
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _init_schema(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
              id TEXT PRIMARY KEY,
              content TEXT NOT NULL,
              ts REAL NOT NULL,
              types_json TEXT NOT NULL DEFAULT '{}',
              embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS edges (
              src TEXT NOT NULL,
              dst TEXT NOT NULL,
              kind TEXT NOT NULL,
              score REAL NOT NULL,
              PRIMARY KEY (src, dst, kind),
              FOREIGN KEY (src) REFERENCES nodes(id),
              FOREIGN KEY (dst) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
              content, id UNINDEXED, content='nodes', content_rowid='rowid'
            );
            """
        )
        # Mem v2 columns (safe on existing DBs).
        self._ensure_column("nodes", "tier", "TEXT NOT NULL DEFAULT 'fact'")
        self._ensure_column("nodes", "scope_project", "TEXT")
        self._ensure_column("nodes", "scope_person", "TEXT")
        self._ensure_column("nodes", "scope_session", "TEXT")
        self._ensure_column("nodes", "valid_from", "REAL")
        self._ensure_column("nodes", "valid_to", "REAL")
        self._ensure_column("nodes", "source_position", "INTEGER")
        self._ensure_column("nodes", "source_role", "TEXT")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_valid_to ON nodes(valid_to)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_scope_project ON nodes(scope_project)"
        )
        # Lattice derived views (Mem v3).
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS timeline_events (
              id TEXT PRIMARY KEY,
              node_id TEXT NOT NULL,
              t_sort REAL NOT NULL,
              t_label TEXT,
              entity TEXT,
              scope_session TEXT,
              FOREIGN KEY (node_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_timeline_sort ON timeline_events(t_sort);
            CREATE INDEX IF NOT EXISTS idx_timeline_session
              ON timeline_events(scope_session);
            CREATE TABLE IF NOT EXISTS conflict_pairs (
              id TEXT PRIMARY KEY,
              a_id TEXT NOT NULL,
              b_id TEXT NOT NULL,
              topic TEXT,
              ts REAL NOT NULL,
              FOREIGN KEY (a_id) REFERENCES nodes(id),
              FOREIGN KEY (b_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_conflict_topic ON conflict_pairs(topic);
            CREATE TABLE IF NOT EXISTS profiles (
              scope_key TEXT PRIMARY KEY,
              content TEXT NOT NULL,
              updated_ts REAL NOT NULL
            );
            """
        )
        self._ensure_column("timeline_events", "t_kind", "TEXT")
        c.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
              INSERT INTO nodes_fts(rowid, content, id) VALUES (new.rowid, new.content, new.id);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
              INSERT INTO nodes_fts(nodes_fts, rowid, content, id)
                VALUES('delete', old.rowid, old.content, old.id);
            END;
            """
        )
        c.commit()

    def close(self) -> None:
        self._conn.close()

    def add_node(
        self,
        content: str,
        types: dict[str, float] | None = None,
        ts: float | None = None,
        node_id: str | None = None,
        *,
        tier: str = "fact",
        scope_project: str | None = None,
        scope_person: str | None = None,
        scope_session: str | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        embedding: bytes | None = None,
        source_position: int | None = None,
        source_role: str | None = None,
    ) -> MemNode:
        nid = node_id or uuid.uuid4().hex
        ts_v = float(ts if ts is not None else time.time())
        tier_v = tier if tier in TIERS else "fact"
        types_j = json.dumps(types or {})
        vf = float(valid_from) if valid_from is not None else ts_v
        self._conn.execute(
            """
            INSERT INTO nodes(
              id, content, ts, types_json, embedding, tier,
              scope_project, scope_person, scope_session, valid_from, valid_to,
              source_position, source_role
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                nid,
                content,
                ts_v,
                types_j,
                embedding,
                tier_v,
                scope_project,
                scope_person,
                scope_session,
                vf,
                valid_to,
                int(source_position) if source_position is not None else None,
                source_role,
            ),
        )
        self._conn.commit()
        return MemNode(
            id=nid,
            content=content,
            ts=ts_v,
            types=types or {},
            tier=tier_v,
            scope_project=scope_project,
            scope_person=scope_person,
            scope_session=scope_session,
            valid_from=vf,
            valid_to=valid_to,
            has_embedding=bool(embedding),
            source_position=(int(source_position) if source_position is not None else None),
            source_role=source_role,
        )

    def get_node(self, node_id: str) -> MemNode | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return MemNode.from_row(row) if row else None

    def invalidate_node(self, node_id: str, *, at: float | None = None) -> None:
        ts_v = float(at if at is not None else time.time())
        self._conn.execute(
            "UPDATE nodes SET valid_to=? WHERE id=? AND valid_to IS NULL",
            (ts_v, node_id),
        )
        self._conn.commit()

    def mark_superseded(
        self, a: MemNode, b: MemNode, score: float
    ) -> tuple[MemNode, MemNode]:
        """Keep both nodes; mark the *older by event time* superseded (PRD W7).

        ``a`` is the node being written (arrival order breaks ties). Returns
        ``(newer, older)``. The older node's ``valid_to`` is the newer's event time,
        so "as of" reads stay correct even when sessions are ingested out of order.
        """
        def explicit(n: MemNode) -> bool:
            # valid_from == ts means "unset" (defaulted to write time), not an event date.
            return n.valid_from is not None and abs(n.valid_from - (n.ts or 0.0)) > 1.0

        # Compare event times only when both nodes have one; otherwise arrival order.
        if explicit(a) and explicit(b) and a.valid_from < b.valid_from:  # type: ignore[operator]
            newer, older = b, a
        else:
            newer, older = a, b
        at = newer.valid_from if newer.valid_from is not None else newer.ts
        self.invalidate_node(older.id, at=at)
        self.add_edge(newer.id, older.id, "supersedes", score)
        return newer, older

    def set_embedding(self, node_id: str, blob: bytes) -> None:
        self._conn.execute(
            "UPDATE nodes SET embedding=? WHERE id=?",
            (blob, node_id),
        )
        self._conn.commit()

    def nodes_with_embeddings(self, limit: int = 5000) -> list[tuple[MemNode, bytes]]:
        rows = self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE embedding IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        out: list[tuple[MemNode, bytes]] = []
        for r in rows:
            out.append((MemNode.from_row(r), bytes(r["embedding"])))
        return out

    def add_edge(self, src: str, dst: str, kind: str, score: float) -> None:
        if kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {kind}")
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(src, dst, kind, score) VALUES (?,?,?,?)",
            (src, dst, kind, float(score)),
        )
        self._conn.commit()

    def neighbors(
        self, node_id: str, kind: str | None = None, limit: int = 20
    ) -> list[tuple[MemNode, str, float]]:
        if kind:
            rows = self._conn.execute(
                """
                SELECT n.*, e.kind, e.score FROM edges e
                JOIN nodes n ON n.id = e.dst
                WHERE e.src=? AND e.kind=?
                ORDER BY e.score DESC LIMIT ?
                """,
                (node_id, kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT n.*, e.kind, e.score FROM edges e
                JOIN nodes n ON n.id = e.dst
                WHERE e.src=?
                ORDER BY e.score DESC LIMIT ?
                """,
                (node_id, limit),
            ).fetchall()
        return [(MemNode.from_row(r), str(r["kind"]), float(r["score"])) for r in rows]

    def _fts_rows(
        self, match: str, limit: int, *, current_only: bool, ranked: bool
    ) -> list[sqlite3.Row]:
        where = "nodes_fts MATCH ?" + (" AND n.valid_to IS NULL" if current_only else "")
        order = " ORDER BY bm25(nodes_fts)" if ranked else ""
        return self._conn.execute(
            f"SELECT n.* FROM nodes_fts f JOIN nodes n ON n.id = f.id "
            f"WHERE {where}{order} LIMIT ?",
            (match, limit),
        ).fetchall()

    def fts_search(
        self,
        query: str,
        limit: int = 20,
        *,
        current_only: bool = False,
        mode: str | None = None,
    ) -> list[MemNode]:
        """Full-text search.

        ``and`` (legacy): every one of the first 12 words must appear, so a long
        natural-language question almost never matches anything. ``or`` (default):
        exact AND matches first (unchanged), then the rest of the budget is filled
        with documents matching *any* distinctive word, ranked by BM25. Set
        ``HOMEBASE_MEM_FTS_MODE=and`` to restore the legacy behavior.
        """
        q = query.strip()
        if not q:
            return []
        mode = (mode or os.environ.get("HOMEBASE_MEM_FTS_MODE") or "or").strip().lower()
        tokens = [t for t in q.replace('"', " ").split() if t]
        if not tokens:
            return []
        and_match = " ".join(f'"{t}"' for t in tokens[:12])
        try:
            rows = self._fts_rows(and_match, limit, current_only=current_only, ranked=False)
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{tokens[0]}%", limit),
            ).fetchall()
        out = [MemNode.from_row(r) for r in rows]
        if mode == "and" or len(out) >= limit:
            return out
        words: list[str] = []
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", q):
            t = t.strip("'-_").lower()
            if len(t) >= 2 and t not in _FTS_STOP and t not in words:
                words.append(t)
        if len(words) < 2:
            return out
        or_match = " OR ".join(f'"{w}"' for w in words[:16])
        try:
            more = self._fts_rows(or_match, limit * 3, current_only=current_only, ranked=True)
        except sqlite3.OperationalError:
            return out
        seen = {n.id for n in out}
        for r in more:
            if r["id"] in seen:
                continue
            out.append(MemNode.from_row(r))
            seen.add(r["id"])
            if len(out) >= limit:
                break
        return out

    def recent(self, limit: int = 20, *, current_only: bool = False) -> list[MemNode]:
        if current_only:
            rows = self._conn.execute(
                """
                SELECT * FROM nodes WHERE valid_to IS NULL
                ORDER BY ts DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [MemNode.from_row(r) for r in rows]

    def all_except(self, node_id: str, limit: int = 50) -> list[MemNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE id != ? ORDER BY ts DESC LIMIT ?",
            (node_id, limit),
        ).fetchall()
        return [MemNode.from_row(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()
        return int(row["c"])

    def all_nodes(self, limit: int = 5000) -> list[MemNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY ts DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [MemNode.from_row(r) for r in rows]

    def all_edges(self, limit: int = 20000) -> list[tuple[str, str, str, float]]:
        rows = self._conn.execute(
            """
            SELECT src, dst, kind, score FROM edges
            ORDER BY score DESC LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        return [
            (str(r["src"]), str(r["dst"]), str(r["kind"]), float(r["score"]))
            for r in rows
        ]

    def edge_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()
        return int(row["c"])

    def add_timeline_event(
        self,
        node_id: str,
        *,
        t_sort: float,
        t_label: str | None = None,
        entity: str | None = None,
        scope_session: str | None = None,
        event_id: str | None = None,
        t_kind: str | None = None,
    ) -> str:
        """``t_kind``: absolute (session/mention date) | relative (event date resolved
        from a phrase like "two weeks ago") | ordinal (source transcript position).
        """
        eid = event_id or uuid.uuid4().hex
        self._conn.execute(
            """
            INSERT OR REPLACE INTO timeline_events(
              id, node_id, t_sort, t_label, entity, scope_session, t_kind
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (eid, node_id, float(t_sort), t_label, entity, scope_session, t_kind),
        )
        self._conn.commit()
        return eid

    def timeline_for_nodes(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Earliest timeline event per node: ``{id: {t_sort, t_label, t_kind}}``."""
        ids = [i for i in node_ids if i]
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT node_id, t_sort, t_label, t_kind FROM timeline_events "
            f"WHERE node_id IN ({marks}) ORDER BY t_sort ASC",
            ids,
        ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            out.setdefault(
                str(r["node_id"]),
                {"t_sort": float(r["t_sort"]), "t_label": r["t_label"], "t_kind": r["t_kind"]},
            )
        return out

    def are_conflicting(self, a_id: str, b_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM conflict_pairs WHERE (a_id=? AND b_id=?) OR (a_id=? AND b_id=?) LIMIT 1",
            (a_id, b_id, b_id, a_id),
        ).fetchone()
        return row is not None

    def search_timeline_ex(
        self,
        query: str,
        *,
        limit: int = 20,
        scope_session: str | None = None,
        direction: str = "asc",
        transcript_order: bool = False,
        speaker_role: str | None = None,
    ) -> TimelineSearch:
        """Token/entity match over timeline events, ordered by ``t_sort``.

        An A-vs-B question ("which came first, X or Y") is split into sides and each
        side is searched separately so both events appear in the result.
        """
        sides = split_sides(query)
        if transcript_order:
            # The wording of list/sequence questions is shared across many prompts.
            # Remove that frame before matching so retrieval is driven by the topic.
            sides = [[t for t in side if t not in _TRANSCRIPT_ORDER_STOP] for side in sides]
            return self._search_transcript(
                sides, limit=limit, scope_session=scope_session, speaker_role=speaker_role
            )
        all_tokens = [t for side in sides for t in side]
        stems = {t: stem(t) for t in all_tokens}
        params: list[Any] = []
        where: list[str] = []
        if scope_session:
            where.append("t.scope_session = ?")
            params.append(scope_session)
        if speaker_role:
            where.append("LOWER(COALESCE(n.source_role,'')) = ?")
            params.append(speaker_role.lower())
        if all_tokens:
            likes = []
            for tok in all_tokens:
                likes.append(
                    "(LOWER(COALESCE(t.t_label,'')) LIKE ? OR LOWER(COALESCE(t.entity,'')) LIKE ?"
                    " OR LOWER(n.content) LIKE ?)"
                )
                params.extend([f"%{stems[tok]}%"] * 3)
            where.append("(" + " OR ".join(likes) + ")")
        sql = (
            "SELECT n.*, t.t_sort AS _t_sort FROM timeline_events t "
            "JOIN nodes n ON n.id = t.node_id"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " LIMIT ?"
        params.append(_TIMELINE_CANDIDATE_CAP)
        rows = self._conn.execute(sql, params).fetchall()

        def order_key(row: sqlite3.Row) -> tuple[float, float]:
            if transcript_order and row["source_position"] is not None:
                return (float(row["source_position"]), float(row["ts"]))
            return (float(row["_t_sort"]), float(row["ts"]))

        def score(row: sqlite3.Row, toks: list[str]) -> int:
            blob = (row["content"] or "").lower()
            return sum(1 for t in toks if stems.get(t, stem(t)) in blob)

        picked: dict[str, tuple[tuple[float, float], sqlite3.Row]] = {}
        side_hits: list[int] = []
        per_side = max(1, limit // len(sides))
        for toks in sides:
            ranked = sorted(
                (r for r in rows if score(r, toks) > 0),
                key=lambda r: (-score(r, toks), order_key(r)),
            )[:per_side]
            side_hits.append(len(ranked))
            for r in ranked:
                picked.setdefault(r["id"], (order_key(r), r))
        if not all_tokens:
            for r in sorted(rows, key=order_key)[:limit]:
                picked.setdefault(r["id"], (order_key(r), r))
        ordered = sorted(picked.values(), key=lambda p: p[0], reverse=(direction == "desc"))
        nodes = [MemNode.from_row(r) for _, r in ordered[:limit]]
        return TimelineSearch(
            nodes=nodes,
            tokens=all_tokens,
            sides=sides,
            side_hits=side_hits,
            both_events_present=len(sides) > 1 and all(h > 0 for h in side_hits),
            zero_hit=not nodes,
            n_candidates=len(rows),
        )

    def _search_transcript(
        self,
        sides: list[list[str]],
        *,
        limit: int,
        scope_session: str | None,
        speaker_role: str | None,
    ) -> TimelineSearch:
        """Mention-order search over transcript turns, spread across the conversation.

        Reads turns straight from ``nodes`` (transcript position lives there; timeline
        rows add nothing and a capped, unordered timeline query kept only the earliest
        turns). Each turn is scored by topic-token hits, one chunk per position is kept,
        and ``spread_by_position`` picks the result so every stretch of the conversation
        can contribute instead of the earliest ties.
        """
        all_tokens = [t for side in sides for t in side]
        stems = {t: stem(t) for t in all_tokens}
        where = ["source_position IS NOT NULL"]
        params: list[Any] = []
        if scope_session:
            where.append("scope_session = ?")
            params.append(scope_session)
        if speaker_role:
            where.append("LOWER(COALESCE(source_role,'')) = ?")
            params.append(speaker_role.lower())
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE " + " AND ".join(where), params
        ).fetchall()
        best: dict[int, tuple[int, MemNode]] = {}
        for r in rows:
            blob = (r["content"] or "").lower()
            hits = sum(1 for t in all_tokens if stems[t] in blob)
            if all_tokens and hits == 0:
                continue
            node = MemNode.from_row(r)
            pos = int(node.source_position)
            if pos not in best or hits > best[pos][0]:
                best[pos] = (hits, node)
        ranked = [n for _, n in sorted(best.values(), key=lambda p: (-p[0], p[1].source_position))]
        nodes = spread_by_position(ranked, limit)
        return TimelineSearch(
            nodes=nodes,
            tokens=all_tokens,
            sides=sides,
            side_hits=[len(nodes)],
            both_events_present=False,
            zero_hit=not nodes,
            n_candidates=len(rows),
        )

    def search_timeline(
        self,
        query: str,
        *,
        limit: int = 20,
        scope_session: str | None = None,
        direction: str = "asc",
    ) -> list[MemNode]:
        return self.search_timeline_ex(
            query, limit=limit, scope_session=scope_session, direction=direction
        ).nodes

    def add_conflict_pair(
        self,
        a_id: str,
        b_id: str,
        *,
        topic: str | None = None,
        ts: float | None = None,
        pair_id: str | None = None,
    ) -> str:
        pid = pair_id or uuid.uuid4().hex
        ts_v = float(ts if ts is not None else time.time())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO conflict_pairs(id, a_id, b_id, topic, ts)
            VALUES (?,?,?,?,?)
            """,
            (pid, a_id, b_id, topic, ts_v),
        )
        self._conn.commit()
        try:
            self.add_edge(a_id, b_id, "conflicts_with", 1.0)
            self.add_edge(b_id, a_id, "conflicts_with", 1.0)
        except Exception:
            pass
        return pid

    def search_conflicts(
        self, query: str, *, limit: int = 10
    ) -> list[tuple[MemNode, MemNode, str | None]]:
        q = (query or "").strip().lower()
        rows = self._conn.execute(
            """
            SELECT a_id, b_id, topic FROM conflict_pairs
            ORDER BY ts DESC LIMIT ?
            """,
            (max(limit * 4, 40),),
        ).fetchall()
        out: list[tuple[MemNode, MemNode, str | None]] = []
        for r in rows:
            topic = str(r["topic"] or "")
            if q and q not in topic.lower():
                # Soft filter: keep if either body matches.
                a = self.get_node(str(r["a_id"]))
                b = self.get_node(str(r["b_id"]))
                if not a or not b:
                    continue
                blob = (a.content + " " + b.content).lower()
                if q not in blob and not any(t in blob for t in q.split() if len(t) > 3):
                    continue
            else:
                a = self.get_node(str(r["a_id"]))
                b = self.get_node(str(r["b_id"]))
                if not a or not b:
                    continue
            out.append((a, b, topic or None))
            if len(out) >= limit:
                break
        return out

    def upsert_profile(self, scope_key: str, content: str) -> None:
        self._conn.execute(
            """
            INSERT INTO profiles(scope_key, content, updated_ts)
            VALUES (?,?,?)
            ON CONFLICT(scope_key) DO UPDATE SET
              content=excluded.content,
              updated_ts=excluded.updated_ts
            """,
            (scope_key, content[:800], time.time()),
        )
        self._conn.commit()

    def get_profile(self, scope_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM profiles WHERE scope_key=?",
            (scope_key,),
        ).fetchone()
        return str(row["content"]) if row else None

    def find_profile_for_query(self, query: str, *, limit: int = 3) -> list[str]:
        rows = self._conn.execute(
            "SELECT scope_key, content FROM profiles ORDER BY updated_ts DESC LIMIT 40"
        ).fetchall()
        qtoks = {t.lower() for t in (query or "").split() if len(t) > 3}
        scored: list[tuple[int, str]] = []
        for r in rows:
            content = str(r["content"] or "")
            key = str(r["scope_key"] or "")
            blob = (key + " " + content).lower()
            hits = sum(1 for t in qtoks if t in blob)
            if hits or not qtoks:
                scored.append((hits, content))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:limit]]

    def supersede_neighbors(self, node_id: str, limit: int = 5) -> list[MemNode]:
        """Nodes this claim supersedes (old bodies kept readable)."""
        return [n for n, _k, _s in self.neighbors(node_id, kind="supersedes", limit=limit)]

    def clear_all(self) -> None:
        """Wipe lattice tables (bench isolation).

        ``nodes_fts`` is an external-content FTS5 index kept in sync by triggers on
        ``nodes``. Deleting the FTS rows and *then* the nodes made the delete trigger
        remove index entries that were already gone, which corrupts FTS5 ("database
        disk image is malformed") on the first clear. Instead: delete the base rows
        with the delete trigger dropped, empty the index with FTS5's ``delete-all``,
        and recreate the triggers.
        """
        c = self._conn
        c.execute("DROP TRIGGER IF EXISTS nodes_ad")
        c.execute("DROP TRIGGER IF EXISTS nodes_au")
        for table in ("edges", "timeline_events", "conflict_pairs", "profiles", "nodes"):
            try:
                c.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        c.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('delete-all')")
        c.commit()
        self._init_schema()  # idempotent: recreates the dropped triggers
        c.commit()

    def backup_to(self, dst: sqlite3.Connection) -> None:
        """Consistent copy of the whole DB (FTS shadow tables included) into ``dst``."""
        self._conn.commit()
        self._conn.backup(dst)

    def restore_from(self, src: sqlite3.Connection) -> None:
        """Overwrite this DB with ``src`` in place (bench snapshots).

        Page-level copy through the live connection, so the external-content FTS index
        and its base rows arrive together; there is no row-by-row delete that could
        desynchronise them the way the old double clear did.
        """
        self._conn.commit()
        src.backup(self._conn)
        self._init_schema()  # snapshots from an older schema gain new columns
        self._conn.commit()
