"""Authenticated AML textual Add/Search adapter for mem-lite + Lattice + Jev.

Each exact user_id has a separate database. Add publishes a complete new SQLite
snapshot atomically, including its retry receipt; a failed write never becomes
searchable. Only memory sources are returned, without answer/proof instructions.
AML forbids Search from generating answers or disguising them as memory records, so
the question-specific computed gap is off unless AML_COMPUTED_GAPS=1 (rehearsal only).
Run as one worker: existing retrieval telemetry uses process-global state.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from homebase.brain.config import Registry, load_registry
from homebase.brain.jev_client import JevClient
from homebase.brain.mem.cards import fact_cards
from homebase.brain.mem.gaps import gap_fact
from homebase.brain.mem.relative import resolve_phrases
from homebase.brain.mem.retrieve import retrieve
from homebase.brain.mem.store import MemNode, MemStore, stem
from homebase.brain.mem.write import write_observation

Nonempty = Annotated[StrictStr, Field(min_length=1)]
RECALL_BUDGET_TOKENS = 100_000  # estimated tokens of memory text; AML's answer input budget is 117,760 tokens
RECALL_BLOCK_CHARS = 6_000      # over budget, selection works on blocks of consecutive turns of about this size
RECALL_FOCUS_TURNS = 12         # over budget, the top retrieval hits are repeated verbatim at the end


def code_sha():
    """Short hash of the memory code that shapes Search output, so run reports can name their build."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _copy_user_db(target, pending):
    if target.exists():
        with sqlite3.connect(target) as source, sqlite3.connect(pending) as destination:
            source.backup(destination)


def _prior_add_receipt(store, request_id, digest, receipt):
    existing = store._conn.execute('SELECT digest FROM aml_receipts WHERE request_id=?', (request_id,)).fetchone()
    if not existing:
        return None
    if existing[0] != digest:
        raise HTTPException(422, 'request_id reused with a different payload')
    return receipt


def _message_stamp(message):
    if message.timestamp is None:
        return None, None
    seconds = message.timestamp/1000
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(), seconds


def _publish_snapshot(pending, target, root):
    with pending.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(pending, target)
    dir_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['user', 'assistant']
    content: Nonempty
    timestamp: Annotated[StrictInt, Field(ge=0, le=253402300799999)] | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: Nonempty
    user_id: Nonempty
    session_id: Nonempty
    messages: Annotated[list[Message], Field(min_length=1)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: Nonempty
    user_id: Nonempty
    top_k: Annotated[StrictInt, Field(ge=1, le=100)]
    options: list[StrictStr] | None = None


class MemoryBackend:
    def __init__(self, root, registry, jev, writer, *, window=0, whole_session=False, computed_gaps=False,
                 fact_cards=False, full_recall=False, recall_budget_tokens=RECALL_BUDGET_TOKENS,
                 recall_block_chars=RECALL_BLOCK_CHARS, retriever=None):
        self.window, self.whole_session = int(window), bool(whole_session)
        self.computed_gaps, self.fact_cards = bool(computed_gaps), bool(fact_cards)
        self.full_recall, self.recall_budget_tokens = bool(full_recall), int(recall_budget_tokens)
        self.recall_block_chars = int(recall_block_chars)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.registry, self.jev, self.writer = registry, jev, writer
        self.retriever = retriever or retrieve
        self.lock = threading.RLock()

    @contextmanager
    def exclusive(self):
        # Serialize the non-thread-local retrieval state and publication across processes.
        with self.lock, (self.root/'.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def path(self, user):
        return self.root / (hashlib.sha256(user.encode()).hexdigest()+'.sqlite')

    def _append_messages(self, store, request, position):
        session = hashlib.sha256(request.session_id.encode()).hexdigest()
        for message in request.messages:
            date, valid_from = _message_stamp(message)
            header = f'[AML] session={session} position={position} turn={position} {message.role}:'
            if date:
                header += f' date={date}'
            # Relative phrases ("last week") are resolved against the message's own date and stored with it.
            notes = resolve_phrases(message.content, valid_from)
            body = message.content + (f"\n(time: {'; '.join(notes)})" if notes else '')
            self.writer(header+'\n'+body, store, self.jev, self.registry,
                tier='episodic', scope_session=session, valid_from=valid_from)
            if self.fact_cards:
                for card in fact_cards(message.content, valid_from):
                    self.writer(f'[AML-CARD] session={session} position={position}\n{card}', store, self.jev,
                        self.registry, tier='episodic', scope_session=session, valid_from=valid_from)
            position += 1
        return position

    def add(self, request):
        payload = request.model_dump()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        receipt = {k: payload[k] for k in ['request_id','user_id','session_id']}
        receipt['success'] = True
        with self.exclusive():
            target = self.path(request.user_id)
            # Avoid publishing write_observation's intermediate commits on a failed Add.
            with tempfile.TemporaryDirectory(prefix='.pending-', dir=self.root) as directory:
                pending = Path(directory)/'memory.sqlite'
                _copy_user_db(target, pending)
                store = MemStore(pending)
                try:
                    store._conn.executescript('''
                        CREATE TABLE IF NOT EXISTS aml_receipts (request_id TEXT PRIMARY KEY, digest TEXT NOT NULL);
                        CREATE TABLE IF NOT EXISTS aml_sessions (session_id TEXT PRIMARY KEY, next_position INTEGER NOT NULL);
                    ''')
                    duplicate = _prior_add_receipt(store, request.request_id, digest, receipt)
                    if duplicate:
                        return duplicate
                    previous = store._conn.execute('SELECT next_position FROM aml_sessions WHERE session_id=?', (request.session_id,)).fetchone()
                    position = self._append_messages(store, request, previous[0] if previous else 0)
                    store._conn.execute('INSERT OR REPLACE INTO aml_sessions VALUES (?,?)', (request.session_id,position))
                    store._conn.execute('INSERT INTO aml_receipts VALUES (?,?)', (request.request_id,digest))
                    store._conn.commit()
                    store._conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                finally:
                    store._conn.close()
                # A single committed image holds both data and the idempotency receipt.
                _publish_snapshot(pending, target, self.root)
        return receipt

    def search(self, request):
        with self.exclusive():
            path = self.path(request.user_id)
            if not path.exists():
                return {'data':[]}
            store = MemStore(path)
            try:
                query = request.query
                if request.options:
                    query += '\nOptions:\n'+'\n'.join(request.options)
                if self.full_recall:
                    anchors = []
                    if _history_tokens(store) > self.recall_budget_tokens:
                        hits = self.retriever(query, store, self.jev, self.registry, top_k=request.top_k, run_prove=False)
                        anchors = [_source_turn(store, node).id for node in hits]
                    return {'data': _full_recall(store, query, self.recall_budget_tokens, request.top_k,
                        self.recall_block_chars, anchors)}
                # AML owns answering and scoring; return ranked source memories only.
                nodes = self.retriever(query, store, self.jev, self.registry,
                    top_k=request.top_k, run_prove=False)
                items = _search_items(store, nodes, window=self.window, whole_session=self.whole_session)
                # A question-specific computed answer is a rehearsal diagnostic only: AML forbids Search from
                # generating answers or presenting them as memory records.
                computed = _gap_item(store, nodes, request.query) if self.computed_gaps else None
                return {'data': ([computed] if computed else []) + items[:request.top_k - bool(computed)]}
            finally:
                store._conn.close()


_CARD_HEADER = re.compile(r'^\[AML-CARD\] session=\S+ position=\d+\n')
_TURN_HEADER = re.compile(
    r'^\[AML\] session=(?P<session>\S+) position=(?P<position>\d+) turn=\d+ (?P<role>user|assistant):(?: date=\S+)?\n')


def _source_turn(store, node):
    """A derived claim's source turn (claims link to it by a temporal edge)."""
    if _TURN_HEADER.match(node.content) or _CARD_HEADER.match(node.content):
        return node
    row = store._conn.execute("SELECT dst FROM edges WHERE src=? AND kind='temporal'", (node.id,)).fetchone()
    source = store.get_node(row[0]) if row else None
    return source if source is not None and (_TURN_HEADER.match(source.content) or _CARD_HEADER.match(source.content)) else node


def _turn_text(node: MemNode):
    match = _TURN_HEADER.match(node.content)
    card = _CARD_HEADER.match(node.content)
    if card:
        return node.content[card.end():].strip()
    return (f"{match['role']}: {node.content[match.end():]}" if match else node.content).strip()


def _search_item(node: MemNode, content=None):
    item = {'id': node.id, 'content': content if content is not None else _turn_text(node)}
    # The conversation time, never ingest time; omitted when the message had none.
    if node.event_time is not None:
        item['created_at'] = datetime.fromtimestamp(node.event_time, timezone.utc).isoformat()
    return item


def _sessions_in_time_order(store):
    """Every stored conversation turn, grouped by session; sessions ordered by their conversation date.

    Undated sessions follow the dated ones in the order they were added (the only order available).
    """
    sessions = {}
    for row in store._conn.execute("SELECT * FROM nodes WHERE content LIKE '[AML] session=%'"):
        node = MemNode.from_row(row)
        match = _TURN_HEADER.match(node.content)
        if match:
            sessions.setdefault(match['session'], {}).setdefault(int(match['position']), node)
    ordered = []
    for turns in sessions.values():
        nodes = [turns[p] for p in sorted(turns)]
        first = next((n.event_time for n in nodes if n.event_time is not None), None)
        ordered.append((first is None, first or 0.0, min(n.ts or 0.0 for n in nodes), nodes))
    ordered.sort(key=lambda entry: entry[:3])
    return [entry[3] for entry in ordered]


def _words(text):
    return [stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())]


def _relevance_order(query, texts):
    """Sessions ranked by BM25 against the query (most relevant first)."""
    docs = [_words(t) for t in texts]
    count, average = len(docs), (sum(map(len, docs)) / len(docs)) or 1.0
    frequency = Counter(w for d in docs for w in set(d))
    terms = _words(query)
    def score(doc):
        tf = Counter(doc)
        return sum(math.log(1 + (count - frequency[w] + .5) / (frequency[w] + .5)) * tf[w] * 2.5
                   / (tf[w] + 1.5 * (.25 + .75 * len(doc) / average)) for w in terms if w in tf)
    return sorted(range(count), key=lambda i: -score(docs[i]))


_PIECE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def estimate_tokens(text):
    """Tokenizer-free token estimate: letter runs, single digits and symbols, plus 5%. Within about 1% of
    Qwen3's tokenizer on LoCoMo, LongMemEval and code-heavy BEAM text (characters per token there range
    3.2-4.9, so a character budget would overflow on code)."""
    return math.ceil(1.05 * len(_PIECE.findall(text)))


def _history_tokens(store):
    return sum(estimate_tokens(_turn_text(n)) for nodes in _sessions_in_time_order(store) for n in nodes)


def _blocks(nodes, block_chars):
    """Consecutive turns of one session cut into blocks of about ``block_chars`` characters."""
    blocks, current, size = [], [], 0
    for node in nodes:
        length = len(_turn_text(node)) + 1
        if current and size + length > block_chars:
            blocks.append(current)
            current, size = [], 0
        current.append(node)
        size += length
    return blocks + ([current] if current else [])


def _full_recall(store, query, budget, top_k, block_chars=RECALL_BLOCK_CHARS, anchors=()):
    """The whole history in time order when it fits ``budget`` (estimated tokens). Otherwise the blocks of turns most
    relevant to the query that fit, still in time order, neighbouring blocks of a session joined back together.
    Blocks holding ``anchors`` (retrieval hits, best first) are taken before keyword (BM25) matches.
    Verbatim turns only (no cards, no computed text)."""
    sessions = _sessions_in_time_order(store)
    items = [_search_item(nodes[0], '\n'.join(_turn_text(n) for n in nodes)) for nodes in sessions]
    if len(items) > top_k:
        items = _merge_neighbours(items, top_k)
    # Count the final text, including session date headers introduced by merging.
    if sum(estimate_tokens(item['content']) for item in items) <= budget:
        return items
    focus = None
    if sessions:
        # Long contexts bury single facts: repeat the best retrieval hits, verbatim, nearest the question.
        by_id = {n.id: n for nodes in sessions for n in nodes}
        top = [by_id[a] for a in list(dict.fromkeys(anchors)) if a in by_id][:RECALL_FOCUS_TURNS]
        prefix = 'Most relevant passages (repeated from above):\n'
        kept_focus = []
        for node in top if top_k > 1 else []:
            text = prefix + '\n'.join(_turn_text(n) for n in kept_focus + [node])
            if estimate_tokens(text) <= budget:
                kept_focus.append(node)
        if kept_focus:
            focus = {'id': f'focus-{kept_focus[0].id}', 'content': prefix
                     + '\n'.join(_turn_text(n) for n in kept_focus)}
            budget -= estimate_tokens(focus['content'])
            top_k -= 1
        units = [(s, b) for s, nodes in enumerate(sessions) for b in _blocks(nodes, block_chars)]
        texts = ['\n'.join(_turn_text(n) for n in block) for _, block in units]
        where = {n.id: i for i, (_, block) in enumerate(units) for n in block}
        first = list(dict.fromkeys(where[a] for a in anchors if a in where))
        chosen, used = set(), 0
        for index in first + [i for i in _relevance_order(query, texts) if i not in set(first)]:
            cost = estimate_tokens(texts[index])
            if len(sessions) > top_k:
                # Reserve a date header per block; merging may need fewer, never more.
                stamp = _search_item(units[index][1][0]).get('created_at')
                if stamp:
                    cost += estimate_tokens(f'(conversation on {stamp[:10]})\n')
            if used + cost <= budget:
                chosen.add(index)
                used += cost
        kept = {}
        for index in sorted(chosen):
            kept.setdefault(units[index][0], []).append(units[index][1])
        sessions = [[n for block in kept[s] for n in block] for s in sorted(kept)]
    items = [_search_item(nodes[0], '\n'.join(_turn_text(n) for n in nodes)) for nodes in sessions]
    if len(items) > top_k:
        items = _merge_neighbours(items, top_k)
    return items + ([focus] if focus else [])


def _merge_neighbours(items, top_k):
    # More sessions than result slots: merge neighbours, keeping each session's date in the text.
    merged, size = [], math.ceil(len(items) / top_k)
    for start in range(0, len(items), size):
        group = items[start:start + size]
        parts = [(f"(conversation on {g['created_at'][:10]})\n" if g.get('created_at') else '') + g['content'] for g in group]
        merged.append({**group[0], 'content': '\n'.join(parts)})
    return merged


def _gap_item(store, nodes, query):
    """A computed day/week gap for "how long between A and B" questions, from the retrieved turns' dates."""
    turns = [_source_turn(store, node) for node in nodes]
    fact = gap_fact(query, [(_turn_body(turn), turn.event_time) for turn in turns])
    return {'id': 'computed-gap', 'content': fact} if fact else None


def _turn_body(node: MemNode):
    match = _TURN_HEADER.match(node.content)
    return node.content[match.end():] if match else node.content


def _session_turns(store, session, cache):
    if session not in cache:
        rows = store._conn.execute("SELECT * FROM nodes WHERE content LIKE ?", (f'[AML] session={session} position=%',))
        turns = {}
        for row in rows:
            node = MemNode.from_row(row)
            turns.setdefault(int(_TURN_HEADER.match(node.content)['position']), node)
        cache[session] = turns
    return cache[session]


def _windows(store, nodes, window, whole_session):
    """Group hits into rank-ordered turn spans; a hit overlapping an earlier span joins it."""
    groups, cache = [], {}
    for node in nodes:
        match = _TURN_HEADER.match(node.content)
        if not match:
            groups.append({'anchor': node, 'session': None})
            continue
        session, position = match['session'], int(match['position'])
        turns = _session_turns(store, session, cache)
        low, high = (min(turns), max(turns)) if whole_session else (position - window, position + window)
        for group in groups:
            if group['session'] == session and low <= group['high'] + 1 and high >= group['low'] - 1:
                group['low'], group['high'] = min(low, group['low']), max(high, group['high'])
                break
        else:
            groups.append({'anchor': node, 'session': session, 'low': low, 'high': high})
    items = []
    for group in groups:
        if group['session'] is None:
            items.append(_search_item(group['anchor']))
            continue
        turns = _session_turns(store, group['session'], cache)
        span = [turns[p] for p in sorted(turns) if group['low'] <= p <= group['high']]
        items.append(_search_item(group['anchor'], '\n'.join(_turn_text(n) for n in span)))
    return items


def _search_items(store, nodes, window=0, whole_session=False):
    """Rank-ordered readable memories: claims resolve to their source turn, duplicates dropped.

    ``window`` > 0 widens each hit to that many neighbouring turns of its session;
    ``whole_session`` returns each hit's full session.
    """
    sources, seen = [], set()
    for node in nodes:
        node = _source_turn(store, node)
        if node.id not in seen and node.content.strip():
            seen.add(node.id)
            sources.append(node)
    candidates = _windows(store, sources, window, whole_session) if window or whole_session \
        else [_search_item(node) for node in sources]
    items = []
    for item in candidates:
        if all(item['content'] != kept['content'] for kept in items):
            items.append(item)
    # Contents are now distinct, so containment means a strictly shorter fragment.
    return [item for item in items
            if not any(other is not item and item['content'] in other['content'] for other in items)]


def create_app(root, *, secret, registry=None, jev=None, writer=write_observation,
               window=0, whole_session=False, computed_gaps=False, fact_cards=False,
               full_recall=False, recall_budget_tokens=RECALL_BUDGET_TOKENS,
               recall_block_chars=RECALL_BLOCK_CHARS, deployment_info=None, retriever=None):
    if not secret:
        raise ValueError('AML_MEMORY_SECRET is required')
    backend = MemoryBackend(root, registry or load_registry(), jev or JevClient(), writer,
        window=window, whole_session=whole_session, computed_gaps=computed_gaps, fact_cards=fact_cards,
        full_recall=full_recall, recall_budget_tokens=recall_budget_tokens,
        recall_block_chars=recall_block_chars, retriever=retriever)
    app = FastAPI(title='Home Base AML textual memory adapter')
    app.state.backend = backend

    def authenticate(request: Request):
        authorization = request.headers.get('authorization','')
        scheme, _, token = authorization.partition(' ')
        if scheme.lower() not in ('bearer','token'):
            token = request.headers.get('x-api-key','')
        if not token or not hmac.compare_digest(token.encode(),secret.encode()):
            raise HTTPException(401,'Invalid memory API credential')

    @app.get('/health')
    def health():
        return {'ok':True,'track':'textual','build':{'window':backend.window,'whole_session':backend.whole_session,
            'computed_gaps':backend.computed_gaps,'fact_cards':backend.fact_cards,
            'full_recall':backend.full_recall,'recall_budget_tokens':backend.recall_budget_tokens,
            'recall_block_chars':backend.recall_block_chars,'code_sha':code_sha(), **(deployment_info or {})}}

    @app.post('/add', dependencies=[Depends(authenticate)])
    def add(request: AddRequest):
        return backend.add(request)

    @app.post('/search', dependencies=[Depends(authenticate)])
    def search(request: SearchRequest):
        return backend.search(request)

    return app


def main():
    import uvicorn
    jev = JevClient()
    if not jev.configured():
        raise RuntimeError('Configure TYPESAFE_API_KEY before starting the adapter')
    app = create_app(os.environ.get('AML_MEMORY_DATA_DIR', str(Path.home()/'.homebase/aml-memory')),
        secret=os.environ.get('AML_MEMORY_SECRET',''), jev=jev,
        window=int(os.environ.get('AML_SEARCH_WINDOW', '0')),
        whole_session=os.environ.get('AML_SEARCH_SESSION') == '1',
        computed_gaps=os.environ.get('AML_COMPUTED_GAPS') == '1',
        fact_cards=os.environ.get('AML_FACT_CARDS') == '1',
        full_recall=os.environ.get('AML_FULL_RECALL') == '1',
        recall_budget_tokens=int(os.environ.get('AML_RECALL_BUDGET_TOKENS', str(RECALL_BUDGET_TOKENS))),
        recall_block_chars=int(os.environ.get('AML_RECALL_BLOCK_CHARS', str(RECALL_BLOCK_CHARS))))
    uvicorn.run(app, host=os.environ.get('AML_MEMORY_HOST','127.0.0.1'),
        port=int(os.environ.get('AML_MEMORY_PORT','9091')), workers=1, access_log=False)


if __name__ == '__main__':
    main()
