"""No inference or embeddings: preserve source turns and rank them with BM25."""
from homebase.brain.aml_adapter import _TURN_HEADER, _relevance_order, _sessions_in_time_order, _turn_text


class DisabledModel:
    """Fail loudly if a future change accidentally routes inference into this profile."""
    def system_one(self, *args, **kwargs):
        raise RuntimeError('LLM components are disabled in the model-free AML profile')


def write_source(content, store, client, registry, **kwargs):
    match = _TURN_HEADER.match(content)
    if match is None:
        raise ValueError('model-free Add accepts only source turn records')
    return store.add_node(content, source_position=int(match['position']),
                          source_role=match['role'], **kwargs)


def retrieve_sources(query, store, client, registry, *, top_k, run_prove=False):
    nodes = [node for session in _sessions_in_time_order(store) for node in session]
    order = _relevance_order(query, [_turn_text(node) for node in nodes])
    return [nodes[i] for i in order[:top_k]]
