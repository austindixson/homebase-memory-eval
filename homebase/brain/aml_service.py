"""Fixed Search/full-recall deployment; one process and a persistent SQLite volume."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

from homebase.brain.aml_adapter import create_app
from homebase.brain.config import Registry, load_registry
from homebase.brain.jev_client import DEFAULT_JEV_BASE, DEFAULT_JEV_MODEL, JevClient
from homebase.brain.mem.write import write_observation


def _required(name):
    value = os.environ.get(name, '')
    if not value.strip():
        raise ValueError(f'{name} is required')
    return value


def _integer(name, default, minimum, maximum=None):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f'{name} must be an integer') from None
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside the allowed range')
    return value


def build_app():
    profile = os.environ.get('AML_MEMORY_PROFILE', 'jev')
    if profile not in ('jev', 'model-free'):
        raise ValueError('AML_MEMORY_PROFILE must be jev or model-free')
    secret = _required('AML_MEMORY_SECRET')
    key = _required('TYPESAFE_API_KEY') if profile == 'jev' else None
    root = Path(_required('AML_MEMORY_DATA_DIR'))
    if not root.is_absolute():
        raise ValueError('AML_MEMORY_DATA_DIR must be absolute and on a persistent volume')
    commit = _required('AML_SOURCE_COMMIT')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('AML_SOURCE_COMMIT must be a fixed 40-character commit SHA')
    full_recall = os.environ.get('AML_FULL_RECALL', '0')
    if full_recall not in ('0', '1'):
        raise ValueError('AML_FULL_RECALL must be 0 or 1')
    for name in ('AML_COMPUTED_GAPS', 'AML_FACT_CARDS', 'AML_SEARCH_WINDOW', 'AML_SEARCH_SESSION'):
        if os.environ.get(name, '0') != '0':
            raise ValueError(f'{name} must be 0 for this comparison')
    # Both profiles disable embeddings and reject runtime model/registry changes.
    for name in ('BRAIN_REGISTRY', 'BRAIN_EMBED_URL', 'HOMEBASE_MEM_EMBED_URL',
                 'TYPESAFE_BASE_URL', 'TYPESAFE_MODEL', 'JEV_BASE_URL', 'JEV_MODEL'):
        if os.environ.get(name):
            raise ValueError(f'{name} is not allowed in this fixed deployment')
    for name in os.environ:
        if name.startswith('HOMEBASE_MEM_') and os.environ[name]:
            raise ValueError(f'{name} is not allowed in this fixed deployment')
    budget = _integer('AML_RECALL_BUDGET_TOKENS', 100_000, 1, 100_000)
    blocks = _integer('AML_RECALL_BLOCK_CHARS', 6_000, 1)
    registry = load_registry() if profile == 'jev' else Registry()
    registry.embed_url = ''
    if profile == 'model-free':
        from homebase.brain.aml_model_free import DisabledModel, retrieve_sources, write_source
        client, writer, retriever = DisabledModel(), write_source, retrieve_sources
    else:
        client = JevClient(api_key=key, base_url=DEFAULT_JEV_BASE, model=DEFAULT_JEV_MODEL)
        writer, retriever = write_observation, None
    fingerprint = hashlib.sha256(json.dumps(asdict(registry), sort_keys=True).encode()).hexdigest()
    app = create_app(root, secret=secret, registry=registry, jev=client, writer=writer, retriever=retriever,
                     full_recall=full_recall == '1', recall_budget_tokens=budget,
                     recall_block_chars=blocks,
                     deployment_info={'source_commit': commit, 'registry_sha': fingerprint,
                                      'profile': profile,
                                      'retrieval': 'bm25-source-turns' if profile == 'model-free' else 'jev-memory',
                                      'llm_model': DEFAULT_JEV_MODEL if profile == 'jev' else None,
                                      'embedding_model': None})
    return app


def main():
    import uvicorn
    port = _integer('PORT', 9091, 1, 65535)
    uvicorn.run(build_app(), host='0.0.0.0', port=port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
