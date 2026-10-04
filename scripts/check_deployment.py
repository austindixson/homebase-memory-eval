"""Synthetic API checks. Never prints the service credential or uses model APIs."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--config', help='Private service JSON; alternatively set AML_MEMORY_SECRET')
    parser.add_argument('--out', required=True)
    parser.add_argument('--commit', default='4aec8864ec39166681e1ff876b9aead390bbcd91')
    parser.add_argument('--resume', help='Verify persistence using a previous check receipt')
    args = parser.parse_args()
    secret = (json.loads(Path(args.config).read_text())['AML_MEMORY_SECRET']
              if args.config else os.environ['AML_MEMORY_SECRET'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    user = 'deployment-check-' + uuid.uuid4().hex
    result = {'pass': False, 'base': args.base, 'source_commit': args.commit,
              'test_user': user, 'checks': [],
              'pass_rule': 'All assertions pass; no quality score or model calls.'}
    out.write_text(json.dumps(result, indent=2))
    with httpx.Client(base_url=args.base.rstrip('/') + '/', timeout=60) as client:
        health = client.get('health')
        health.raise_for_status()
        build = health.json()['build']
        assert build['source_commit'] == args.commit
        assert build['profile'] == 'model-free' and build['full_recall']
        assert build['llm_model'] is None and build['embedding_model'] is None
        result['checks'].append('health and fixed model-free provenance')
        query = {'query': 'Find verification facts', 'user_id': user, 'top_k': 100}
        assert client.post('search', json=query).status_code == 401
        assert client.post('search', json=query,
                           headers={'Authorization': 'Token invalid'}).status_code == 401
        client.headers['Authorization'] = 'Token ' + secret
        result['checks'].append('missing and invalid credentials rejected')
        latencies = []
        for chunk in range(5):
            body = {'request_id': user + ':' + str(chunk), 'user_id': user,
                    'session_id': user + ':session', 'messages': [
                        {'role': 'user', 'content': f'Synthetic fact {i}: item code cobalt-{i:03d}.',
                         'timestamp': 1704067200000 + i * 1000}
                        for i in range(chunk * 20, (chunk + 1) * 20)]}
            started = time.perf_counter()
            response = client.post('add', json=body)
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)
            assert response.json() == {'success': True, **{
                k: body[k] for k in ('request_id', 'user_id', 'session_id')}}
        assert client.post('add', json=body).json() == response.json()
        assert client.post('add', json={**body, 'messages': [
            {'role': 'user', 'content': 'Conflicting retry'}]}).status_code == 422
        result['checks'].append('Add echo, idempotency and conflict rejection')
        started = time.perf_counter()
        response = client.post('search', json=query)
        response.raise_for_status()
        result['search_seconds'] = time.perf_counter() - started
        items = response.json()['data']
        content = '\n'.join(item['content'] for item in items)
        assert 0 < len(items) <= 100
        assert all(f'cobalt-{i:03d}' in content for i in range(100))
        assert content.count('cobalt-099') == 1
        assert all(item['id'] and item['content'] for item in items)
        assert len(client.post('search', json={**query, 'top_k': 1}).json()['data']) <= 1
        assert client.post('search', json={**query, 'user_id': user + '-other'}).json() == {'data': []}
        result['checks'].extend(['100/100 original facts preserved', 'top_k bound', 'user isolation'])
        if args.resume:
            previous = json.loads(Path(args.resume).read_text())
            prior = client.post('search', json={**query, 'user_id': previous['test_user']})
            prior.raise_for_status()
            text = '\n'.join(item['content'] for item in prior.json()['data'])
            assert all(f'cobalt-{i:03d}' in text for i in range(100))
            result['checks'].append('previous data persisted across service restart')
        result.update({'pass': True, 'add_seconds': latencies, 'source_turns': 100,
                       'returned_items': len(items), 'build': build})
        out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
