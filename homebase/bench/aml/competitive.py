"""Source-only preparation and durable transport for the unmodified Refined scorer.

No answer or judge logic is implemented here. Answers use AML's published prompt;
scoring runs the benchmark's own shell entry point, runtime, parser and summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

from homebase.bench.aml.pipelines import AML_COMMIT, load

CANDIDATE = '745609d5de83648d2183e732f34567bd878e2ea3'
BENCHMARK = '887091190789e8d6760e70b9edd696539923dc4f'
MODEL_REVISION = 'da33cf28f06636847fd9e93e0a03d819b84cb55e'
MODEL_NAME = 'Qwen/Qwen3-14B'
SOURCE_HASHES = {
    'data/raw/locomo_refined.json':'1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17',
    'data/public/questions.jsonl':'4dd84cf65a28ece0ed6b2a1b7b2700ea639f4e7b2c5e1afcc511e47e81732790',
    'data/public/conversations.jsonl':'abbe220013815b1761c07d9ab7d6147ee08fb089f1a8a083fd31188cdcd91444',
    'src/llm_judge.py':'a24d3480e003d985175b1889d6de208585fa694cdb1f02a3739f3cda30a9bb2a',
    'src/llm_judge_runtime.py':'6fa81ca9b0347293f81666b794788b18e70018dadb5793895a8f0e53437d5790',
}
METHODS = ('full-recall', 'retrieval', 'raw-full-context')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def write_frozen(path: Path, value: Any) -> None:
    payload = json_bytes(value)
    write_frozen_bytes(path, payload)


def write_frozen_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'frozen file differs: {path}')
        return
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def verify_sources(root: Path) -> None:
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != BENCHMARK:
        raise ValueError(f'benchmark revision differs: {head}')
    # Check all tracked source, data and scripts, not merely the two judge files.
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    for name in paths:
        expected = subprocess.check_output(['git', '-C', str(root), 'show', f'{BENCHMARK}:{name}'])
        if (root / name).read_bytes() != expected:
            raise ValueError(f'benchmark tracked file modified: {name}')
    for name, expected in SOURCE_HASHES.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f'benchmark hash differs: {name}')


def verify_candidate(root: Path, export_manifest: Path | None = None) -> dict[str, str]:
    if export_manifest is not None:
        export = json.loads(export_manifest.read_text())
        if export.get('original_private_candidate_commit') != CANDIDATE:
            raise ValueError('public mapping names a different candidate')
        hashes = export.get('runtime_hashes', {})
        if not hashes:
            raise ValueError('public runtime mapping is empty')
        for name, expected in hashes.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or not name.startswith('homebase/brain/') or path.suffix != '.py':
                raise ValueError('invalid public runtime path')
            file = root / path
            if not file.is_file() or digest(file.read_bytes()) != expected:
                raise ValueError(f'public runtime differs from original candidate: {name}')
        return hashes
    names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only',
                                     CANDIDATE, 'homebase/brain'], text=True).splitlines()
    hashes = {}
    for name in names:
        if not name.endswith('.py'):
            continue
        expected = subprocess.check_output(['git', '-C', str(root), 'show', f'{CANDIDATE}:{name}'])
        if (root / name).read_bytes() != expected:
            raise ValueError(f'candidate runtime differs from frozen commit: {name}')
        hashes[name] = digest(expected)
    return hashes


def source_sessions(talk: dict) -> list[dict]:
    """Published transcript fields only. QA/evidence never enters Add."""
    result = []
    for session in talk['sessions']:
        stamp = int(datetime.strptime(session['date_time'], '%I:%M %p on %d %B, %Y')
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
        messages = []
        for turn in session['messages']:
            if turn['speaker'] not in (talk['speaker_a'], talk['speaker_b']):
                raise ValueError('unknown conversation speaker')
            text = f"{turn['speaker']}: {turn['text']}"
            if turn.get('images'):
                text += '\n  [images] ' + ', '.join(turn['images'])
            if turn.get('blip_caption'):
                text += '\n  [caption] ' + turn['blip_caption']
            if turn.get('query'):
                text += '\n  [query] ' + turn['query']
            messages.append({'role':'user' if turn['speaker'] == talk['speaker_a'] else 'assistant',
                             'content':text, 'timestamp':stamp})
        result.append({'session_id':f"session-{session['session_index']}", 'messages':messages})
    return result


def candidate_client(store: Path, *, full_recall: bool):
    from fastapi.testclient import TestClient
    from homebase.brain.aml_adapter import create_app
    from homebase.brain.aml_model_free import DisabledModel, retrieve_sources, write_source
    from homebase.brain.config import Registry
    registry = Registry()
    registry.embed_url = ''
    return TestClient(create_app(store, secret='local-benchmark', registry=registry,
                                jev=DisabledModel(), writer=write_source, retriever=retrieve_sources,
                                full_recall=full_recall, recall_budget_tokens=100000,
                                recall_block_chars=6000))


def cap_evidence(text: str, count, budget: int) -> tuple[str, bool]:
    """Deterministic character prefix within the frozen actual-token evidence budget."""
    if count(text) <= budget:
        return text, False
    # Token count isn't strictly monotone at character boundaries. Choose a safe
    # deterministic prefix; this policy is not optimized using benchmark results.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    result = text[:low]
    while count(result) > budget:
        result = result[:-1]
    return result, True


def prepare_rows(talks: list[dict], questions: list[dict], *, client, render, count,
                 budget: int = 100000) -> list[dict]:
    from homebase.bench.aml.render import render_memories
    by_sample = {talk['sample_id']:talk for talk in talks}
    if len(by_sample) != len(talks):
        raise ValueError('duplicate conversation ID')
    if client is not None:
        for talk in talks:
            for session in source_sessions(talk):
                response = client.post('/add', headers={'X-Api-Key':'local-benchmark'}, json={
                    'request_id':f"{talk['sample_id']}:{session['session_id']}",
                    'user_id':talk['sample_id'], **session})
                response.raise_for_status()
    result = []
    for q in questions:
        talk = by_sample[q['sample_id']]
        elapsed = 0.0
        if client is None:
            items = []
            evidence = talk['conversation_history_multimodal_text']
        else:
            started = time.monotonic()
            response = client.post('/search', headers={'X-Api-Key':'local-benchmark'}, json={
                'user_id':q['sample_id'], 'query':q['question'], 'top_k':20})
            response.raise_for_status()
            elapsed = time.monotonic() - started
            items = response.json()['data']
            evidence = render_memories(items)
        evidence, truncated = cap_evidence(evidence, count, budget)
        answer_row = {'question':q['question'], 'speaker_1_name':talk['speaker_a'],
                      'speaker_2_name':talk['speaker_b'], 'speaker_1_memories':evidence,
                      'speaker_2_memories':''}
        # This allowlisted row cannot carry gold answers, QA evidence or judgments.
        prompt = render(answer_row)
        result.append({'qa_id':q['qa_id'], 'sample_id':q['sample_id'], 'category':str(q['category']),
                       'prompt':prompt, 'prompt_sha256':digest(prompt.encode()), 'evidence':evidence,
                       'evidence_tokens':count(evidence), 'prompt_tokens':count(prompt),
                       'truncated':truncated, 'items':items})
        # Latency is measured separately, never mixed into frozen input hashes.
        if client is not None:
            client.benchmark_timings = getattr(client, 'benchmark_timings', []) + [elapsed]
    return result


def validate_model(info: dict) -> dict:
    models = [x for x in info.get('data', []) if x.get('id') == MODEL_REVISION]
    if len(models) != 1:
        raise ValueError('Qwen checkpoint revision is not the frozen revision')
    model = models[0]
    if not model.get('loaded') or model.get('context_length', 0) < 102048:
        raise ValueError('model is unloaded or served context is too small')
    if model.get('meta', {}).get('architecture') != 'qwen3' or model['meta'].get('quantization') != '8-bit':
        raise ValueError('model architecture/precision differs from registration')
    return model


def prediction_map(path: Path, expected_ids: set[str]) -> dict[str, str]:
    result = {}
    for row in load_records(path):
        key = row['qa_id']
        if key not in expected_ids:
            raise ValueError(f'unknown checkpoint ID: {key}')
        if key in result:
            raise ValueError(f'duplicate checkpoint ID: {key}')
        if not isinstance(row['predicted_answer'], str):
            raise ValueError('prediction is not text')
        result[key] = row['predicted_answer']
    return result


def register_gpu_job(root: Path, job: dict) -> None:
    """Append one own job using the existing scheduler lock, with no other edits."""
    if job.get('lane') != 'gpu' or not job.get('name') or not job.get('cmd'):
        raise ValueError('a named GPU job and command are required')
    with (root / 'health.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'jobs.json'
        original_bytes = path.read_bytes()
        config = json.loads(original_bytes)
        existing = [row for row in config['jobs'] if row['name'] == job['name']]
        if existing:
            if existing != [job]:
                raise ValueError('existing job registration differs')
            return
        config['jobs'].append(job)
        temporary = root / f'jobs.{job["name"]}.{os.getpid()}.tmp'
        write_frozen(temporary, config)
        if path.read_bytes() != original_bytes:
            temporary.unlink()
            raise ValueError('shared queue changed during registration')
        temporary.replace(path)


class RecordedTransport:
    """Serialize local inference and retain the first successful response on resume.

    The official judge may issue parallel calls for alternative gold answers. Each
    exact request has its own receipt, so these calls survive scorer restarts without
    new model votes. Model alias routing is the only upstream request transformation.
    """
    def __init__(self, upstream: str, directory: Path, official_root: Path):
        self.upstream = upstream.rstrip('/')
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.phase = 'answer'
        self.lock = threading.Lock()
        sys.path.insert(0, str(official_root / 'src'))
        from llm_judge_runtime import extract_json_object
        self.extract_json_object = extract_json_object

    def complete(self, body: dict) -> dict:
        if body.get('model') != MODEL_NAME or body.get('temperature') != 0:
            raise ValueError('model or temperature differs from registration')
        if body.get('enable_thinking') is not False:
            raise ValueError('thinking must explicitly be disabled')
        if len(body.get('messages', [])) != 1 or body['messages'][0]['role'] != 'user':
            raise ValueError('unexpected prompt wrapper')
        key = digest(json_bytes({'phase':self.phase, 'request':body}))
        receipt = self.directory / self.phase / f'{key}.json'
        with self.lock:
            if receipt.exists():
                return json.loads(receipt.read_text())['response']
            forwarded = {**body, 'model':MODEL_REVISION}
            started = time.monotonic()
            try:
                response = httpx.post(self.upstream + '/chat/completions', json=forwarded, timeout=1200)
                response.raise_for_status()
                payload = response.json()
                content = payload['choices'][0]['message']['content']
                cacheable = True
                if self.phase == 'judge':
                    try:
                        json.loads(self.extract_json_object(str(content or '')))
                    except (ValueError, json.JSONDecodeError):
                        cacheable = False
                record = {'time':time.time(), 'phase':self.phase, 'key':key, 'request':body,
                          'forwarded_request':forwarded, 'response':payload,
                          'elapsed_s':time.monotonic()-started, 'cacheable':cacheable}
                append_record(self.directory / 'attempts.jsonl', record)
                if cacheable:
                    write_frozen(receipt, record)
                return payload
            except Exception as error:
                append_record(self.directory / 'attempts.jsonl', {
                    'time':time.time(), 'phase':self.phase, 'key':key, 'request':body,
                    'elapsed_s':time.monotonic()-started, 'error':type(error).__name__})
                raise

    def serve(self):
        transport = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    self.send_error(404)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    payload = json_bytes(transport.complete(body))
                    self.send_response(200)
                except Exception as error:
                    payload = json_bytes({'error':{'message':type(error).__name__}})
                    self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server


def prepare(args) -> None:
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[3]
    verify_sources(args.refined_root)
    runtime_hashes = verify_candidate(root, getattr(args, 'runtime_manifest', None))
    pipeline = load('locomo-refined', args.aml_root)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    questions = load_records(args.refined_root / 'data/public/questions.jsonl')
    talks = load_records(args.refined_root / 'data/public/conversations.jsonl')
    if len(questions) != 1382 or len({q['qa_id'] for q in questions}) != 1382 or len(talks) != 10:
        raise ValueError('official full-set counts differ')
    if Counter(str(q['category']) for q in questions) != {'1':213,'2':299,'3':68,'4':802}:
        raise ValueError('official category counts differ')
    args.out.mkdir(parents=True, exist_ok=True)
    timings = {}
    for method in METHODS:
        client = None if method == 'raw-full-context' else candidate_client(
            args.out / 'stores' / method, full_recall=method == 'full-recall')
        started = time.monotonic()
        rows = prepare_rows(talks, questions, client=client, render=pipeline.render_answer_prompt, count=count)
        if any(row['prompt_tokens'] + 2048 + 1024 > 131072 for row in rows):
            raise ValueError('prompt exceeds served context with template/output reserve')
        write_frozen(args.out / method / 'inputs.json', rows)
        timings[method] = {'preparation_s':time.monotonic()-started,
                           'search_seconds':getattr(client, 'benchmark_timings', [])}
        print(f'Prepared {method}: {len(rows)} inputs, max evidence {max(r["evidence_tokens"] for r in rows)} tokens', flush=True)
    write_frozen_bytes(args.out / 'questions.jsonl', (args.refined_root / 'data/public/questions.jsonl').read_bytes())
    write_frozen_bytes(args.out / 'tokenizer.json', args.tokenizer.read_bytes())
    write_frozen(args.out / 'manifest.json', {
        'benchmark_commit':BENCHMARK, 'source_hashes':SOURCE_HASHES, 'candidate_commit':CANDIDATE,
        'runtime_hashes':runtime_hashes, 'aml_commit':AML_COMMIT,
        'answer_prompt_sha256':digest(pipeline.OPEN_ENDED_ANSWER_TEMPLATE.encode()),
        'aml_pipeline_sha256':pipeline.AML_PIPELINE_SHA256,
        'tokenizer_sha256':digest(args.tokenizer.read_bytes()), 'question_count':1382,
        'categories':{'1':213,'2':299,'3':68,'4':802}, 'methods':list(METHODS),
        'reader_model':MODEL_NAME, 'judge_model':MODEL_NAME, 'temperature':0,
        'enable_thinking':False, 'reader_max_tokens':2048, 'judge_max_tokens':'server default 1024; official request unchanged',
        'evidence_token_cap':100000, 'top_k':20, 'model_revision':MODEL_REVISION,
        'model_repository':'mlx-community/Qwen3-14B-8bit', 'quantization':'8-bit',
        'served_context':131072, 'prefix_policy':'rank-order evidence character prefix',
        'rendering':'AML CL-bench-style dated bullets; both speakers retain names in one evidence field',
        'multimodal_policy':'official transcript text, image URLs, captions and queries; no image encoder',
        'candidate_gate_correct':880, 'hosted_cost':0, 'railway_cost':0,
        'data_exposure':'original LoCoMo used for prior development; overlapping public Refined set is not pristine held-out',
        'inputs_sha256':{m:digest((args.out / m / 'inputs.json').read_bytes()) for m in METHODS},
    })
    append_record(args.out / 'preparation-timings.jsonl', {'time':time.time(), 'methods':timings})
    (args.out / 'PREPARED').touch()


def selected_methods(manifest: dict, requested: list[str]) -> list[str]:
    if len(set(requested)) != len(requested):
        raise ValueError('duplicate requested methods')
    if not requested or any(m not in manifest['methods'] for m in requested):
        raise ValueError('requested method is not registered in the frozen manifest')
    return requested


def run(args) -> None:
    verify_sources(args.refined_root)
    manifest = json.loads((args.out / 'manifest.json').read_text())
    requested = getattr(args, 'methods', None) or (list(METHODS) if args.controls else ['full-recall'])
    methods = selected_methods(manifest, requested)
    for relative, expected in manifest['runtime_hashes'].items():
        if digest((Path(__file__).resolve().parents[3] / relative).read_bytes()) != expected:
            raise ValueError(f'runtime hash differs: {relative}')
    if digest((args.out / 'questions.jsonl').read_bytes()) != SOURCE_HASHES['data/public/questions.jsonl']:
        raise ValueError('frozen questions differ')
    with (args.out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = validate_model(httpx.get(args.model_url.rstrip('/')+'/models', timeout=20).json())
        # Dynamic provenance is archived separately from immutable protocol/config.
        append_record(args.out / 'model-provenance.jsonl', {'time':time.time(), 'model':model})
        recorded = RecordedTransport(args.model_url, args.out / 'requests', args.refined_root)
        server = recorded.serve()
        base_url = f'http://127.0.0.1:{server.server_port}/v1'
        try:
            for method in methods:
                directory = args.out / method
                rows_path = directory / 'inputs.json'
                if digest(rows_path.read_bytes()) != manifest['inputs_sha256'][method]:
                    raise ValueError('prepared inputs differ')
                rows = json.loads(rows_path.read_text())
                expected = {row['qa_id'] for row in rows}
                predictions = directory / 'predictions.jsonl'
                saved = prediction_map(predictions, expected)
                recorded.phase = 'answer'
                for index, row in enumerate(rows):
                    if row['qa_id'] in saved:
                        continue
                    response = recorded.complete({'model':MODEL_NAME, 'temperature':0, 'max_tokens':2048,
                                                  'enable_thinking':False,
                                                  'messages':[{'role':'user','content':row['prompt']}]})
                    append_record(predictions, {'qa_id':row['qa_id'], 'predicted_answer':
                                               str(response['choices'][0]['message']['content'] or '').strip()})
                    print(f'{method} answers {index+1}/{len(rows)}', flush=True)
                if set(prediction_map(predictions, expected)) != expected:
                    raise ValueError('answer coverage is incomplete')
                (directory / 'ANSWERS_COMPLETE').touch()
                recorded.phase = 'judge'
                env = {**os.environ, 'LOCOMO_PYTHON_BIN':sys.executable,
                       'LOCOMO_QUESTIONS_PATH':str(args.out / 'questions.jsonl'),
                       'LOCOMO_PREDICTIONS_PATH':str(predictions),
                       'LOCOMO_SCORED_PATH':str(directory / 'scored.jsonl'),
                       'LOCOMO_SUMMARY_PATH':str(directory / 'summary.json'),
                       'LOCOMO_MARKDOWN_SUMMARY_PATH':str(directory / 'summary.md'),
                       'EVALUATOR_MODEL':MODEL_NAME, 'EVALUATOR_API_BASE':base_url,
                       'EVALUATOR_API_KEY':'local-benchmark'}
                # Mandatory official path: no substitute judge function or patched parser.
                with (directory / 'official-scorer.log').open('a') as log:
                    process = subprocess.run(['/bin/bash', str(args.refined_root / 'scripts/run_eval.sh'),
                                              '--metrics','llm','f1','bleu','--llm-judge','refined','--concurrency','1'],
                                             env=env, cwd=args.refined_root, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f'official scorer failed: {process.returncode}')
                scored = load_records(directory / 'scored.jsonl')
                if len(scored) != 1382 or {r['qa_id'] for r in scored} != expected:
                    raise ValueError('official scored coverage differs')
                # Ranking gate derives only from the official parsed labels.
                if any(r.get('llm_score') not in (0, 1) for r in scored):
                    raise ValueError('official primary score is missing or non-binary')
                correct = sum(int(r['llm_score']) for r in scored)
                write_frozen(directory / 'gate.json', {'correct':correct,'n':1382,
                    'accuracy':100*correct/1382, 'exceeds_published_reference':correct>=880})
                (directory / 'COMPLETE').touch()
                print(json.dumps({'method':method,'correct':correct,'n':1382}), flush=True)
                if method == 'full-recall' and correct < 880:
                    print('Candidate failed published-reference gate; no control/competitor inference started.', flush=True)
                    break
            (args.out / 'CANDIDATE_EVALUATION_COMPLETE').touch()
        finally:
            server.shutdown()
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run'])
    parser.add_argument('--refined-root', required=True, type=Path)
    parser.add_argument('--aml-root', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--model-url', default='http://127.0.0.1:11240/v1')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Explicit methods already frozen in the execution manifest')
    parser.add_argument('--runtime-manifest', type=Path, help='Original byte hashes for a public export without private Git history')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.refined_root = args.refined_root.resolve()
    if args.action == 'prepare':
        if args.aml_root is None or args.tokenizer is None:
            parser.error('prepare requires --aml-root and --tokenizer')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
