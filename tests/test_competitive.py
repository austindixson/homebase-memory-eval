"""Integrity tests for the official Refined run; no model inference."""
import json
from pathlib import Path
import asyncio
import sys

import pytest

from homebase.bench.aml.competitive import (
    append_record, candidate_client, load_records, prepare_rows,
    prediction_map, source_sessions, validate_model, write_frozen,
    register_gpu_job,
)


def transcript():
    return {'sample_id': 'fixture', 'speaker_a': 'Alice', 'speaker_b': 'Bob',
            'conversation_history_multimodal_text': 'Alice: Original history.',
            'sessions': [{'session_index': 1, 'date_time': '1:56 pm on 8 May, 2023',
                          'messages': [{'speaker': 'Alice', 'role': 'user',
                                        'text': 'My dog is Biscuit.',
                                        'images': ['https://example.test/dog.png'],
                                        'blip_caption': 'A brown dog.', 'query': 'family dog'}]}]}


def questions():
    return [{'qa_id': 'fixture#q0000', 'sample_id': 'fixture', 'question': 'What is my dog called?',
             'answer': ['GOLD_CANARY'], 'evidence': ['EVIDENCE_CANARY'], 'category': '4'}]


def test_source_only_caption_query_date_and_isolation(tmp_path):
    talk = transcript()
    sessions = source_sessions(talk)
    body = json.dumps(sessions)
    assert 'GOLD_CANARY' not in body and 'EVIDENCE_CANARY' not in body
    assert 'Biscuit' in body and '[caption] A brown dog.' in body and '[query] family dog' in body
    assert sessions[0]['messages'][0]['timestamp'] == 1683554160000
    client = candidate_client(tmp_path / 'store', full_recall=True)
    def render(row):
        assert not {'answer', 'gold_answer', 'evidence'} & row.keys()
        return row['question'] + '\n' + row['speaker_1_memories']
    rows = prepare_rows([talk], questions(), client=client, render=render, count=len, budget=10000)
    assert len(rows) == 1 and rows[0]['qa_id'] == 'fixture#q0000'
    assert 'Biscuit' in rows[0]['prompt'] and 'GOLD_CANARY' not in json.dumps(rows)
    result = client.post('/search', headers={'X-Api-Key': 'local-benchmark'},
                         json={'user_id': 'another-user', 'query': 'dog', 'top_k': 20})
    assert result.json()['data'] == []
    # Idempotent Add and frozen preparation on retry.
    assert rows == prepare_rows([talk], questions(), client=client, render=render, count=len, budget=10000)


def test_prefix_cap_and_raw_history_without_date_notes(tmp_path):
    talk = transcript()
    rows = prepare_rows([talk], questions(), client=None, render=lambda r:r['speaker_1_memories'],
                        count=len, budget=10)
    assert rows[0]['prompt'] == 'Alice: Ori'
    assert rows[0]['evidence_tokens'] == 10
    assert rows[0]['truncated'] is True


def test_frozen_files_and_prediction_checkpoint_fail_closed(tmp_path):
    path = tmp_path / 'file.json'
    write_frozen(path, {'a': 1})
    write_frozen(path, {'a': 1})
    with pytest.raises(ValueError, match='differs'):
        write_frozen(path, {'a': 2})
    preds = tmp_path / 'predictions.jsonl'
    append_record(preds, {'qa_id':'q1', 'predicted_answer':'first'})
    assert prediction_map(preds, {'q1'}) == {'q1': 'first'}
    append_record(preds, {'qa_id':'q1', 'predicted_answer':'replacement'})
    with pytest.raises(ValueError, match='duplicate'):
        prediction_map(preds, {'q1'})
    with pytest.raises(ValueError, match='unknown'):
        prediction_map(preds, {'q2'})


def test_checkpoint_revision_and_quantization_verified():
    info = {'data':[{'id':'da33cf28f06636847fd9e93e0a03d819b84cb55e', 'loaded':True,
                     'context_length':131072, 'meta':{'architecture':'qwen3','quantization':'8-bit'}}]}
    assert validate_model(info)['id'] == info['data'][0]['id']
    for field, value in [('id','different'), ('loaded',False), ('context_length',32768)]:
        changed = json.loads(json.dumps(info))
        changed['data'][0][field] = value
        with pytest.raises(ValueError):
            validate_model(changed)


def test_truncated_checkpoint_is_not_silently_ignored(tmp_path):
    path = tmp_path / 'broken.jsonl'
    path.write_text('{"qa_id":"q1"}\n{"qa_id":')
    with pytest.raises(json.JSONDecodeError):
        load_records(path)


def test_queue_registration_preserves_every_other_job_and_is_idempotent(tmp_path):
    before = {'jobs':[{'name':'prior-job','lane':'gpu','enabled':True,'cmd':'original command',
                       'run_dir':'runs/prior','needs':['model']}], 'services':{'model':{'port':11240}}}
    (tmp_path/'jobs.json').write_text(json.dumps(before))
    job = {'name':'locomo-refined-full','lane':'gpu','run_dir':'runs/own',
           'done_file':'CANDIDATE_EVALUATION_COMPLETE','needs':['model'], 'cmd':'own command'}
    register_gpu_job(tmp_path, job)
    after = json.loads((tmp_path/'jobs.json').read_text())
    assert after['jobs'][:-1] == before['jobs']
    assert after['services'] == before['services'] and after['jobs'][-1] == job
    register_gpu_job(tmp_path, job)
    assert json.loads((tmp_path/'jobs.json').read_text()) == after
    with pytest.raises(ValueError, match='differs'):
        register_gpu_job(tmp_path, {**job, 'cmd':'changed after registration'})


def test_official_scorer_uses_gold_alternatives_and_durable_unmodified_requests(tmp_path, monkeypatch):
    import httpx
    from homebase.bench.aml.competitive import RecordedTransport, MODEL_NAME, MODEL_REVISION
    official = Path('/tmp/clm-locomo-refined-verification-20261002')
    if not official.exists():
        pytest.skip('pinned public scorer checkout not present')
    sys.path.insert(0, str(official / 'src'))
    from evaluate import evaluate_public_predictions
    from llm_judge import JUDGE_PROMPTS
    calls = []
    def mock_post(url, *, json, timeout):
        calls.append(json)
        assert json['model'] == MODEL_REVISION
        assert json['temperature'] == 0 and json['enable_thinking'] is False
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'id':'fixture', 'object':'chat.completion', 'created':0, 'model':MODEL_REVISION,
            'choices':[{'index':0,'finish_reason':'stop','message':
                        {'role':'assistant','content':'{"label":"CORRECT"}'}}]})
    monkeypatch.setattr(httpx, 'post', mock_post)
    transport = RecordedTransport('http://fixture.invalid/v1', tmp_path/'receipts', official)
    transport.phase = 'judge'
    server = transport.serve()
    question_path, prediction_path = tmp_path/'q.jsonl', tmp_path/'p.jsonl'
    append_record(question_path, {'qa_id':'fixture#q0000','sample_id':'fixture', 'category':'4',
                                 'question':'Fixture question?', 'answer':['first gold','second gold']})
    append_record(prediction_path, {'qa_id':'fixture#q0000','predicted_answer':'fixture paraphrase'})
    async def score():
        return await evaluate_public_predictions(questions_path=question_path, predictions_path=prediction_path,
            metrics=['llm','f1','bleu'], evaluator_model=MODEL_NAME,
            evaluator_base_url=f'http://127.0.0.1:{server.server_port}/v1', evaluator_api_key='fixture', concurrency=1)
    try:
        result = asyncio.run(score())
        assert result[0]['llm_score'] == 1
        assert len(calls) == 2
        assert {call['messages'][0]['content'] for call in calls} == {
            JUDGE_PROMPTS['refined'].format(question='Fixture question?', gold_answer=gold,
                                           generated_answer='fixture paraphrase')
            for gold in ['first gold','second gold']}
        assert asyncio.run(score()) == result
        assert len(calls) == 2  # No new judge votes when the official scorer resumes.
    finally:
        server.shutdown()
        server.server_close()
