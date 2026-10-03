"""Source-only integration of pinned native competitor retrieval with the common reader.

MemPalace's retrieval algorithm is executed unmodified. Its native retrieval-recall
score is unused; selected session IDs feed the same reader and official scorer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from homebase.bench.aml.competitive import (BENCHMARK, SOURCE_HASHES, cap_evidence,
    digest, json_bytes, source_sessions, verify_sources, write_frozen, write_frozen_bytes)
from homebase.bench.aml.pipelines import load
from homebase.bench.aml.render import render_memories

MEMPALACE_COMMIT = 'a33fdcc98637f0b2c4194b978fec6439fa2a8b93'
METHOD = 'mempalace-hybrid-session'


def source_text(turn: dict) -> str:
    text = turn['text']
    if turn.get('images'):
        text += '\n  [images] ' + ', '.join(turn['images'])
    if turn.get('blip_caption'):
        text += '\n  [caption] ' + turn['blip_caption']
    if turn.get('query'):
        text += '\n  [query] ' + turn['query']
    return text


def mempalace_data(talks: list[dict], questions: list[dict]) -> list[dict]:
    result = []
    for talk in talks:
        source_sessions(talk)  # Validate timestamps and source speakers without using QA.
        conversation = {'speaker_a':talk['speaker_a'], 'speaker_b':talk['speaker_b']}
        for session in talk['sessions']:
            name = f"session_{session['session_index']}"
            conversation[name + '_date_time'] = session['date_time']
            conversation[name] = [{'speaker':t['speaker'], 'text':source_text(t), 'dia_id':t['dia_id']}
                                  for t in session['messages']]
        result.append({'sample_id':talk['sample_id'], 'conversation':conversation,
                       'qa':[{'question':q['question'],'category':int(q['category'])}
                             for q in questions if q['sample_id']==talk['sample_id']]})
    return result


def selected_evidence(talk: dict, ids: list[str], count) -> list[dict]:
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate native session ID')
    sessions = {f"session_{s['session_index']}":s for s in talk['sessions']}
    if any(i not in sessions for i in ids):
        raise ValueError('unknown native session ID')
    items = []
    for key in ids:
        session = sessions[key]
        stamp = datetime.strptime(session['date_time'],'%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc)
        # Exact native build_corpus_from_sessions(session) document formatting.
        text = '\n'.join(f'{t["speaker"]} said, "{source_text(t)}"' for t in session['messages'])
        items.append({'id':key,'created_at':stamp.isoformat(),'content':text})
    return items


def prepare(args) -> None:
    from tokenizers import Tokenizer
    verify_sources(args.refined_root)
    head = subprocess.check_output(['git','-C',str(args.mempalace_root),'rev-parse','HEAD'],text=True).strip()
    if head != MEMPALACE_COMMIT:
        raise ValueError('MemPalace revision differs')
    relative = 'benchmarks/locomo_bench.py'
    benchmark_bytes = (args.mempalace_root / relative).read_bytes()
    original = subprocess.check_output(['git','-C',str(args.mempalace_root),'show',f'{head}:{relative}'])
    if benchmark_bytes != original:
        raise ValueError('native MemPalace benchmark is modified')
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    count = lambda text:len(tokenizer.encode(text).ids)
    if digest(args.tokenizer.read_bytes()) != candidate_manifest['tokenizer_sha256']:
        raise ValueError('tokenizer differs from candidate')
    pipeline = load('locomo-refined',args.aml_root)
    if digest(pipeline.OPEN_ENDED_ANSWER_TEMPLATE.encode()) != candidate_manifest['answer_prompt_sha256']:
        raise ValueError('reader prompt differs from candidate')
    talks = [json.loads(s) for s in (args.refined_root/'data/public/conversations.jsonl').read_text().splitlines()]
    questions = [json.loads(s) for s in (args.refined_root/'data/public/questions.jsonl').read_text().splitlines()]
    data = mempalace_data(talks,questions)
    args.out.mkdir(parents=True,exist_ok=True)
    native_input = args.out/'native-source-only.json'
    write_frozen(native_input,data)
    native_output = args.out/'native-selected.json'
    config = {'method':METHOD,'source_commit':head,'native_benchmark_sha256':digest(benchmark_bytes),
              'mode':'hybrid','granularity':'session','top_k':20,'embed_model':'default',
              'llm_rerank':False,'native_input_sha256':digest(native_input.read_bytes()),
              'source_policy':'Verbatim source plus image URLs/captions/queries; native speaker said formatting; no QA gold/evidence, summaries or date notes',
              'reader_protocol_sha256':digest(json_bytes(candidate_manifest))}
    write_frozen(args.out/'competitor-registration.json',config)
    if not native_output.exists():
        # No edits to upstream retrieval code; no gold answers or annotations in its input.
        with (args.out/'native-retrieval.log').open('a') as log:
            subprocess.run([sys.executable,str(args.mempalace_root/relative),str(native_input),
                            '--top-k','20','--mode','hybrid','--granularity','session',
                            '--embed-model','default','--out',str(native_output)],
                           check=True,stdout=log,stderr=subprocess.STDOUT)
    selections = json.loads(native_output.read_text())
    ordered = [q for t in talks for q in questions if q['sample_id']==t['sample_id']]
    if len(selections) != len(ordered) or len(ordered)!=1382:
        raise ValueError('native selection coverage differs')
    by_sample = {t['sample_id']:t for t in talks}
    rows = []
    for selected,q in zip(selections,ordered):
        if (selected['sample_id'],selected['question']) != (q['sample_id'],q['question']):
            raise ValueError('native selected question order/identity differs')
        items = selected_evidence(by_sample[q['sample_id']],selected['retrieved_ids'],count)
        evidence,truncated = cap_evidence(render_memories(items),count,100000)
        t = by_sample[q['sample_id']]
        prompt = pipeline.render_answer_prompt({'question':q['question'],'speaker_1_name':t['speaker_a'],
                      'speaker_2_name':t['speaker_b'],'speaker_1_memories':evidence,'speaker_2_memories':''})
        if count(prompt)+2048+1024 > 131072:
            raise ValueError('prompt exceeds served context')
        rows.append({'qa_id':q['qa_id'],'sample_id':q['sample_id'],'category':str(q['category']),
                     'evidence':evidence,'items':items,'evidence_tokens':count(evidence),
                     'prompt':prompt,'prompt_tokens':count(prompt),'prompt_sha256':digest(prompt.encode()),
                     'truncated':truncated})
    inputs = args.out/METHOD/'inputs.json'
    write_frozen(inputs,rows)
    write_frozen_bytes(args.out/'questions.jsonl',(args.refined_root/'data/public/questions.jsonl').read_bytes())
    write_frozen_bytes(args.out/'tokenizer.json',args.tokenizer.read_bytes())
    manifest = {**candidate_manifest,'methods':[METHOD],'runtime_hashes':{},
                'inputs_sha256':{METHOD:digest(inputs.read_bytes())},'competitor':config}
    write_frozen(args.out/'manifest.json',manifest)
    (args.out/'PREPARED').touch()
    print(f'Prepared {METHOD}: {len(rows)} inputs; native retrieval only, no reader or judge inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['refined-root','mempalace-root','aml-root','tokenizer','candidate-manifest','out']:
        parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    for name,value in vars(args).items():
        setattr(args,name,value.resolve())
    prepare(args)


if __name__=='__main__':
    main()
