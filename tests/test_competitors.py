import copy
import pytest

from homebase.bench.aml.competitors import mempalace_data, selected_evidence
from homebase.bench.aml.competitive import CANDIDATE, digest, selected_methods, verify_candidate


def talk():
    return {'sample_id':'c', 'speaker_a':'Ann', 'speaker_b':'Bob', 'qa':{'answer':'DO_NOT_COPY'},
            'sessions':[{'session_index':1, 'date_time':'1:00 pm on 8 May, 2023',
                         'messages':[{'dia_id':'D1:1', 'speaker':'Ann', 'text':'Source.',
                                      'images':['https://example.test/image'], 'blip_caption':'Image caption',
                                      'query':'Image query', 'answer':'DO_NOT_COPY'}]}]}


def test_native_ingestion_is_source_only_and_questions_have_no_gold():
    questions=[{'qa_id':'c#q0','sample_id':'c','question':'Question?', 'category':'4',
                'answer':['DO_NOT_COPY'],'evidence':['DO_NOT_COPY']}]
    original=copy.deepcopy(questions)
    data=mempalace_data([talk()],questions)
    assert questions==original
    assert data[0]['qa']==[{'question':'Question?','category':4}]
    assert 'DO_NOT_COPY' not in str(data)
    text=data[0]['conversation']['session_1'][0]['text']
    assert 'Image caption' in text and 'Image query' in text and 'https://example.test/image' in text


def test_selected_native_ids_preserve_order_and_time_and_reject_unknown():
    result=selected_evidence(talk(),['session_1'],lambda x:len(x.split()))
    assert result[0]['content'].startswith('Ann said, "Source.')
    assert result[0]['created_at']=='2023-05-08T13:00:00+00:00'
    with pytest.raises(ValueError,match='unknown'):
        selected_evidence(talk(),['session_2'],lambda x:len(x.split()))
    with pytest.raises(ValueError,match='duplicate'):
        selected_evidence(talk(),['session_1','session_1'],lambda x:len(x.split()))


def test_run_cannot_infer_for_unregistered_or_duplicate_methods():
    manifest={'methods':['mempalace-hybrid-session']}
    assert selected_methods(manifest,['mempalace-hybrid-session'])==['mempalace-hybrid-session']
    with pytest.raises(ValueError,match='registered'):
        selected_methods(manifest,['full-recall'])
    with pytest.raises(ValueError,match='duplicate'):
        selected_methods(manifest,['mempalace-hybrid-session']*2)


def test_public_runtime_mapping_rejects_tampering_and_missing_files(tmp_path):
    import json
    file=tmp_path/'homebase/brain/aml_model_free.py'
    file.parent.mkdir(parents=True)
    file.write_text('immutable candidate source')
    manifest=tmp_path/'runtime-export.json'
    hashes={'homebase/brain/aml_model_free.py':digest(file.read_bytes())}
    manifest.write_text(json.dumps({'original_private_candidate_commit':CANDIDATE,'runtime_hashes':hashes}))
    assert verify_candidate(tmp_path,manifest)==hashes
    file.write_text('modified')
    with pytest.raises(ValueError,match='runtime'):
        verify_candidate(tmp_path,manifest)
    manifest.write_text(json.dumps({'original_private_candidate_commit':'wrong','runtime_hashes':hashes}))
    with pytest.raises(ValueError,match='candidate'):
        verify_candidate(tmp_path,manifest)
