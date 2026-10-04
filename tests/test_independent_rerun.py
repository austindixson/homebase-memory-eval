import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'verify_rerun', Path(__file__).resolve().parents[1] / 'scripts/verify_rerun.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_official_outcomes_require_complete_unique_successful_ids(tmp_path):
    path = tmp_path / 'scored.jsonl'
    rows = [{'qa_id': q, 'llm_score': score, 'success': True, 'errors': [],
             'prediction_found': True, 'llm_judge': 'refined'}
            for q, score in [('a', 1.0), ('b', 0.0)]]
    write_rows(path, rows)
    assert module.correct_count(path, {'a', 'b'}) == 1
    for invalid in [rows[:1], rows + rows[:1],
                    [dict(rows[0], success=False), rows[1]],
                    [dict(rows[0], llm_score=None), rows[1]],
                    [dict(rows[0], errors=['transport failure']), rows[1]]]:
        write_rows(path, invalid)
        with pytest.raises(ValueError):
            module.correct_count(path, {'a', 'b'})


def test_registered_gate_does_not_round_difference_or_select_best_repeat():
    assert module.compare_counts(925)['pass'] is True
    assert module.compare_counts(951)['pass'] is True
    assert module.compare_counts(924)['pass'] is False
    assert module.compare_counts(952)['pass'] is False
    assert module.compare_counts(879)['exceeds_published_reference'] is False
