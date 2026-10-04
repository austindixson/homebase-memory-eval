"""Check saved official outcomes against the registered reproduction gate; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

QUESTION_SHA256 = '4dd84cf65a28ece0ed6b2a1b7b2700ea639f4e7b2c5e1afcc511e47e81732790'
ORIGINAL_CORRECT = 938
N = 1382


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_ids(records, expected):
    ids = [row['qa_id'] for row in records]
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError('missing, duplicate or unknown question IDs')


def correct_count(path, expected):
    records = rows(path)
    unique_ids(records, expected)
    for row in records:
        if (row.get('success') is not True or row.get('errors') != []
                or row.get('prediction_found') is not True
                or row.get('llm_judge') != 'refined'
                or isinstance(row.get('llm_score'), bool)
                or row.get('llm_score') not in (0, 1)):
            raise ValueError('incomplete, unsuccessful or invalid official judgment')
    return sum(int(row['llm_score']) for row in records)


def compare_counts(correct):
    difference = 100 * abs(correct - ORIGINAL_CORRECT) / N
    reference = correct >= 880
    return {'n': N, 'original_correct': ORIGINAL_CORRECT, 'rerun_correct': correct,
            'original_score_percent': 100 * ORIGINAL_CORRECT / N,
            'rerun_score_percent': 100 * correct / N,
            'absolute_difference_pp': difference,
            'exceeds_published_reference': reference,
            'within_one_percentage_point': difference <= 1,
            'pass': reference and difference <= 1,
            'operator_independence': 'must be established separately; cannot be inferred from files'}


def verify(out):
    questions = out / 'questions.jsonl'
    if hashlib.sha256(questions.read_bytes()).hexdigest() != QUESTION_SHA256:
        raise ValueError('questions differ from the frozen official file')
    expected = {row['qa_id'] for row in rows(questions)}
    if len(expected) != N:
        raise ValueError('official question count differs')
    directory = out / 'full-recall'
    for path in [out / 'PREPARED', directory / 'ANSWERS_COMPLETE', directory / 'COMPLETE',
                 out / 'CANDIDATE_EVALUATION_COMPLETE']:
        if not path.is_file():
            raise ValueError(f'missing completion marker: {path.name}')
    predictions = rows(directory / 'predictions.jsonl')
    unique_ids(predictions, expected)
    scored = rows(directory / 'scored.jsonl')
    correct = correct_count(directory / 'scored.jsonl', expected)
    answers = {row['qa_id']: row['predicted_answer'] for row in predictions}
    if any(row.get('predicted_answer') != answers[row['qa_id']] for row in scored):
        raise ValueError('official judgments do not match saved predictions')
    manifest = json.loads((out / 'manifest.json').read_text())
    inputs = directory / 'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != manifest['inputs_sha256']['full-recall']:
        raise ValueError('prepared candidate inputs differ from the execution manifest')
    return compare_counts(correct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({'complete': False, 'pass': False, 'error': str(error)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({'complete': True, **result}, indent=2))
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
