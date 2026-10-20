"""Count frozen development selections; descriptive, without reselecting models."""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from common import AUTHORIZED_ROOT, read_jsonl, sha_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != AUTHORIZED_ROOT.resolve():
        raise ValueError('unexpected experiment root')
    path = root / 'reports/development_review/report.json'
    review = json.loads(path.read_text())
    if review['status'] != 'DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW':
        raise ValueError('complete validated development review required')
    decisions = {}
    for arm, selected in review['selected'].items():
        directory = Path(selected['validation_output'])
        source = directory / 'decisions.jsonl'
        if sha_file(source) != selected['output_hashes']['decisions.jsonl']:
            raise ValueError('selected decisions changed')
        rows = read_jsonl(source)
        index = {(r['protein_id'], str(r['budget']), r['generator']): r for r in rows}
        if len(index) != len(rows):
            raise ValueError('duplicate decision key')
        decisions[arm] = index
    expected = set(decisions['C'])
    if any(set(rows) != expected for rows in decisions.values()):
        raise ValueError('decision rosters differ')
    comparisons = {}
    for other in 'BSAD':
        summary = collections.defaultdict(lambda: collections.Counter(
            {'WT': 0, 'same_candidate': 0, 'different_candidate': 0,
             'C_T_better': 0, 'C_T_worse': 0, 'T_tie': 0, 'overlapping_unknown_bounds': 0}))
        differences = []
        for key in sorted(expected):
            c, b = decisions['C'][key], decisions[other][key]
            counts = summary[(key[1], key[2])]
            counts['WT'] += 1
            same = c['selected'] == b['selected']
            counts['same_candidate' if same else 'different_candidate'] += 1
            lo, hi = c['T_lower'] - b['T_upper'], c['T_upper'] - b['T_lower']
            counts['C_T_better' if lo > 0 else 'C_T_worse' if hi < 0 else
                   'T_tie' if lo == hi == 0 else 'overlapping_unknown_bounds'] += 1
            if not same:
                differences.append({'protein_id': key[0], 'budget': key[1], 'generator': key[2],
                                    'C_selected': c['selected'], 'other_selected': b['selected'],
                                    'T_difference_bounds': [lo, hi],
                                    'coverage_difference': c['C'] - b['C']})
        comparisons['C-' + other] = {
            'strata': [{'budget': k[0], 'generator': k[1], **v} for k, v in sorted(summary.items())],
            'changed_selections': differences}
    result = {'status': 'FROZEN_DEVELOPMENT_CHOICE_AGREEMENT_DIAGNOSTIC',
              'development_review_sha256': sha_file(path), 'implementation_sha256': sha_file(__file__),
              'WT': len(review['validation_wt_components']), 'comparisons': comparisons,
              'interpretation': 'Descriptive historical selection agreement; source and mixed pools overlap, so strata counts are not independent samples. No model or candidate reselection.',
              'GPU_calls': 0, 'structure_calls': 0}
    write_json(root / 'reports/decision_agreement_diagnostic.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
