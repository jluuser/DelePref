"""Confirm official score-table labels and sequences match the frozen cohort."""
import csv
import json
from pathlib import Path
LOCAL=Path(__file__).resolve().parents[1]


def main():
    rows=[json.loads(l) for l in (LOCAL/'caches/recomb_expanded_20261017/rows.jsonl').read_text().splitlines()]
    tables={}
    for assay in sorted({r['assay_id'] for r in rows}):
        with (LOCAL/'caches/recomb_public_baselines_20261017'/(assay+'.csv')).open() as f:
            tables[assay]={r['mutated_sequence']:r for r in csv.DictReader(f)}
    deltas=[];binary=0
    for r in rows:
        source=tables[r['assay_id']][r['mutant_sequence']]
        deltas.append(abs(float(source['DMS_score'])-r['measurement']))
        binary+=int(int(float(source['DMS_score_bin']))!=r['official_binary'])
    assert binary==0 and max(deltas)<1e-7
    result=dict(status='PUBLIC_SCORE_SEQUENCE_AND_LABEL_IDENTITY_AUDITED',observations=len(rows),
        measured_value_max_abs_difference=max(deltas),binary_disagreements=binary,
        exact_assay_and_complete_mutant_sequence_join=True,
        direction='Official merged scores already transformed by pinned merge.py/config.json; no additional score sign choice')
    path=LOCAL/'reports/recomb_experiment_design_20261017/public_score_identity_audit.json'
    path.write_text(json.dumps(result,indent=2)+'\n',newline='\n');print(json.dumps(result))


if __name__=='__main__':main()
