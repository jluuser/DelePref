"""Verify journal ownership, actual-call states and exact artifacts before metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write


def audit(root):
    sys.path.insert(0,str(root/'code'))
    from prediction_journal import PredictionJournal
    cat=root/'reports/recomb_benign_prediction_catalog_20261017'
    base=root/'reports/recomb_benign_structures_20261017'
    out=root/'reports/recomb_benign_prediction_integrity_20261017'
    if out.exists():raise FileExistsError('integrity audit already completed')
    frozen=read(root/'configs/recomb_benign_confirmation_freeze_20261017.json')
    for f in frozen['files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('frozen dependency changed')
    dispatch=read(root/'configs/recomb_benign_structure_dispatch_20261017.json')
    if sha(dispatch['metric_binary']['path'])!=dispatch['metric_binary']['sha256']:raise ValueError('metric binary changed')
    physical={p['task_id']:p for p in read(cat/'physical_requests.json')}
    reuse=read(cat/'compatible_cache.json');wanted=set(physical)-set(reuse)
    snapshot=PredictionJournal(cat/'statistical_protocol.json',base/'prediction_journal.jsonl').snapshot()
    latest={}
    for record in snapshot['attempts']:
        if record['task_id'] not in wanted:raise ValueError('unplanned prediction in journal')
        if record['task_id'] not in latest or latest[record['task_id']]['attempt']<record['attempt']:latest[record['task_id']]=record
    if set(latest)!=wanted:raise ValueError('planned predictions not all attempted; review array state before metrics')
    states={};passed=0;failed=0
    for task,r in latest.items():
        p=physical[task]
        for key,source in [('sequence_sha','sequence_sha256'),('predictor_weights_sha','weights_sha256'),('profile_sha','profile_sha256'),('seed','seed'),('sample','sample')]:
            if r[key]!=p[source]:raise ValueError('journal/prediction identity mismatch')
        record=read(base/'records'/(task+'.json'))
        if r['state']=='succeeded':
            if record['status']!='SUCCEEDED' or sha(r['output_path'])!=r['output_sha'] or record['output_sha256']!=r['output_sha']:
                raise ValueError('successful journal and artifact differ')
            with np.load(r['output_path'],allow_pickle=False) as z:
                if str(z['sequence_sha256'])!=p['sequence_sha256'] or str(z['sequence'])!=p['chains'][0]['sequence']:
                    raise ValueError('output sequence does not match request')
                if z['coordinates'].shape!=(len(p['chains'][0]['sequence']),3):raise ValueError('coordinate dimension mismatch')
                if p['requires_all_atom_output'] and 'chains_json' not in z:raise ValueError('missing application atom output')
            passed+=1
        elif r['state']=='failed':
            if record['status']!='FAILED':raise ValueError('failure state mismatch')
            failed+=1
        else:raise ValueError('unfinished journal entry; do not evaluate as complete')
        states[task]=r['state']
    old=read(cat/'legacy_cache_journal_audit.json')
    if old['reused_cache']!=len(reuse):raise ValueError('legacy cache journal audit differs')
    for item in reuse.values():
        if sha(item['record_path'])!=item['record_sha256'] or sha(item['artifact_path'])!=item['artifact_sha256']:
            raise ValueError('verified cache changed')
    out.mkdir();write(out/'completion.json',dict(status='JOURNAL_SEQUENCE_ARTIFACT_AND_CACHE_IDENTITIES_VERIFIED',
        planned_new_physical=len(wanted),actual_calls=snapshot['consumed'],successful=passed,failed=failed,reused_cache=len(reuse),
        historical_calls=758,total_historical_and_current_calls=758+snapshot['consumed'],all_failures_preserved=True,
        journal_sha256=sha(base/'prediction_journal.jsonl'),dispatch_sha256=sha(root/'configs/recomb_benign_structure_dispatch_20261017.json'),
        audit_source_sha256=sha(Path(__file__))))
    print(json.dumps(read(out/'completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    audit(p.parse_args().root)
