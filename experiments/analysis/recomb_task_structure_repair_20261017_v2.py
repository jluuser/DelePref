"""Repair only ledger variant metadata; preserve failed freeze and all scientific inputs."""
import json
from pathlib import Path
import sys
import tempfile
from recomb_task_budget_20261017 import ROOT,read,write
from delepref_v8_runtime import sha
sys.path.insert(0,str(ROOT/'code'))
from budget_ledger import BudgetLedger
OLD='reports/recomb_task_structure_20261017'
NEW='reports/recomb_task_structure_20261017_v2'
FREEZE='configs/recomb_task_structure_freeze_20261017_v2.json'


def main():
    if (ROOT/FREEZE).exists():raise FileExistsError('v2 already frozen')
    old_freeze=read('configs/recomb_task_structure_freeze_20261017.json')
    for f in old_freeze['files']:assert sha(f['path'])==f['sha256']
    ledger=ROOT/'reports/recomb_task_structures_20261017/budget_ledger.jsonl'
    assert not ledger.exists() or not ledger.read_bytes(), 'audit actual old attempts before repair'
    old_requests=read(OLD+'/physical_requests.json');new_requests=[]
    for r in old_requests:
        assert r['wt_variant']=='edited'
        fixed=dict(r,wt_variant='variant')
        assert {k:v for k,v in r.items() if k!='wt_variant'}=={k:v for k,v in fixed.items() if k!='wt_variant'}
        new_requests.append(fixed)
    for name in ['logical_requests.json','reused_old_requests.json','statistical_protocol.json','semantic_audit.json']:
        write(NEW+'/'+name,read(OLD+'/'+name))
    write(NEW+'/physical_requests.json',new_requests)
    with tempfile.TemporaryDirectory(prefix='recomb_v2_ledger_contract_') as directory:
        test=BudgetLedger(ROOT/NEW/'statistical_protocol.json',Path(directory)/'dry_test_only.jsonl')
        for r in new_requests:
            test.reserve(task_id=r['task_id'],sequence_sha=r['sequence_sha256'],wt_variant=r['wt_variant'],
                predictor_weights_sha=r['weights_sha256'],profile_sha=r['profile_sha256'],seed=r['seed'],
                category=r['category'],job_id='CPU_DRY_CONTRACT_TEST_NOT_A_PREDICTION',sample=r['sample'])
        dry=test.snapshot();assert len(dry['attempts'])==117 and all(r['state']=='reserved' for r in dry['attempts'])
    repair=dict(status='ALL_117_LEDGER_REQUEST_CONTRACTS_PASS_BEFORE_GPU',
        reason='Original metadata used edited, while ledger accepts WT or variant; failure occurred before reserve/start/forward',
        failed_array='1117520',original_freeze_sha256=sha(ROOT/'configs/recomb_task_structure_freeze_20261017.json'),
        actual_new_structure_calls_before_repair=0,dry_contract_reservations=117,dry_GPU_forwards=0,
        changed_scientific_inputs=False,all_sequence_masks_profiles_weights_seeds_identical=True,old_logs_and_freeze_preserved=True)
    write(NEW+'/technical_repair_audit.json',repair)
    files=old_freeze['files']+[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in
        ['analysis/recomb_task_structure_repair_20261017_v2.py','analysis/recomb_task_fold_worker_20261017_v2.py',
         'analysis/recomb_task_structure_evaluate_20261017_v2.py']+
        [NEW+'/'+n for n in ['logical_requests.json','physical_requests.json','reused_old_requests.json','statistical_protocol.json',
                             'semantic_audit.json','technical_repair_audit.json']]]
    new=dict(old_freeze,status='APPROVED_SUPPLEMENT_TECHNICAL_REPAIR_V2_FROZEN',files=files,
             technical_repair=repair,old_failed_version_immutable=True)
    write(FREEZE,new)
    write(NEW+'/preparation_completion.json',dict(repair,freeze_sha256=sha(ROOT/FREEZE),physical_requests=117,logical_requests=140))
    print(json.dumps(read(NEW+'/preparation_completion.json')),flush=True)


if __name__=='__main__':main()
