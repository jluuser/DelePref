"""Independent semantic audit, old-cache reuse and bounded supplementary dispatch freeze."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from recomb_task_budget_20261017 import ROOT, BASE as GEN, read, write
from delepref_unlimited_planning import validate_plan
from delepref_v8_runtime import sha

BASE='reports/recomb_task_structure_20261017'
FREEZE='configs/recomb_task_structure_freeze_20261017.json'
sys.path.insert(0,str(ROOT/'code'))
from budget_ledger import BudgetLedger


def main():
    if (ROOT/FREEZE).exists():raise FileExistsError('supplement already frozen')
    completion=read(GEN+'/completion.json')
    for name,h in completion['outputs'].items():assert sha(ROOT/GEN/name)==h
    panel=read('reports/delepref_v8_native_panel/main_panel.json'); wt={x['protein_id']:x for x in panel}
    selections=read(GEN+'/all_selections.json');candidates=read(GEN+'/candidate_scores.json')
    pools={}
    for x in candidates:pools.setdefault((x['protein_id'],float(x['budget']),int(x['generation_seed']),x['method']),[]).append(x)
    assert len(selections)==3200 and len(pools)==3200
    for row in selections:
        key=(row['protein_id'],float(row['budget']),int(row['generation_seed']),row['method']);pool=pools[key]
        assert len(pool)==8 and len({x['deletion_mask_sha256'] for x in pool})==8
        assert row['deletion_mask_sha256']==min(pool,key=lambda x:x['scorer_rank'])['deletion_mask_sha256']
        seq=wt[row['protein_id']]['sequence']; mask=validate_plan(len(seq),row['budget'],json.loads(row['deleted_intervals']))
        assert hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()==row['deletion_mask_sha256']
        assert ''.join(a for a,d in zip(seq,mask) if not d)==row['edited_sequence']
    old_ledger_path=ROOT/'reports/delepref_v8_structures/budget_ledger.jsonl'
    old=BudgetLedger(ROOT/'reports/delepref_v8_evaluation_preparation/statistical_protocol.json',old_ledger_path).snapshot()
    assert len(old['attempts'])==641 and all(r['state']=='succeeded' for r in old['attempts'])
    old_states={r['task_id']:r for r in old['attempts']}
    old_requests=read('reports/delepref_v8_structure_preparation/physical_requests.json')
    esm=[r for r in old_requests if r['predictor']=='esmfold'];template=esm[0]
    assert len({(r['weights_sha256'],r['profile_sha256'],r['seed'],r['sample']) for r in esm})==1
    subset=set(read('reports/delepref_v8_evaluation_preparation/subsets.json')['completion_ablation'])
    logical=[];physical={};reused={}
    for i,x in enumerate(read(GEN+'/fold_selections.json')):
        assert x['protein_id'] in subset and x['generation_seed']==13
        seq=wt[x['protein_id']]['sequence'];mask=validate_plan(len(seq),x['budget'],json.loads(x['deleted_intervals']))
        task='esmfold_'+x['edited_sequence_sha256']
        request=dict(predictor='esmfold',profile_sha256=template['profile_sha256'],weights_sha256=template['weights_sha256'],
                     seed=template['seed'],sample=template['sample'],sequence=x['edited_sequence'],
                     sequence_sha256=x['edited_sequence_sha256'],task_id=task,category='supplementary_outputs',wt_variant='edited')
        logical.append(dict(logical_id='supplement_'+str(i),physical_id=task,protein_id=x['protein_id'],method=x['method'],
                            budget=x['budget'],role='supplementary_budget_curve',predictor='esmfold',kept=np.flatnonzero(~mask).tolist(),
                            deleted_intervals=json.loads(x['deleted_intervals']),edited_sequence_sha256=x['edited_sequence_sha256']))
        if task in old_states:
            state=old_states[task]
            assert sha(state['output_path'])==state['output_sha']
            original=next(r for r in esm if r['task_id']==task)
            assert all(original[k]==request[k] for k in ['sequence','profile_sha256','weights_sha256','seed','sample'])
            reused[task]=state
        else:physical.setdefault(task,request)
    assert len(logical)==140 and len(physical)<=140
    write(BASE+'/logical_requests.json',logical);write(BASE+'/physical_requests.json',list(physical.values()))
    write(BASE+'/reused_old_requests.json',reused)
    write(BASE+'/statistical_protocol.json',dict(status='SUPPLEMENTARY_NOT_UNTOUCHED_CONFIRMATION',
        old_actual_calls=641,project_limit=800,maximum_new_planned_calls=140,maximum_new_calls_including_retries=159,
        paired_roster=sorted(subset),budgets=[.1,.2,.3,.4],methods=['delepref_unlimited','rag_unlimited','terminal'],
        components='Use unchanged100WT panel component IDs',bootstrap_replicates=10000,bootstrap_seed=20261017,
        metrics='Joint report of T, retained CA lDDT, full-WT fixed-correspondence TM, contact retention, static loss; functional annotations separately',
        high_budget='30/40% original-model extrapolation; no presumption of any winning method',
        unknown_and_failure='Keep complete logical roster and metric bounds, no silent deletions',
        compute=dict(structure_call_limit=159,structure_categories=dict(supplementary_outputs=140,retry=19),
                     structure_retries_max=1,max_task_GPUs=2,GPU_only_Slurm=True)))
    audit=dict(status='ALL_3200_SELECTIONS_AND_NEW_140_LOGICAL_REQUESTS_VERIFIED',K=8,WT=100,conditions=800,
               no_new_encoder_calls=True,old_calls_verified=641,new_physical_requests=len(physical),
               cached_old_physical_requests=len(reused),new_structure_calls=0,
               old_ledger_sha256=sha(old_ledger_path),selection_output_sha256=sha(ROOT/GEN/'all_selections.json'))
    write(BASE+'/semantic_audit.json',audit)
    old_freeze=read('configs/delepref_v8_confirmation_freeze.json')
    sources=['analysis/recomb_task_structure_prepare_20261017.py','analysis/recomb_task_fold_worker_20261017.py',
             'analysis/recomb_task_structure_evaluate_20261017.py','analysis/delepref_v8_structure_metrics.py',
             'code/build_pairs.py','code/budget_ledger.py','code/paired_statistics.py']
    inputs=[GEN+'/completion.json',GEN+'/fold_selections.json',
            'reports/delepref_v8_native_panel/main_panel.json',
            'reports/delepref_v8_structure_preparation/predictor_profiles.json',
            'reports/delepref_v8_structure_evaluation/per_output_metrics.json',
            'reports/delepref_v8_structures/budget_ledger.jsonl']
    inputs += [BASE+'/'+n for n in ['logical_requests.json','physical_requests.json','reused_old_requests.json','statistical_protocol.json','semantic_audit.json']]
    files=[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in sources+inputs]
    # Preserve complete original predictor identity and native reference/source closure.
    files+=old_freeze['files']
    write(FREEZE,dict(status='APPROVED_SUPPLEMENT_COMPLETE_FREEZE',fold_allowed=True,all_selected_outputs_frozen=True,
        predictor_reports=old_freeze['predictor_reports'],files=files,old_calls=641,new_physical_calls=len(physical),
        new_logical_requests=140,maximum_total_calls=800,all_original_freezes_immutable=True,paper_changes=False))
    write(BASE+'/preparation_completion.json',dict(audit,freeze_sha256=sha(ROOT/FREEZE),
        physical_requests_sha256=sha(ROOT/BASE/'physical_requests.json')))
    print(json.dumps(read(BASE+'/preparation_completion.json')),flush=True)


if __name__=='__main__':main()
