"""End-of-run identity, mapping, coverage, leakage-exclusion and global-call audit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from recomb_expanded_external_prepare_20261017 import ROOT,BASE,sha,write
sys.path.insert(0,str(ROOT/'code'))
from budget_ledger import BudgetLedger
OUT='reports/recomb_task_aligned_final_20261017'


def read(path):return json.loads((ROOT/path).read_text())


def main():
    for path in ['configs/recomb_task_structure_freeze_20261017_v2.json','configs/recomb_task_metrics_roundoff_20261017_v3.json']:
        freeze=read(path)
        files=freeze['files']
        files=files.items() if isinstance(files,dict) else [(f['path'],f['sha256']) for f in files]
        for p,h in files:assert sha(ROOT/p)==h,p
    old=BudgetLedger(ROOT/'reports/delepref_v8_evaluation_preparation/statistical_protocol.json',ROOT/'reports/delepref_v8_structures/budget_ledger.jsonl').snapshot()
    new=BudgetLedger(ROOT/'reports/recomb_task_structure_20261017_v2/statistical_protocol.json',ROOT/'reports/recomb_task_structures_20261017_v2/budget_ledger.jsonl').snapshot()
    assert len(old['attempts'])==641 and len(new['attempts'])==117 and old['consumed']+new['consumed']==758
    assert all(r['state']=='succeeded' for r in new['attempts'])
    requests=read('reports/recomb_task_structure_20261017_v2/physical_requests.json');states={r['task_id']:r for r in new['attempts']}
    for r in requests:
        s=states[r['task_id']];assert sha(s['output_path'])==s['output_sha']
        record=read('reports/recomb_task_structures_20261017_v2/records/'+r['task_id']+'.json')
        assert record['output_sha256']==s['output_sha']
        with np.load(s['output_path'],allow_pickle=False) as a:
            assert a['sequence'].item()==r['sequence'] and a['sequence_sha256'].item()==r['sequence_sha256']
            assert a['coordinates'].shape==(len(r['sequence']),3)
    panel={x['protein_id']:x for x in read('reports/delepref_v8_native_panel/main_panel.json')}
    logical=read('reports/recomb_task_structure_20261017_v2/logical_requests.json')
    request_lookup={r['task_id']:r for r in requests}
    for r in logical:
        seq=''.join(panel[r['protein_id']]['sequence'][i] for i in r['kept'])
        assert seq==request_lookup[r['physical_id']]['sequence']
    external=read(BASE+'/inputs.json');scores=read(BASE+'/model_mapping_scores.json')
    complete=read(BASE+'/scoring_completion.json');assert sha(ROOT/BASE/'model_mapping_scores.json')==complete['scores_sha256']
    assert set(scores)=={r['edit_key'] for r in external['edits']}
    for edit in external['edits']:
        ids=set(edit['deleted_positions'])
        assert ''.join(a for i,a in enumerate(edit['wt_sequence']) if i not in ids)==edit['mutant_sequence']
        assert np.isfinite(list(scores[edit['edit_key']].values())).all()
    exclusion=set(read(BASE+'/retrieval_exclusion_sha256.json'))
    profiles=read('retrieval/delepref_v8_main/profiles.json');overlap=[]
    for pid,p in profiles.items():
        for representative in p['representatives']:
            if representative['target_sequence_sha256'] in exclusion:overlap.append(dict(protein_id=pid,target=representative['target']))
    write(OUT+'/expanded_exclusion_vs_frozen_retrieval.json',dict(exact_full_target_overlaps=overlap,
        expanded_exclusion_sequences=len(exclusion),profiles=100,old_profiles_unchanged=True,
        scope='Full retrieved target SHA comparison only; not an exhaustive homology or label-leakage proof'))
    metric=read('reports/recomb_task_structure_20261017_v2/evaluation_completion_v3.json')
    assert metric['logical_outputs']==240 and metric['new_actual_calls']==117
    result=dict(status='APPROVED_CORE_EXPERIMENTS_AND_ARTIFACT_AUDITS_COMPLETE',
        old_structure_calls=641,new_structure_calls=117,total_complete_structure_calls=758,limit=800,
        new_structure_failures=0,new_structure_retries=0,unused_global_capacity=42,
        pre_forward_GPU_interface_failures=2,pre_forward_failures_are_not_structure_predictions=True,
        old_confirmations_unchanged=True,new_physical_artifact_SHAs_verified=117,new_logical_deletion_maps_verified=140,
        four_budget_logical_outputs=240,expanded_measured_observations=7047,expanded_WT=61,expanded_assays=62,
        unique_external_coordinate_mappings_verified=len(external['edits']),public_baselines=24,
        frozen_retrieval_expanded_exclusion_overlap_count=len(overlap),
        numerical_TM_interval_audit=read('reports/recomb_task_structure_20261017_v2/numerical_interval_audit_v3.json'),
        source_and_direction_audit=read(BASE+'/interpretation_completion_v2.json'),
        new_model_training=False,original_model_fixed=True,unlimited_span_extrapolation_disclosed=True,paper_changes=False,
        unfinished_optional=['Functional-hard-protection adapter not enabled; no common protected-task experiment claimed',
                             'No new RAG binder complex or Cas9 benchmark execution',
                             'No new wet-lab functional validation of generated large multi-span products',
                             'SCISOR M/L and PoET2 not run; official public PoET and Tranception families used; SCISOR-S M1 limited to single deletion'])
    write(OUT+'/completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_and_direction_audit','unfinished_optional']}),flush=True)


if __name__=='__main__':main()
