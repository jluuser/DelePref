"""Approved supplementary budgets, using genuine frozen WT cache and original heads."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import delepref_v8_runtime as rt
import delepref_v8_main as main_v8
import delepref_v8_controls as controls
import delepref_unlimited_planning as planning
from delepref_v8_rag_unlimited import cached_chunk

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = 'reports/recomb_task_budget_20261017'
FREEZE = 'configs/recomb_task_budget_20261017.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def prepare():
    if (ROOT / FREEZE).exists():
        raise FileExistsError('supplementary generation already frozen')
    paths = [__file__, ROOT / 'analysis/delepref_v8_main.py', ROOT / 'analysis/delepref_v8_controls.py',
             ROOT / 'analysis/delepref_unlimited_planning.py', ROOT / 'analysis/delepref_unlimited_scorer.py',
             ROOT / 'analysis/delepref_v8_runtime.py', ROOT / 'reports/delepref_v8_main/wt_cache.pt',
             ROOT / 'reports/delepref_v8_main/encoding_completion.json', ROOT / 'reports/delepref_v8_main/all_selections.json',
             ROOT / 'reports/delepref_v8_main/candidate_scores.json',
             ROOT / 'reports/delepref_v8_evaluation_preparation/main_encoder_inputs.json',
             ROOT / 'reports/delepref_v8_evaluation_preparation/subsets.json',
             ROOT / 'retrieval/delepref_v8_main/profiles.json', ROOT / 'configs/delepref_v8_main_freeze.json']
    write(FREEZE, dict(status='SUPPLEMENTARY_GENERATION_FROZEN', budgets=[.1,.2,.3,.4], seeds=[13,29],
        K=8, qmax=None, methods=['delepref_unlimited','rag_unlimited','msa_conservation','terminal'],
        checkpoint_sha256=rt.CHECKPOINT_SHA, files={str(p):rt.sha(p) for p in paths},
        original_main_freeze_sha256=rt.sha(ROOT/'configs/delepref_v8_main_freeze.json'),
        fold_subset='existing completion_ablation20WT', new_fold_roster='Terminal10%; original/RAG/Terminal30/40%; seed13 only',
        maximum_new_logical_predictions=140, higher_budgets_training_extrapolation=True,
        structure_allowed=False, no_new_training=True, old_results_immutable=True))
    print(json.dumps(dict(freeze_sha256=rt.sha(ROOT/FREEZE))))


def run(checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU Slurm allocation required')
    if rt.sha(ROOT/FREEZE) != checksum: raise ValueError('supplement freeze changed')
    freeze = read(FREEZE)
    for path, expected in freeze['files'].items():
        if rt.sha(path) != expected: raise ValueError('frozen input changed: '+path)
    main_v8.verify(ROOT, freeze['original_main_freeze_sha256'])
    start = time.time(); torch.set_num_threads(4)
    runtime, api = rt.load(ROOT, device='cpu', cached_heads_only=True)
    cache = torch.load(ROOT/'reports/delepref_v8_main/wt_cache.pt', map_location='cpu', weights_only=True)
    assert cache['real_encoder'] and cache['unique_WT']==100 and cache['checkpoint_sha256']==rt.CHECKPOINT_SHA
    records = read('reports/delepref_v8_evaluation_preparation/main_encoder_inputs.json')
    profiles = read('retrieval/delepref_v8_main/profiles.json')
    chunk = cached_chunk(api, cache, records)
    old_selections = {(x['protein_id'], float(x['budget']), int(x['generation_seed']), x['method']):x
                      for x in read('reports/delepref_v8_main/all_selections.json')}
    old_candidates = {}
    for x in read('reports/delepref_v8_main/candidate_scores.json'):
        old_candidates.setdefault((x['protein_id'],float(x['budget']),int(x['generation_seed']),x['method']),[]).append(x)
    selected=[]; candidates=[]; diagnostics=[]; reused=0
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id']; sequence=record['wildtype_sequence']; profile=profiles[pid]
            msa_logits, msa_audit = controls.conservation_logits(sequence,profile)
            for budget in freeze['budgets']:
                for seed in freeze['seeds']:
                    keys=[(pid,budget,seed,m) for m in freeze['methods']]
                    if all(k in old_selections for k in keys):
                        chosen=[dict(old_selections[k]) for k in keys]
                        scores=[dict(x) for k in keys for x in old_candidates[k]]
                        diag=dict(protein_id=pid,budget=budget,seed=seed,reused_frozen_outputs=True)
                        reused+=len(chosen)
                    else:
                        cseed=controls.condition_seed(seed,pid,sequence,budget)
                        msa,cost=([],dict(attempts=0,duplicates=0)) if msa_audit['all_unknown'] else planning.policy_proposals(msa_logits,budget,cseed)
                        control={}
                        for method, items in [('terminal',controls.terminal_proposals(len(sequence),budget)),('msa_conservation',msa)]:
                            control[method]=[]
                            for item in items:
                                mask=planning.validate_plan(len(sequence),budget,item['spans'])
                                control[method].append(dict(deleted_intervals=item['spans'],
                                    deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                                    generation_cost=cost if method=='msa_conservation' else dict(attempts=8,duplicates=0)))
                        scores,chosen,diag=main_v8.score_condition(runtime,api,chunk,dict(record,budget=budget),profile,seed,control_rows=control)
                        # Old outputs are a compatibility oracle, not a quality-selection criterion.
                        for x in chosen:
                            k=(pid,budget,seed,x['method'])
                            if k in old_selections and x['deletion_mask_sha256']!=old_selections[k]['deletion_mask_sha256']:
                                raise RuntimeError('old frozen selection changed: '+str(k))
                    for x in chosen:
                        x['supplement_role']='CPU_stability_only' if seed==29 else 'budget_curve_generation'
                        x['budget_training_extrapolation']=budget>.2
                        mask=planning.validate_plan(len(sequence),budget,json.loads(x['deleted_intervals']))
                        assert x['edited_sequence']==''.join(a for a,d in zip(sequence,mask) if not d)
                    selected.extend(chosen);candidates.extend(scores);diagnostics.append(diag)
            print('BUDGET_CPU_WT_COMPLETE '+pid,flush=True)
    assert len(selected)==3200 and len(candidates)==25600
    subset=set(read('reports/delepref_v8_evaluation_preparation/subsets.json')['completion_ablation'])
    fold=[x for x in selected if x['protein_id'] in subset and x['generation_seed']==13 and
          ((x['budget']==.1 and x['method']=='terminal') or
           (x['budget'] in (.3,.4) and x['method'] in ('delepref_unlimited','rag_unlimited','terminal')))]
    assert len(fold)==140 and len(subset)==20
    write(BASE+'/all_selections.json',selected);write(BASE+'/candidate_scores.json',candidates)
    write(BASE+'/generation_diagnostics.json',diagnostics);write(BASE+'/fold_selections.json',fold)
    report=dict(status='FOUR_BUDGET_CPU_GENERATION_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,
        WT=100,conditions=800,K=8,selected_outputs=len(selected),scored_candidates=len(candidates),
        reused_old_selections=reused,new_logical_fold_requests=len(fold),new_encoder_forwards=0,new_training=False,
        new_structure_calls=0,freeze_sha256=checksum,
        outputs={n:rt.sha(ROOT/BASE/n) for n in ['all_selections.json','candidate_scores.json','generation_diagnostics.json','fold_selections.json']},
        selected_span_count_max=max(x['span_count'] for x in selected))
    write(BASE+'/completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.freeze_sha256)
