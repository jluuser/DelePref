"""End-to-end original Policy/Scorer + retrieval planning using real WT cache."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import delepref_rag_planning_v2 as planning
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT, BASE as ORIGINAL_BASE, write

FREEZE='configs/delepref_v8_rag_freeze_v3.json'
OUT_BASE='reports/delepref_v8_rag_v3'


def prepare(root):
    source=['analysis/delepref_rag_planning_v2.py','analysis/delepref_v8_rag_v3.py',
        'analysis/tests/test_delepref_rag_planning_v2.py','analysis/tests/test_delepref_v8_rag_v3_integration.py','slurm/delepref_v8_rag_v3.sbatch']
    pins={p:rt.sha(root/p) for p in ['configs/delepref_v8_baseline_freeze.json',
        'reports/rag_v6/profiles.json',ORIGINAL_BASE+'/inputs.json']}
    (root/OUT_BASE).mkdir(exist_ok=False)
    write(root/FREEZE,dict(status='DELEPREF_V8_RAG_ENGINEERING_FROZEN',contract=planning.CONTRACT,
        sources={p:rt.sha(root/p) for p in source},inputs=pins,
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,
        profiles_reused=True,profile_sha256=pins['reports/rag_v6/profiles.json'],
        cache_required='verified_real_original_WT_encoding',new_PLM_forwards=0,
        trained_heads_fixed=True,new_training=False,new_structure_predictions=0,paper_changes=False))
    return dict(freeze_sha256=rt.sha(root/FREEZE))


def make_rows(api,original,mixed,length,logits):
    rows=[]
    for rank,item in enumerate(mixed,1):
        row=dict(item.get('row',original[0]))
        spans=tuple(tuple(s) for s in item['spans'])
        row.update(raw_candidate_rank=rank,deleted_intervals=api._intervals_json(spans),
            deleted_length=sum(b-a+1 for a,b in spans),deletion_mask_sha256=api._mask_sha256(length,spans),
            span_count=len(spans),policy_energy=float(logits[planning.mask_for(length,spans)].mean()),
            scorer_score='',scorer_rank=0,generation_kind=item['kind'],
            retrieval_source=item.get('source_target'),retrieval_seed_spans=item.get('seed_spans'))
        rows.append(row)
    native=sorted(range(len(rows)),key=lambda i:(-rows[i]['policy_energy'],rows[i]['deleted_intervals']))
    for rank,i in enumerate(native,1):rows[i]['policy_native_rank']=rank
    return rows


def cached_chunk(api,cache,records):
    from delepref.student_model import EncodedWildTypes,wildtype_key
    normalized=tuple(api._normalize_record(row) for row in records)
    for row in normalized:
        if cache['sequences'].get(row['protein_id'])!=row['wildtype_sequence']:
            raise ValueError('cache/input WT identity mismatch')
    encoded=EncodedWildTypes(
        embeddings={wildtype_key(p,cache['sequences'][p]):v for p,v in cache['embeddings'].items()},
        sequences={wildtype_key(p,s):s for p,s in cache['sequences'].items()},input_count=len(records))
    return api.EncodedChunk(encoded=encoded,records=normalized)


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU Slurm allocation required')
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('RAG freeze mismatch')
    freeze=json.loads((root/FREEZE).read_text())
    for p,h in {**freeze['sources'],**freeze['inputs']}.items():
        if rt.sha(root/p)!=h:raise ValueError('RAG source/input changed: '+p)
    baseline_freeze=json.loads((root/'configs/delepref_v8_baseline_freeze.json').read_text())
    for p,h in {**baseline_freeze['inputs'],**baseline_freeze['code']}.items():
        if rt.sha(root/p)!=h:raise ValueError('original frozen source/input changed: '+p)
    original_out=root/ORIGINAL_BASE;out=root/OUT_BASE
    complete=json.loads((original_out/'baseline_completion.json').read_text())
    if complete['checkpoint_sha256']!=rt.CHECKPOINT_SHA:raise ValueError('wrong original model')
    if rt.sha(original_out/'original_wt_cache.pt')!=complete['cache_sha256']:raise ValueError('WT cache changed')
    if rt.sha(original_out/'baseline_scores.json')!=complete['scores_sha256']:raise ValueError('GPU baseline changed')
    write(out/'rag_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    start=time.time();torch.set_num_threads(4)
    runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    from delepref.student_model import validate_strict_plan
    cache=torch.load(original_out/'original_wt_cache.pt',map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:
        raise ValueError('cache is not real original ESMC output')
    records=json.loads((original_out/'inputs.json').read_text())
    chunk=cached_chunk(api,cache,records);encoded=chunk.encoded
    profiles=json.loads((root/'reports/rag_v6/profiles.json').read_text())
    gpu_rows=json.loads((original_out/'baseline_scores.json').read_text())
    gpu={(x['protein_id'],x['budget'],x['generation_seed'],x['deletion_mask_sha256']):x for x in gpu_rows}
    original_all=[];rag_all=[];decisions=[];attempt_log=[];compat=[]
    with torch.inference_mode():
        for seed in (13,29):
            original_raw=api.generate_raw_candidates(runtime,chunk,global_generation_seed=seed)
            groups={}
            for row in original_raw:groups.setdefault((row['protein_id'],row['budget']),[]).append(row)
            for (pid,budget),pool in groups.items():
                pool=sorted(pool,key=lambda x:x['raw_candidate_rank']);sequence=pool[0]['wildtype_sequence']
                pool=api.score_candidates(runtime,chunk,pool)
                original_all.extend(pool)
                key=encoded.key_for(pid,sequence)
                logits=runtime.model.propose_from_embeddings([encoded.embeddings[key]],[budget])[0].residue_logits.detach().cpu().double().numpy()
                if not np.isfinite(logits).all() or np.std(logits)==0:raise ValueError('trained Policy produced invalid/constant logits')
                native=[dict(spans=tuple(map(tuple,json.loads(x['deleted_intervals']))),kind='original_policy',row=x) for x in pool]
                retrieved,attempts=planning.retrieved_proposals(logits,budget,profiles[pid],pool[0]['condition_seed'])
                mixed,fallback=planning.mix_candidates(native,retrieved)
                raw=make_rows(api,pool,mixed,len(sequence),logits)
                for row in raw:
                    validate_strict_plan(len(sequence),budget,json.loads(row['deleted_intervals']),qmax=3,
                        min_span_len=1,max_span_len=max(1,int(len(sequence)*.5)))
                # Exact fallback preserves the original scored output, including batch numerics.
                rag=([dict(x, generation_kind='original_policy', retrieval_source=None, retrieval_seed_spans=None) for x in pool]
                     if fallback else api.score_candidates(runtime,chunk,raw))
                rag_all.extend(rag)
                original_best=min(pool,key=lambda x:x['scorer_rank'])
                rag_best=min(rag,key=lambda x:x['scorer_rank'])
                # Withheld-retrieval counterfactual must preserve the exact original path.
                none,withheld_fallback=planning.mix_candidates(native,[])
                withheld=[x['row'] for x in none]
                if not withheld_fallback or [x['deletion_mask_sha256'] for x in withheld]!=[x['deletion_mask_sha256'] for x in pool]:
                    raise RuntimeError('no-evidence candidates differ from original')
                fallback_error=max(abs(x['scorer_score']-y['scorer_score']) for x,y in zip(withheld,pool))
                if fallback_error>1e-7:raise RuntimeError('no-evidence scoring differs from original')
                overlap=0;errors=[]
                for x in pool:
                    gx=gpu.get((pid,budget,seed,x['deletion_mask_sha256']))
                    if gx is not None:overlap+=1;errors.append(abs(gx['scorer_score']-x['scorer_score']))
                compat.append(dict(protein_id=pid,budget=budget,seed=seed,mask_overlap=overlap,
                    max_scorer_CPU_GPU_difference=max(errors) if errors else None))
                item=dict(protein_id=pid,budget=budget,seed=seed,K_original=len(pool),K_RAG=len(rag),
                    retrieved_unique_proposals=len(retrieved),retrieval_slots=sum(x['generation_kind'].startswith('retrieval') for x in rag),
                    retrieval_attempts=len(attempts),fallback=fallback,withheld_fallback_exact=True,
                    fallback_score_difference=fallback_error,policy_logit_std=float(np.std(logits)),
                    original_selected_mask=original_best['deletion_mask_sha256'],rag_selected_mask=rag_best['deletion_mask_sha256'],
                    changed_choice=original_best['deletion_mask_sha256']!=rag_best['deletion_mask_sha256'],
                    selected_RAG_kind=rag_best['generation_kind'])
                decisions.append(item)
                attempt_log.append(dict(protein_id=pid,budget=budget,seed=seed,attempts=attempts))
                print(json.dumps(dict(event='RAG_CONDITION_COMPLETE',**item)),flush=True)
    write(out/'original_CPU_scores.json',original_all)
    write(out/'rag_scores.json',rag_all)
    write(out/'rag_selected.json',api.select_scorer_top1(rag_all))
    write(out/'rag_attempts.json',attempt_log)
    write(out/'CPU_GPU_compatibility.json',compat)
    report=dict(status='ORIGINAL_DELEPREF_AND_RAG_END_TO_END_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,planner_version='v2_provable_infeasibility_prune',WT=6,conditions=len(decisions),original_candidates=len(original_all),RAG_candidates=len(rag_all),
        checkpoint_sha256=rt.CHECKPOINT_SHA,freeze_sha256=checksum,
        retrieval_proposals=sum(d['retrieval_slots'] for d in decisions),fallback_conditions=sum(d['fallback'] for d in decisions),
        changed_choices=sum(d['changed_choice'] for d in decisions),
        selected_retrieval_proposals=sum(d['selected_RAG_kind'].startswith('retrieval') for d in decisions),
        all_no_evidence_fallback_tests_exact=True,decisions=decisions,
        cached_head_profile='CPU_float32_heads_on_verified_original_bfloat16_WT_cache',
        extra_PLM_encodings=0,extra_mutant_encodings=0,new_training=False,new_structure_predictions=0,
        actual_structure_quality_measured=False,paper_changes=False,independent_confirmation=False)
    write(out/'rag_completion.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='decisions'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    else:execute(a.root,a.freeze_sha256)
