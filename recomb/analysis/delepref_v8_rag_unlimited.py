"""Original trained heads + unlimited-span Policy/RAG on a verified real WT cache."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import delepref_unlimited_planning as planning
import delepref_unlimited_scorer as scoring
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT, BASE as ORIGINAL_BASE, write

FREEZE='configs/delepref_v8_rag_unlimited_freeze.json'
OUT_BASE='reports/delepref_v8_rag_unlimited'


def prepare(root):
    source=['analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
        'analysis/delepref_v8_rag_unlimited.py','analysis/tests/test_delepref_unlimited_planning.py',
        'analysis/tests/test_delepref_unlimited_interface.py','slurm/delepref_v8_rag_unlimited.sbatch']
    pins={p:rt.sha(root/p) for p in ['configs/delepref_v8_baseline_freeze.json',
        'reports/rag_v6/profiles.json',ORIGINAL_BASE+'/inputs.json']}
    (root/OUT_BASE).mkdir(exist_ok=False)
    write(root/FREEZE,dict(status='DELEPREF_V8_UNLIMITED_RAG_ENGINEERING_FROZEN',contract=planning.CONTRACT,
        sources={p:rt.sha(root/p) for p in source},inputs=pins,
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,
        reason='User explicitly removed the deletion-span count limit on 2026-10-16',
        reference_only='Published native <=3-span output; not the current task constraint',
        profiles_reused=True,cache_required='verified_real_original_WT_encoding',new_PLM_forwards=0,
        trained_heads_fixed=True,new_training=False,new_structure_predictions=0,paper_changes=False))
    return dict(freeze_sha256=rt.sha(root/FREEZE))


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


def make_rows(api,template,items,length,logits,generation=None):
    rows=[]
    for rank,item in enumerate(items,1):
        spans=tuple(tuple(s) for s in item['spans'])
        mask=planning.validate_plan(length,float(template['budget']),spans)
        row=dict(template)
        row.update(raw_candidate_rank=rank,deleted_intervals=api._intervals_json(spans),
            deleted_length=int(mask.sum()),deletion_mask_sha256=api._mask_sha256(length,spans),
            span_count=len(spans),policy_energy=float(logits[mask].mean()),scorer_score='',scorer_rank=0,
            generation_kind=item['kind'],retrieval_source=item.get('source_target'),
            retrieval_seed_spans=item.get('seed_spans'),all_reliable_events_used=item.get('all_reliable_events_used'),
            decoder=planning.CONTRACT['decoder'],qmax=None,max_span_len=None,strict_plan_valid=True,
            generation_reason='unlimited_span_count_exact_budget',
            generation_attempts=(generation or {}).get('attempts'),duplicate_decodes=(generation or {}).get('duplicates'))
        rows.append(row)
    ordered=sorted(range(len(rows)),key=lambda i:(-rows[i]['policy_energy'],rows[i]['deleted_intervals']))
    for rank,index in enumerate(ordered,1):rows[index]['policy_native_rank']=rank
    return rows


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU Slurm allocation required')
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('unlimited RAG freeze mismatch')
    freeze=json.loads((root/FREEZE).read_text())
    for p,h in {**freeze['sources'],**freeze['inputs']}.items():
        if rt.sha(root/p)!=h:raise ValueError('source/input changed: '+p)
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
    cache=torch.load(original_out/'original_wt_cache.pt',map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:
        raise ValueError('cache is not real original ESMC output')
    records=json.loads((original_out/'inputs.json').read_text())
    chunk=cached_chunk(api,cache,records);encoded=chunk.encoded
    profiles=json.loads((root/'reports/rag_v6/profiles.json').read_text())
    gpu_rows=json.loads((original_out/'baseline_scores.json').read_text())
    gpu={(x['protein_id'],x['budget'],x['generation_seed'],x['deletion_mask_sha256']):x for x in gpu_rows}
    native_all=[];unlimited_all=[];rag_all=[];decisions=[];attempt_log=[];compat=[]
    with torch.inference_mode():
        for seed in (13,29):
            # Reuse already persisted historical masks; do not repeat restricted DP generation.
            native_raw=[row for row in gpu_rows if int(row['generation_seed'])==seed]
            groups={}
            for row in native_raw:groups.setdefault((row['protein_id'],row['budget']),[]).append(row)
            for (pid,budget),native in groups.items():
                native=api.score_candidates(runtime,chunk,sorted(native,key=lambda x:x['raw_candidate_rank']))
                native_all.extend(native);template=native[0];sequence=template['wildtype_sequence']
                key=encoded.key_for(pid,sequence)
                logits=runtime.model.propose_from_embeddings([encoded.embeddings[key]],[budget])[0].residue_logits.detach().cpu().double().numpy()
                if not np.isfinite(logits).all() or np.std(logits)==0:raise ValueError('invalid trained Policy logits')
                decoder=runtime.config['decoder']
                original_items,generation=planning.policy_proposals(logits,budget,template['condition_seed'],
                    perturbation_scale=float(decoder.get('perturbation_scale',1.)),
                    max_attempts=8*int(decoder.get('max_attempts_multiplier',16)))
                if len(original_items)!=8:raise RuntimeError('incomplete unrestricted original K: '+str(generation))
                pool=scoring.score_candidates(runtime,api,chunk,make_rows(api,template,original_items,len(sequence),logits,generation))
                unlimited_all.extend(pool)
                original=[dict(item,row=row) for item,row in zip(original_items,pool)]
                retrieved,attempts=planning.retrieved_proposals(logits,budget,profiles[pid],template['condition_seed'])
                mixed,fallback=planning.mix_candidates(original,retrieved)
                rag=([dict(x) for x in pool] if fallback else scoring.score_candidates(runtime,api,chunk,
                    make_rows(api,template,mixed,len(sequence),logits,generation)))
                if len(rag)!=8:raise RuntimeError('incomplete RAG K')
                rag_all.extend(rag)
                withheld,withheld_fallback=planning.mix_candidates(original,[])
                if not withheld_fallback or [x['row'] for x in withheld]!=pool:
                    raise RuntimeError('no-evidence fallback differs from unrestricted original path')
                original_best=min(pool,key=lambda x:x['scorer_rank']);rag_best=min(rag,key=lambda x:x['scorer_rank'])
                native_best=min(native,key=lambda x:x['scorer_rank'])
                errors=[abs(gpu[(pid,budget,seed,x['deletion_mask_sha256'])]['scorer_score']-x['scorer_score'])
                    for x in native if (pid,budget,seed,x['deletion_mask_sha256']) in gpu]
                compat.append(dict(protein_id=pid,budget=budget,seed=seed,native_mask_overlap=len(errors),
                    max_native_CPU_GPU_score_difference=max(errors) if errors else None))
                item=dict(protein_id=pid,budget=budget,seed=seed,K_original=len(pool),K_RAG=len(rag),
                    retrieved_unique_proposals=len(retrieved),retrieval_slots=sum(x['generation_kind'].startswith('retrieval') for x in rag),
                    retrieval_attempts=len(attempts),fallback=fallback,withheld_fallback_exact=True,
                    original_selected_mask=original_best['deletion_mask_sha256'],rag_selected_mask=rag_best['deletion_mask_sha256'],
                    historical_native_selected_mask=native_best['deletion_mask_sha256'],
                    changed_choice=original_best['deletion_mask_sha256']!=rag_best['deletion_mask_sha256'],
                    selected_RAG_kind=rag_best['generation_kind'],original_selected_spans=original_best['span_count'],
                    RAG_selected_spans=rag_best['span_count'],maximum_candidate_spans=max(x['span_count'] for x in pool+rag))
                decisions.append(item);attempt_log.append(dict(protein_id=pid,budget=budget,seed=seed,
                    original_generation=generation,retrieval_attempts=attempts))
                print(json.dumps(dict(event='UNLIMITED_RAG_CONDITION_COMPLETE',**item)),flush=True)
    write(out/'historical_native_CPU_scores.json',native_all)
    write(out/'original_unlimited_scores.json',unlimited_all)
    write(out/'original_unlimited_selected.json',api.select_scorer_top1(unlimited_all))
    write(out/'rag_scores.json',rag_all);write(out/'rag_selected.json',api.select_scorer_top1(rag_all))
    write(out/'rag_attempts.json',attempt_log);write(out/'CPU_GPU_compatibility.json',compat)
    report=dict(status='UNLIMITED_ORIGINAL_DELEPREF_AND_RAG_END_TO_END_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,WT=6,conditions=len(decisions),qmax=None,max_span_len=None,
        original_candidates=len(unlimited_all),RAG_candidates=len(rag_all),checkpoint_sha256=rt.CHECKPOINT_SHA,
        freeze_sha256=checksum,retrieval_proposals=sum(d['retrieval_slots'] for d in decisions),
        fallback_conditions=sum(d['fallback'] for d in decisions),changed_choices=sum(d['changed_choice'] for d in decisions),
        selected_retrieval_proposals=sum(d['selected_RAG_kind'].startswith('retrieval') for d in decisions),
        maximum_candidate_spans=max(d['maximum_candidate_spans'] for d in decisions),
        all_no_evidence_fallback_tests_exact=True,decisions=decisions,
        historical_native_masks_reused_not_regenerated=True,
        scorer_training_span_scale=3,greater_than_3_spans='training_range_extrapolation',
        extra_PLM_encodings=0,extra_mutant_encodings=0,new_training=False,new_structure_predictions=0,
        actual_structure_quality_measured=False,paper_changes=False,independent_confirmation=False)
    write(out/'rag_completion.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='decisions'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    else:execute(a.root,a.freeze_sha256)
