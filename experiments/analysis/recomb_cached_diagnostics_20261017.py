"""Fixed-budget four-plan interaction and cached CPU cost diagnostics, no biology claims."""
from collections import defaultdict
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
from delepref_unlimited_scorer import preference_scores
from delepref_v8_rag_unlimited import cached_chunk
from recomb_expanded_external_prepare_20261017 import ROOT,write,sha
BASE='reports/recomb_cached_diagnostics_20261017'


def intervals(mask):
    spans=[]
    for p in np.flatnonzero(mask):
        if spans and p==spans[-1][1]+1:spans[-1][1]=int(p)
        else:spans.append([int(p),int(p)])
    return spans


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    main_v8.verify(ROOT,sha(ROOT/'configs/delepref_v8_main_freeze.json'));torch.set_num_threads(4)
    runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
    cache=torch.load(ROOT/'reports/delepref_v8_main/wt_cache.pt',map_location='cpu',weights_only=True)
    records=json.loads((ROOT/'reports/delepref_v8_evaluation_preparation/main_encoder_inputs.json').read_text())
    chunk=cached_chunk(api,cache,records)
    profiles=json.loads((ROOT/'retrieval/delepref_v8_main/profiles.json').read_text())
    from delepref.preference_core import DeletionCandidate
    interactions=[];timings=[];retrieval=[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id'];seq=record['wildtype_sequence'];L=len(seq);beta=.2;D=planning.delete_count(L,beta)
            emb=cache['embeddings'][pid]
            logits=runtime.model.propose_from_embeddings([emb],[beta])[0].residue_logits.detach().float().cpu().numpy()
            fixed=sorted(range(2,L-2),key=lambda p:hashlib.sha256(('fixed_four_plan_contrast_20261017\0'+seq+'\0'+str(p)).encode()).hexdigest())[:4]
            a,b,c,d=fixed;base=sorted((p for p in range(L) if p not in fixed),key=lambda p:(-float(logits[p]),p))[:D-2]
            candidates=[];additive=[];counts=[]
            for index,pair in enumerate([(a,c),(a,d),(b,c),(b,d)]):
                mask=np.zeros(L,dtype=bool);mask[base+list(pair)]=True;spans=intervals(mask)
                assert int(mask.sum())==D and spans and int((~mask).sum())>0
                candidates.append(DeletionCandidate(protein_id=pid,perturbation_id='contrast_'+str(index),plan_id='contrast_'+str(index),
                    wildtype_sequence=seq,deleted_intervals=tuple(map(tuple,spans)),deleted_length=D,deletion_fraction=D/L,
                    target_delete_fraction=beta,budget_bin='20pct',generation_policy='fixed_budget_four_plan_diagnostic'))
                additive.append(float(logits[mask].astype(float).sum()));counts.append(len(spans))
            values=preference_scores(runtime.model.scorer,candidates,chunk.encoded).detach().float().cpu().tolist()
            contrast=lambda v:v[0]+v[3]-v[1]-v[2]
            assert abs(contrast(additive))<1e-6
            interactions.append(dict(protein_id=pid,budget=beta,deleted_count=D,Scorer_contrast=contrast(values),
                Policy_additive_contrast=contrast(additive),span_counts=counts,all_four_same_span_count=len(set(counts))==1,
                interpretation='Model-score nonadditivity under equal count; not measured functional epistasis'))
            cseed=controls.condition_seed(13,pid,seq,beta)
            for K in [1,4,8,16]:
                for repeat in range(2):
                    started=time.perf_counter();items,cost=planning.policy_proposals(logits,beta,cseed,k=K)
                    generation_seconds=time.perf_counter()-started;assert len(items)==K
                    plans=[DeletionCandidate(protein_id=pid,perturbation_id='cost_'+str(i),plan_id='cost_'+str(i),
                        wildtype_sequence=seq,deleted_intervals=x['spans'],deleted_length=D,deletion_fraction=D/L,
                        target_delete_fraction=beta,budget_bin='20pct',generation_policy='CPU_timing_only') for i,x in enumerate(items)]
                    started=time.perf_counter();values=preference_scores(runtime.model.scorer,plans,chunk.encoded);scoring_seconds=time.perf_counter()-started
                    assert torch.isfinite(values).all()
                    timings.append(dict(protein_id=pid,K=K,repeat=repeat,CPU_generation_seconds=generation_seconds,
                        CPU_Scorer_seconds=scoring_seconds,attempts=cost['attempts'],duplicate_decodes=cost['duplicates']))
            started=time.perf_counter();items,attempts=planning.retrieved_proposals(logits,beta,profiles[pid],cseed)
            retrieval.append(dict(protein_id=pid,warm_retrieval_planning_seconds=time.perf_counter()-started,
                retrieved_proposals=len(items),retrieval_attempts=attempts,
                scope='Frozen homolog profile already cached; excludes database search/IO/WT encoder'))
    write(BASE+'/matched_budget_four_plan_interactions.json',interactions);write(BASE+'/CPU_K_costs.json',timings)
    write(BASE+'/warm_retrieval_planning_costs.json',retrieval)
    summary=[]
    for K in [1,4,8,16]:
        rs=[r for r in timings if r['K']==K]
        summary.append(dict(K=K,WT=100,repeats_per_WT=2,
            generation_seconds_mean=float(np.mean([r['CPU_generation_seconds'] for r in rs])),
            scorer_seconds_mean=float(np.mean([r['CPU_Scorer_seconds'] for r in rs])),
            attempts_mean=float(np.mean([r['attempts'] for r in rs]))))
    write(BASE+'/completion.json',dict(status='CACHED_MODEL_NONADDITIVITY_AND_COST_DIAGNOSTICS_COMPLETE',
        job_id=os.environ['SLURM_JOB_ID'],WT=100,four_plan_contrasts=100,
        same_span_count_contrasts=sum(r['all_four_same_span_count'] for r in interactions),
        absolute_Scorer_contrast_mean=float(np.mean([abs(r['Scorer_contrast']) for r in interactions])),
        maximum_additive_Policy_contrast=max(abs(r['Policy_additive_contrast']) for r in interactions),CPU_K_cost_summary=summary,
        K_scope='Original Policy generation and common Scorer timing only; deployed RAG remains frozen K8; no quality assigned to unpredicted K configurations',
        retrieval_scope='Warm planning only; cold search and initial WT encoding reuse their existing completion runtimes, no fabricated end-to-end sum',
        new_encoder_forwards=0,new_structure_calls=0,new_training=False,no_measured_epistasis_claim=True))
    print(json.dumps(json.loads((ROOT/BASE/'completion.json').read_text())),flush=True)


if __name__=='__main__':main()
