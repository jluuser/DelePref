"""Predeclared E4/E5 interventions and E7 controls on frozen real General100 heads."""
import argparse
from collections import defaultdict,deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
import delepref_unlimited_planning as planning

BASE=Path('reports/recomb_confirm_v2_General100_mechanisms')
FZ=Path('configs/recomb_confirm_v2_General100_mechanisms_freeze.json')


def h(value):return hashlib.sha256(value.encode()).hexdigest()


def shape(spans,n):
    return sorted((b-a+1,'both' if a==0 and b==n-1 else ('N' if a==0 else ('C' if b==n-1 else 'internal'))) for a,b in spans)


def randomized_seed(spans,n,rng):
    result=[]
    for a,b in spans:
        size=b-a+1
        if a==0:start=0
        elif b==n-1:start=n-size
        elif n-size<=1:return None
        else:start=int(rng.integers(1,n-size))
        end=start+size-1
        if any(not(end+2<=c or d+2<=start) for c,d in result):return None
        result.append((start,end))
    return tuple(sorted(result))


def morphology(logits,beta,retrieved,seed):
    n=len(logits);rng=np.random.default_rng(seed%(2**32));items=[];diagnostics=[]
    for parent in retrieved:
        before=tuple(parent['seed_spans']);accepted=None;attempt=0
        values=np.asarray(logits,dtype=np.float64)
        if parent['kind']=='retrieval_refined':
            salt=int(h(parent['source_target']+str(before))[:8],16)
            values=values+np.random.default_rng((seed+salt)%(2**32)).standard_normal(n)
        for attempt in range(1,201):
            spans=randomized_seed(before,n,rng)
            if spans is None or spans==before:continue
            forced=planning.mask_for(n,spans);kept=np.zeros(n,dtype=bool)
            for a,b in spans:
                if a:kept[a-1]=True
                if b<n-1:kept[b+1]=True
            try:completed,_=planning.complete(values,beta,forced,kept)
            except planning.InfeasibleCompletion:continue
            if completed==parent['spans'] or shape(completed,n)!=shape(parent['spans'],n):continue
            accepted=dict(parent,spans=completed,seed_spans=spans,kind='morphology_position_counterfactual',
                source_target='COUNTERFACTUAL_POSITION_'+h(parent['source_target']+str(spans))[:16],
                all_reliable_events_used=False)
            break
        items.append(accepted or parent)
        diagnostics.append(dict(parent_source=parent['source_target'],attempts=attempt,
            effective_final_shape_matched_intervention=accepted is not None,
            seed_shape=shape(before,n),final_shape=shape(parent['spans'],n),
            failed_match_uses_unchanged_parent=accepted is None,not_observed_natural_homolog=True))
    return items,diagnostics


def source_balanced(rows,n,salt):
    groups=defaultdict(list)
    for r in rows:groups[r['stratum']].append(r)
    queues=[deque(sorted(groups[k],key=lambda r:h(salt+'|'+r['WT_sha256']+'|'+r['protein_id']))) for k in sorted(groups)]
    result=[]
    while any(queues) and len(result)<n:
        for q in queues:
            if q and len(result)<n:result.append(q.popleft())
    return result


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('E4/E5/E7 already frozen')
    panel=read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    profiles=read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
    eligible=[r for r in panel if planning.seed_sets(profiles[r['protein_id']],len(r['sequence']),.2)]
    selected=source_balanced(eligible,40,'v2-E4-input-20261018')
    scarcity=source_balanced(selected,20,'v2-E5-input-20261018')
    write(root/BASE/'subsets.json',dict(E4=[r['protein_id'] for r in selected],E5=[r['protein_id'] for r in scarcity],
        eligibility='at least one reliable intact natural event seed within20pct budget; joint exchange validity separately reported',
        eligible=len(eligible),E4_target=40,E5_target=20,source_balanced_input_hash_only=True))
    names=['analysis/recomb_confirm_v2_mechanisms.py','analysis/recomb_general100_mechanisms_20261017.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/delepref_v8_main.py','analysis/delepref_v8_runtime.py',
        'analysis/delepref_v8_rag_unlimited.py','analysis/delepref_unlimited_scorer.py','analysis/delepref_unlimited_planning.py',
        'docs/superpowers/plans/2026-10-18-recomb-v2-mechanism-instantiation.md',
        'analysis/tests/test_recomb_confirm_v2_mechanisms.py','slurm/recomb_confirm_v2_mechanisms.sbatch',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_General100_plans/candidate_scores.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json','retrieval/recomb_confirm_v2_General100/profiles.json',
        str(BASE/'subsets.json')]
    write(root/FZ,dict(status='V2_E4_E5_E7_CONTROLS_FROZEN_BEFORE_NEW_GENERAL_VARIANT_STRUCTURES',
        files={p:sha(root/p) for p in names},K=8,budget=.2,generation_seed=13,
        morphology_attempts_per_retrieved_proposal=200,final_shape_match_required=True,
        joint_null='same existing100 accepted/10000 attempt double-edge swaps; marginals and row load/count preserved',
        scarcity='ceil N/2, ceil N/4,0 nested hash membership with original quality order',
        original_checkpoint_unchanged=True,new_structure_predictions=0,no_positive_result_gate=True))
    print(dict(E4=len(selected),E5=len(scarcity),eligible=len(eligible),freeze_sha256=sha(root/FZ)),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,d in f['files'].items():
        if sha(root/p)!=d:raise ValueError('Frozen E4/E5/E7 source/input changed: '+p)
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Mechanisms claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as main
    import delepref_unlimited_scorer as scoring
    from recomb_general100_mechanisms_20261017 import null_profile,select_rows
    baseline=read(root/'configs/delepref_v8_baseline_freeze.json')
    for p,d in {**baseline['inputs'],**baseline['code']}.items():
        if sha(root/p)!=d:raise ValueError('Original frozen model dependency changed')
    cachepath=root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    if sha(cachepath)!=read(cachepath.parent/'encoding_completion.json')['cache_sha256']:raise ValueError('True original WT cache changed')
    cache=torch.load(cachepath,map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:raise ValueError('Not original real WT embeddings')
    torch.set_num_threads(4);runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    panel=read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    records=[dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],budget=.2) for r in panel]
    chunk=main.cached_chunk(api,cache,records);profiles=read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
    groups=defaultdict(list)
    for r in read(root/'reports/recomb_confirm_v2_General100_plans/candidate_scores.json'):
        if r['budget']==.2 and r['generation_seed']==13:groups[(r['protein_id'],r['method'])].append(r)
    subset=read(out/'subsets.json');scores=[];selected=[];diags=[]
    def score_items(seq,original,items,logits,method):
        mixed,fallback=planning.mix_candidates(original,items)
        pool=[dict(p['row']) for p in original] if fallback else scoring.score_candidates(runtime,api,chunk,
            main.make_rows(api,original[0]['row'],mixed,len(seq),logits))
        scores.extend(dict(p,method=method) for p in pool)
        selected.append(select_rows(api,seq,pool,method));return pool,fallback
    with torch.inference_mode():
        for pid in subset['E4']:
            original=sorted(groups[(pid,'delepref_unlimited')],key=lambda r:r['raw_candidate_rank'])
            rag=sorted(groups[(pid,'rag_unlimited')],key=lambda r:r['raw_candidate_rank'])
            seq=original[0]['wildtype_sequence'];seed=int(original[0]['condition_seed'])
            native=[dict(spans=tuple(map(tuple,json.loads(p['deleted_intervals']))),kind=p['generation_kind'],row=p) for p in original]
            emb=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
            logits=runtime.model.propose_from_embeddings([emb],[.2])[0].residue_logits.detach().cpu().double().numpy()
            retrieved,_=planning.retrieved_proposals(logits,.2,profiles[pid],seed)
            ref,_=planning.mix_candidates(native,retrieved)
            if [api._mask_sha256(len(seq),p['spans']) for p in ref]!=[p['deletion_mask_sha256'] for p in rag]:
                raise ValueError('Recomputed reference RAG pool differs')
            moved,md=morphology(logits,.2,retrieved,seed)
            _,fallback=score_items(seq,native,moved,logits,'rag_morphology_randomized')
            null,nd=null_profile(seq,profiles[pid],20261018+seed)
            nr,_=planning.retrieved_proposals(logits,.2,null,seed)
            _,nf=score_items(seq,native,nr,logits,'rag_joint_null')
            direct=[dict(spans=p['spans'],source_target=p['target'],seed_spans=p['spans'],kind='retrieval_direct')
                for p in planning.seed_sets(profiles[pid],len(seq),.2) if p['complete_natural_plan']][:4]
            _,df=score_items(seq,native,direct,logits,'rag_direct_only')
            selected.append(select_rows(api,seq,rag,'rag_no_scorer',use_policy=True))
            d=dict(protein_id=pid,morphology=md,joint_null=nd,morphology_fallback=fallback,
                joint_fallback=nf,direct_fallback=df,full_RAG_mask=min(rag,key=lambda p:p['scorer_rank'])['deletion_mask_sha256'])
            if pid in subset['E5']:
                reps=profiles[pid]['representatives'];order=sorted(range(len(reps)),key=lambda i:h('v2-scarcity|'+reps[i]['target']))
                d['scarcity']=[]
                for fraction in [.5,.25,0.]:
                    n=math.ceil(len(reps)*fraction);keep=set(order[:n]);prof=dict(profiles[pid],representatives=[r for i,r in enumerate(reps) if i in keep])
                    rp,_=planning.retrieved_proposals(logits,.2,prof,seed)
                    pool,ff=score_items(seq,native,rp,logits,'rag_evidence_'+str(int(fraction*100)))
                    if fraction==0 and [(p['deletion_mask_sha256'],p['scorer_score']) for p in pool]!=[(p['deletion_mask_sha256'],p['scorer_score']) for p in original]:
                        raise ValueError('Zero evidence is not exact original fallback')
                    d['scarcity'].append(dict(fraction=fraction,full_N=len(reps),retained_N=n,actual_input_changed=n!=len(reps),fallback=ff))
            diags.append(d);print('V2_MECHANISMS_DONE '+pid,flush=True)
        full_pool=[];quota=[]
        fullids=read(root/'reports/recomb_confirm_v2_General100_subsets/subsets.json')['full_pool12']
        for item in fullids:
            pid=item['protein_id'];original=sorted(groups[(pid,'delepref_unlimited')],key=lambda r:r['raw_candidate_rank'])
            rag=sorted(groups[(pid,'rag_unlimited')],key=lambda r:r['raw_candidate_rank']);seq=original[0]['wildtype_sequence']
            full_pool.extend(dict(p,full_pool_source=m) for m,pool in [('original',original),('RAG',rag)] for p in pool)
            winner=min(original,key=lambda p:p['scorer_rank']);pool=[];seen=set()
            for p in [winner]+[p for p in rag if p['generation_kind'].startswith('retrieval')]+original:
                if p['deletion_mask_sha256'] not in seen:pool.append(dict(p));seen.add(p['deletion_mask_sha256'])
                if len(pool)==8:break
            if len(pool)!=8:raise ValueError('Original winner preserved K8 control incomplete')
            pool=[dict(p,method='rag_preserve_original_winner',raw_candidate_rank=i+1) for i,p in enumerate(pool)]
            scored=scoring.score_candidates(runtime,api,chunk,pool)
            quota.append(select_rows(api,seq,scored,'rag_preserve_original_winner'))
    write(out/'all_selections.json',selected);write(out/'candidate_scores.json',scores);write(out/'diagnostics.json',diags)
    write(out/'full_pool_candidates.json',full_pool);write(out/'quota_control_selections.json',quota)
    done=dict(status='V2_E4_E5_E7_REAL_HEADS_CONTROLS_COMPLETE_NOT_STRUCTURE_EVALUATED',
        E4_N=len(subset['E4']),E5_N=len(subset['E5']),E7_N=len(fullids),selected_controls=len(selected),
        effective_joint_null_N=sum(d['joint_null']['effective_intervention'] for d in diags),
        at_least_one_exact_final_morphology_match_N=sum(any(x['effective_final_shape_matched_intervention'] for x in d['morphology']) for d in diags),
        new_encoder_forwards=0,new_training=False,new_structure_predictions=0,seconds=time.monotonic()-start,
        outputs={p:sha(out/p) for p in ['all_selections.json','candidate_scores.json','diagnostics.json','full_pool_candidates.json','quota_control_selections.json']},
        job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
