"""Exact-budget deletion planning with no deletion-span count/length cap."""
import hashlib
import math

import numpy as np

AA = set('ACDEFGHIKLMNPQRSTVWY')
CONTRACT = dict(K=8, original_slots=4, retrieved_slots=4, qmax=None,
    max_span_len=None, max_events_per_homolog=None, max_seed_sets=16,
    refined_passes=1, min_span_len=1, anchor_window=3, anchor_standard_min=2,
    terminal_gaps='unknown', combine_different_homologs=False,
    decoder='exact_budget_top_residue_logits', scorer_training_span_scale=3,
    original_weights_unchanged=True, greater_than_3_spans='training_range_extrapolation')


class InfeasibleCompletion(ValueError):
    pass


def intervals(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(np.int8))
    return tuple(zip(np.flatnonzero(edges==1).tolist(),(np.flatnonzero(edges==-1)-1).tolist()))


def mask_for(length, spans):
    mask=np.zeros(length,dtype=bool);previous=-2
    for a,b in spans:
        if not 0<=a<=b<length or a<=previous+1:
            raise ValueError('invalid/non-maximal spans')
        mask[a:b+1]=True;previous=b
    return mask


def delete_count(length,beta):
    if length<2 or not math.isfinite(beta) or not 0<=beta<=1:
        raise ValueError('invalid length/budget')
    return max(1,min(length-1,round(beta*length)))


def validate_plan(length,beta,spans):
    mask=mask_for(length,spans)
    if mask.sum()!=delete_count(length,beta):
        raise ValueError('incorrect deletion budget')
    return mask


def complete(logits,beta,must_delete=None,must_keep=None):
    """Optimal additive objective: forced deletions plus highest free logits.

    O(L log L); ties resolve by residue coordinate. No segment-count constraint.
    """
    values=np.asarray(logits,dtype=np.float64).reshape(-1);n=len(values)
    d=delete_count(n,beta)
    forced=np.zeros(n,dtype=bool) if must_delete is None else np.asarray(must_delete,dtype=bool)
    kept=np.zeros(n,dtype=bool) if must_keep is None else np.asarray(must_keep,dtype=bool)
    if forced.shape!=(n,) or kept.shape!=(n,) or not np.isfinite(values).all():
        raise ValueError('bad logits or constraint dimensions')
    if np.any(forced&kept):raise InfeasibleCompletion('conflicting constraints')
    remaining=d-int(forced.sum());free=np.flatnonzero(~(forced|kept))
    if remaining<0 or remaining>len(free):raise InfeasibleCompletion('impossible deletion count')
    ordered=free[np.lexsort((free,-values[free]))]
    mask=forced.copy();mask[ordered[:remaining]]=True
    return intervals(mask),float(values[mask].sum())


def natural_events(profile,length):
    result=[]
    for h in profile.get('representatives',[]):
        row=h['projected']
        if len(row)!=length:raise ValueError('homolog projection length mismatch')
        events=[]
        for a,b in intervals([c=='-' for c in row]):
            if a==0 or b==length-1:continue
            left=row[max(0,a-3):a];right=row[b+1:min(length,b+4)]
            if row[a-1] not in AA or row[b+1] not in AA:continue
            if sum(c in AA for c in left)<min(2,len(left)) or sum(c in AA for c in right)<min(2,len(right)):continue
            events.append((a,b))
        if events:
            result.append(dict(target=h['target'],events=tuple(events),identity=h.get('identity'),
                evalue=h.get('evalue'),bits=h.get('bits')))
    return result


def whole_event_subset(events,budget):
    """Maximise natural deletion coverage <=budget using intact events.

    At most one immutable predecessor chain per count; deterministic first reach.
    No limit on selected event count, and no combining different homologs.
    """
    paths={0:()}
    for event in events:
        size=event[1]-event[0]+1
        for count in sorted(tuple(paths),reverse=True):
            new=count+size
            if new<=budget and new not in paths:paths[new]=paths[count]+(event,)
    return paths[max(paths)]


def seed_sets(profile,length,beta):
    k=delete_count(length,beta);result={}
    for rank,h in enumerate(natural_events(profile,length)):
        spans=whole_event_subset(h['events'],k)
        if not spans or spans in result:continue
        deleted=sum(b-a+1 for a,b in spans)
        result[spans]=dict(spans=spans,deleted=deleted,target=h['target'],homolog_rank=rank,
            complete_natural_plan=deleted==k,all_reliable_events_used=spans==h['events'],
            reliable_event_count=len(h['events']),identity=h['identity'])
    ranked=sorted(result.values(),key=lambda x:(not x['complete_natural_plan'],-x['deleted'],x['homolog_rank'],x['spans']))
    return ranked[:CONTRACT['max_seed_sets']]


def retrieved_proposals(logits,beta,profile,condition_seed):
    n=len(logits);result=[];seen=set();attempts=[]
    for item in seed_sets(profile,n,beta):
        forced=mask_for(n,item['spans']);kept=np.zeros(n,dtype=bool)
        for a,b in item['spans']:kept[a-1]=True;kept[b+1]=True
        for refinement in range(2):
            if item['complete_natural_plan'] and refinement:continue
            values=np.asarray(logits,dtype=np.float64)
            if refinement:
                salt=int(hashlib.sha256((item['target']+str(item['spans'])).encode()).hexdigest()[:8],16)
                values=values+np.random.default_rng((condition_seed+salt)%(2**32)).standard_normal(n)
            try:spans,_=complete(values,beta,forced,kept)
            except InfeasibleCompletion as error:
                attempts.append(dict(target=item['target'],seed_spans=item['spans'],refinement=refinement,status='infeasible',reason=str(error)))
                continue
            duplicate=spans in seen
            attempts.append(dict(target=item['target'],seed_spans=item['spans'],refinement=refinement,
                seed_span_count=len(item['spans']),output_span_count=len(spans),status='duplicate' if duplicate else 'accepted'))
            if duplicate:continue
            seen.add(spans)
            result.append(dict(spans=spans,source_target=item['target'],seed_spans=item['spans'],
                all_reliable_events_used=item['all_reliable_events_used'],
                kind='retrieval_direct' if item['complete_natural_plan'] else ('retrieval_refined' if refinement else 'retrieval_completed'),
                policy_energy=float(np.asarray(logits)[mask_for(n,spans)].mean())))
            if len(result)>=CONTRACT['retrieved_slots']:return result,attempts
    return result,attempts


def policy_proposals(logits,beta,condition_seed,k=8,perturbation_scale=1.,max_attempts=128):
    """Use the trained Policy's logits; retain original seeded-noise convention."""
    import torch
    base=torch.as_tensor(np.asarray(logits,dtype=np.float64).copy(),dtype=torch.float64)
    result=[];seen=set();duplicates=0
    for attempt in range(max_attempts):
        if attempt==0 or perturbation_scale==0:values=base
        else:
            generator=torch.Generator(device='cpu').manual_seed(int(condition_seed)+attempt*1_000_003)
            values=base+perturbation_scale*torch.randn(base.shape,generator=generator,dtype=base.dtype)
        spans,_=complete(values.numpy(),beta)
        if spans in seen:duplicates+=1;continue
        seen.add(spans)
        result.append(dict(spans=spans,kind='original_policy_unlimited',
            policy_energy=float(np.asarray(logits)[mask_for(len(base),spans)].mean())))
        if len(result)>=k:break
    return result,dict(attempts=attempt+1,duplicates=duplicates,requested=k,returned=len(result))


def mix_candidates(original,retrieved):
    if not retrieved:return list(original),True
    result=[];seen=set()
    def add(item):
        key=tuple(tuple(s) for s in item['spans'])
        if key not in seen:seen.add(key);result.append(item)
    for item in original[:4]:add(item)
    for item in retrieved[:4]:add(item)
    for item in original:
        if len(result)>=8:break
        add(item)
    return result,False
