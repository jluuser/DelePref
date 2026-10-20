"""Sequence-only retrieved-event proposals and exact constrained completion."""
from collections import deque
import hashlib
import itertools
import math

import numpy as np

AA = set('ACDEFGHIKLMNPQRSTVWY')
CONTRACT = dict(K=8, original_slots=4, retrieved_slots=4, qmax=3,
    max_events_per_homolog=12, max_seed_sets=16, refined_passes=1,
    min_span_len=1, max_span_fraction=.5, anchor_window=3, anchor_standard_min=2,
    terminal_gaps='unknown', combine_different_homologs=False)


class InfeasibleCompletion(ValueError):
    pass


def intervals(mask):
    a=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,a,False].astype(np.int8))
    return tuple(zip(np.flatnonzero(edges==1).tolist(),(np.flatnonzero(edges==-1)-1).tolist()))


def mask_for(length, spans):
    m=np.zeros(length,dtype=bool)
    previous=-2
    for a,b in spans:
        if not 0<=a<=b<length or a<=previous+1:
            raise ValueError('invalid/non-maximal spans')
        m[a:b+1]=True;previous=b
    return m


def delete_count(length,beta):
    if length<2 or not math.isfinite(beta) or not 0<=beta<=1:
        raise ValueError('invalid length/budget')
    return max(1,min(length-1,round(beta*length)))


def constrained_dp(logits,beta,must_delete,must_keep,qmax=3,min_span=1,max_span=None):
    """Exact O(L*D*Q) interval DP with hard deletion and retained-anchor constraints."""
    values=np.asarray(logits,dtype=np.float64).reshape(-1)
    n=len(values);d=delete_count(n,beta)
    forced=np.asarray(must_delete,dtype=bool);kept=np.asarray(must_keep,dtype=bool)
    if forced.shape!=(n,) or kept.shape!=(n,) or not np.isfinite(values).all():
        raise ValueError('bad logits or constraint dimensions')
    if np.any(forced&kept):
        raise InfeasibleCompletion('conflicting constraints')
    max_span=min(n-1,d if max_span is None else int(max_span))
    if qmax<1 or min_span<1 or max_span<min_span:
        raise ValueError('invalid span limits')
    if forced.sum()>d or n-kept.sum()<d:
        raise InfeasibleCompletion('impossible deletion count')
    prefix=np.r_[0.,np.cumsum(values)]
    best=np.full((qmax+1,n+1,d+1),-np.inf)
    choice=np.full(best.shape,-1,dtype=np.int32)
    first_forced=int(np.flatnonzero(forced)[0]) if forced.any() else n
    best[0,:first_forced+1,0]=0.;choice[0,:first_forced+1,0]=0
    last_kept=np.maximum.accumulate(np.where(kept,np.arange(n),-1))
    for q in range(1,qmax+1):
        end=np.full((n+1,d+1),-np.inf)
        endlen=np.zeros((n+1,d+1),dtype=np.int32)
        for retained in range(n+1):
            window=deque()
            for i in range(max(1,retained),min(n,retained+d)+1):
                deleted=i-retained;a=i-min_span
                if a>=0 and 0<=a-retained<=d and (a==0 or not forced[a-1]):
                    prev=0 if a==0 else a-1
                    old=best[q-1,prev,a-retained]
                    if np.isfinite(old):
                        candidate=float(old-prefix[a])
                        while window and candidate>window[-1][1]:window.pop()
                        window.append((a,candidate))
                minimum=max(i-max_span,int(last_kept[i-1])+1)
                while window and window[0][0]<minimum:window.popleft()
                if window and 0<=deleted<=d:
                    a,v=window[0];end[i,deleted]=prefix[i]+v;endlen[i,deleted]=i-a
        for deleted in range(1,d+1):
            for i in range(1,n+1):
                carry=best[q,i-1,deleted] if not forced[i-1] else -np.inf
                if end[i,deleted]>carry:
                    best[q,i,deleted]=end[i,deleted];choice[q,i,deleted]=endlen[i,deleted]
                elif np.isfinite(carry):
                    best[q,i,deleted]=carry;choice[q,i,deleted]=0
    q=1+int(np.argmax(best[1:,n,d]));score=float(best[q,n,d])
    if not np.isfinite(score):raise InfeasibleCompletion('no feasible completion')
    spans=[];i=n;remaining=d
    while q or remaining:
        if i<=0:raise RuntimeError('invalid completion backtrack')
        size=int(choice[q,i,remaining])
        if size==0:i-=1;continue
        if size<0:raise RuntimeError('unreachable completion backtrack')
        a=i-size;spans.append((a,i-1));remaining-=size;q-=1;i=0 if a==0 else a-1
    spans=tuple(reversed(spans));m=mask_for(n,spans)
    if m.sum()!=d or np.any(forced&~m) or np.any(kept&m):
        raise RuntimeError('completion violates hard constraints')
    if not np.isclose(values[m].sum(),score,rtol=1e-12,atol=1e-12):
        raise RuntimeError('completion score mismatch')
    return spans,score


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
            events=sorted(events,key=lambda x:(-(x[1]-x[0]+1),x))[:CONTRACT['max_events_per_homolog']]
            result.append(dict(target=h['target'],events=tuple(sorted(events)),identity=h.get('identity'),
                evalue=h.get('evalue'),bits=h.get('bits'),projected=row))
    return result


def seed_sets(profile,length,beta):
    k=delete_count(length,beta);maxlen=max(1,math.floor(length*.5))
    result={}
    for rank,h in enumerate(natural_events(profile,length)):
        events=[s for s in h['events'] if s[1]-s[0]+1<=maxlen]
        for q in range(1,min(3,len(events))+1):
            for spans in itertools.combinations(events,q):
                deleted=sum(b-a+1 for a,b in spans)
                if deleted>k or spans in result:continue
                result[spans]=dict(spans=spans,deleted=deleted,target=h['target'],homolog_rank=rank,
                    complete_natural_plan=deleted==k,identity=h['identity'])
    ranked=sorted(result.values(),key=lambda x:(not x['complete_natural_plan'],-x['deleted'],x['homolog_rank'],len(x['spans']),x['spans']))
    return ranked[:CONTRACT['max_seed_sets']]


def retrieved_proposals(logits,beta,profile,condition_seed):
    n=len(logits);seeds=seed_sets(profile,n,beta);result=[];seen=set();attempts=[]
    for item in seeds:
        forced=mask_for(n,item['spans']);kept=np.zeros(n,dtype=bool)
        for a,b in item['spans']:kept[a-1]=True;kept[b+1]=True
        for refinement in range(2):
            if item['complete_natural_plan'] and refinement:continue
            values=np.asarray(logits,dtype=np.float64)
            if refinement:
                salt=int(hashlib.sha256((item['target']+str(item['spans'])).encode()).hexdigest()[:8],16)
                values=values+np.random.default_rng((condition_seed+salt)%(2**32)).standard_normal(n)
            try:
                spans,score=constrained_dp(values,beta,forced,kept,max_span=max(1,math.floor(n*.5)))
            except InfeasibleCompletion as error:
                attempts.append(dict(target=item['target'],seed_spans=item['spans'],refinement=refinement,status='infeasible',reason=str(error)))
                continue
            duplicate=spans in seen
            attempts.append(dict(target=item['target'],seed_spans=item['spans'],refinement=refinement,status='duplicate' if duplicate else 'accepted'))
            if duplicate:continue
            seen.add(spans)
            result.append(dict(spans=spans,source_target=item['target'],seed_spans=item['spans'],
                kind='retrieval_direct' if item['complete_natural_plan'] else ('retrieval_refined' if refinement else 'retrieval_completed'),
                policy_energy=float(np.asarray(logits)[mask_for(n,spans)].mean())))
            if len(result)>=4:return result,attempts
    return result,attempts


def mix_candidates(original,retrieved):
    """No usable retrieval returns the exact original ordered objects."""
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
