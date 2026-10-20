"""Original learned Scorer features/weights, with the validation span cap removed.

The original span-count scalar remains count/3, preserving all <=3 features.
Scoring >3 spans is explicit training-range extrapolation, not a retrained model.
"""
from collections import defaultdict

import torch


def preference_scores(scorer,candidates,encoded):
    from delepref.student_model import _candidate_intervals,deletion_mask_from_intervals
    if not candidates:return torch.empty(0,device=next(scorer.parameters()).device)
    features=[];dtype=scorer._feature_dtype()
    for candidate in candidates:
        embeddings=encoded.for_candidate(candidate);length=int(embeddings.shape[0])
        spans=_candidate_intervals(candidate,length)  # Only validation change: no qmax.
        if not spans:raise ValueError('at least one deletion required')
        mask=deletion_mask_from_intervals(length,spans,device=embeddings.device,dtype=embeddings.dtype)
        deleted_count=mask.sum();retained_count=(1.-mask).sum()
        if float(retained_count.detach().cpu())<1.:raise ValueError('at least one retained residue required')
        deleted_pool=(embeddings*mask.unsqueeze(-1)).sum(dim=0)/deleted_count
        retained_pool=(embeddings*(1.-mask).unsqueeze(-1)).sum(dim=0)/retained_count
        global_pool=embeddings.mean(dim=0)
        span_raw=torch.stack([scorer._span_feature(embeddings,a,b) for a,b in spans],dim=0)
        span_aggregate=scorer.span_phi(span_raw).mean(dim=0)
        terminal_count=sum(int(a==0)+int(b==length-1) for a,b in spans)
        scalars=embeddings.new_tensor([float(len(spans))/float(scorer.qmax),
            float(candidate.deleted_length)/float(length),float(candidate.target_delete_fraction),float(terminal_count)/2.])
        features.append(torch.cat([global_pool.to(dtype),deleted_pool.to(dtype),retained_pool.to(dtype),
            span_aggregate,scalars.to(dtype)],dim=-1))
    return scorer.plan_head(torch.stack(features,dim=0)).squeeze(-1)


def score_candidates(runtime,api,chunk,rows):
    candidates=[api._candidate_from_row(row) for row in rows]
    with torch.inference_mode():
        values=preference_scores(runtime.model.scorer,candidates,chunk.encoded)
    scores=values.detach().float().cpu().tolist();scored=[];groups=defaultdict(list)
    for index,(row,score) in enumerate(zip(rows,scores)):
        item=dict(row,scorer_score=float(score),scorer_rank=0,
            scorer_span_extrapolation=int(row['span_count'])>3,scorer_training_span_scale=3)
        if not torch.isfinite(torch.tensor(score)):raise ValueError('nonfinite Scorer output')
        scored.append(item);groups[api._condition_key(item)].append(index)
    for indices in groups.values():
        ordered=sorted(indices,key=lambda i:(-scored[i]['scorer_score'],
            api.parse_deleted_intervals(scored[i]['deleted_intervals'],scored[i]['sequence_length'])))
        for rank,index in enumerate(ordered,1):scored[index]['scorer_rank']=rank
    return scored
