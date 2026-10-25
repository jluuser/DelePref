"""Frozen complete-plan features with unlimited input spans; unchanged weights and count/3 scale."""
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
