"""Validate SCISOR's explicit preserved-token coordinates against WT residues."""
import numpy as np
import delepref_unlimited_planning as planning


def mapped_plan(sequence,budget,preserved,output_ids,tokenizer):
    n=len(sequence);count=planning.delete_count(n,budget)
    if any(not isinstance(i,(int,np.integer)) or isinstance(i,bool) for i in preserved):raise ValueError('noninteger preserved index')
    indices=list(map(int,preserved))
    if len(indices)!=n+2-count or not indices or indices[0]!=0 or indices[-1]!=n+1:
        raise ValueError('preserved token count or special-token endpoints invalid')
    if any(b<=a for a,b in zip(indices,indices[1:])) or any(i<0 or i>n+1 for i in indices):
        raise ValueError('preserved coordinates are not strictly increasing')
    kept=np.array([i-1 for i in indices[1:-1]],dtype=int)
    mask=np.ones(n,dtype=bool);mask[kept]=False
    spans=planning.intervals(mask);planning.validate_plan(n,budget,spans)
    ids=list(map(int,output_ids))
    if len(ids)!=n-count+2 or ids[0]!=tokenizer.cls_token_id or ids[-1]!=tokenizer.eos_token_id:
        raise ValueError('output token count/endpoints invalid')
    tokens=tokenizer.convert_ids_to_tokens(ids[1:-1])
    if any(t not in planning.AA for t in tokens):raise ValueError('nonstandard residue in SCISOR output')
    edited=''.join(tokens);expected=''.join(sequence[i] for i in kept)
    if edited!=expected:raise ValueError('SCISOR token output differs from explicit WT deletion map')
    return dict(deleted_intervals=spans,deleted_length=count,span_count=len(spans),
        kept_wt_indices=kept.tolist(),edited_sequence=edited,qmax=None)
