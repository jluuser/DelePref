"""Portable inference with the frozen trained Policy/Scorer and one WT encoding."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping
import numpy as np
import torch
from safetensors.torch import load_file
from .paths import packaged_asset, resolve_path
from . import planning
from .plan_scoring import preference_scores
from .preference_core import DeletionCandidate
from .student_model import DeletionPreferenceStudent, EncodedWildTypes, FrozenWTEncoder, wildtype_key


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def condition_seed(seed, protein_id, sequence, budget):
    material='{}\0{}\0{}\0{:.2f}'.format(int(seed),protein_id,sequence_sha(sequence),float(budget))
    return int(hashlib.sha256(material.encode()).hexdigest()[:16],16)&((1<<63)-1)


def mask_sha(length, spans):
    mask=bytearray(length)
    for a,b in spans:mask[a:b+1]=b'\x01'*(b-a+1)
    return hashlib.sha256(bytes(mask)).hexdigest()


def validate_sequence(sequence):
    if not isinstance(sequence,str) or not 2<=len(sequence)<=1022:
        raise ValueError('WT sequence must contain 2-1022 residues; no silent truncation')
    if any(a not in planning.AA for a in sequence):
        raise ValueError('WT input must use the 20 standard uppercase amino acids')
    return sequence


def load_profile(path, protein_id, sequence):
    if path is None:return {'representatives':[]}
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if 'representatives' not in obj:
        if protein_id in obj:obj=obj[protein_id]
        else:
            matches=[p for p in obj.values() if isinstance(p,dict) and
                     p.get('query',{}).get('sequence_sha256')==sequence_sha(sequence)]
            if len(matches)!=1:raise ValueError('Profile has no unambiguous matching WT identity')
            obj=matches[0]
    query=obj.get('query',{})
    if query.get('sequence_sha256') and query['sequence_sha256']!=sequence_sha(sequence):
        raise ValueError('Profile/WT sequence hash mismatch')
    for row in obj.get('representatives',[]):
        projection=row.get('projected','')
        if len(projection)!=len(sequence) or any(a not in planning.AA|{'-','.'} for a in projection):
            raise ValueError('Invalid WT-coordinate homolog projection')
    return obj


def _architecture(config):
    if 'model' not in config:return dict(config)
    # Read the old research YAML without altering it or its path/hash.
    m,p=config['model'],config['policy']
    return dict(hidden_size=1152,max_length=m.get('max_length',1022),
        scorer_hidden_dim=m.get('scorer_hidden_dim',256),span_feature_dim=m.get('span_feature_dim',128),
        budget_hidden_dim=p.get('budget_hidden_dim',64),policy_hidden_dim=p.get('policy_hidden_dim',256),
        policy_span_hidden_dim=p.get('span_hidden_dim',256),training_max_spans=p.get('qmax',3),
        min_span_len=p.get('min_span_len',1),max_span_len_fraction=p.get('max_span_len_fraction',.5),
        dropout=m.get('dropout',.1),encoder_dtype=m.get('esmc_dtype','bfloat16'))


class _EmbeddingOnly(torch.nn.Module):
    def forward(self,*args,**kwargs):
        raise RuntimeError('No sequence encoder was called; supply real residue embeddings or a real ESMC snapshot')


@dataclass
class HeadRuntime:
    model: DeletionPreferenceStudent
    config: dict
    weights_sha256: str


def load_heads(weights=None, config=None):
    weights=resolve_path(weights,environment='DELEPREF_HEADS',default=packaged_asset('heads.safetensors'),exists=True)
    config=resolve_path(config,default=packaged_asset('model.json'),exists=True)
    if config.suffix.lower() in ['.yaml','.yml']:
        import yaml
        cfg=_architecture(yaml.safe_load(config.read_text(encoding='utf-8')))
    else:cfg=_architecture(json.loads(config.read_text(encoding='utf-8')))
    encoder=FrozenWTEncoder(_EmbeddingOnly(),None,hidden_size=int(cfg['hidden_size']),max_length=int(cfg['max_length']))
    model=DeletionPreferenceStudent(encoder,hidden_size=int(cfg['hidden_size']),
        scorer_hidden_dim=int(cfg['scorer_hidden_dim']),span_feature_dim=int(cfg['span_feature_dim']),
        budget_hidden_dim=int(cfg['budget_hidden_dim']),policy_hidden_dim=int(cfg['policy_hidden_dim']),
        policy_span_hidden_dim=int(cfg['policy_span_hidden_dim']),qmax=int(cfg['training_max_spans']),
        min_span_len=int(cfg.get('min_span_len',1)),max_span_len_fraction=float(cfg.get('max_span_len_fraction',.5)),
        dropout=float(cfg.get('dropout',.1)))
    if weights.suffix=='.safetensors':state=load_file(str(weights),device='cpu')
    else:
        payload=torch.load(weights,map_location='cpu',weights_only=True)
        state=payload['student_head_state_dict']
    if len(state)!=32 or any(not k.startswith(('policy.','scorer.')) for k in state):
        raise ValueError('Expected exactly the 32 trained Policy/Scorer tensors')
    model.load_state_dict(state,strict=True)
    if not all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items()):
        raise ValueError('Head values changed during loading')
    model.eval()
    return HeadRuntime(model,cfg,file_sha(weights))


def _score(runtime,protein_id,sequence,budget,items,logits,seed,embeddings):
    key=wildtype_key(protein_id,sequence)
    encoded=EncodedWildTypes(embeddings={key:embeddings},sequences={key:sequence},input_count=1)
    candidates=[];rows=[]
    for rank,item in enumerate(items,1):
        spans=tuple(tuple(s) for s in item['spans']);mask=planning.validate_plan(len(sequence),budget,spans)
        digest=mask_sha(len(sequence),spans)
        candidate=DeletionCandidate(protein_id=protein_id,perturbation_id=digest,plan_id=digest,
            wildtype_sequence=sequence,deleted_intervals=spans,deleted_length=int(mask.sum()),
            deletion_fraction=float(mask.sum())/len(sequence),target_delete_fraction=budget)
        candidates.append(candidate)
        rows.append(dict(raw_candidate_rank=rank,deleted_intervals=[list(s) for s in spans],
            deletion_mask_sha256=digest,deleted_length=int(mask.sum()),span_count=len(spans),
            generation_kind=item['kind'],retrieval_source=item.get('source_target'),
            retrieval_seed_spans=item.get('seed_spans'),condition_seed=seed,
            policy_energy=float(logits[mask].mean()),scorer_span_extrapolation=len(spans)>3))
    with torch.inference_mode():scores=preference_scores(runtime.model.scorer,candidates,encoded).cpu().float().tolist()
    for row,score in zip(rows,scores):
        if not np.isfinite(score):raise ValueError('Non-finite complete-plan score')
        row['scorer_score']=float(score)
    order=sorted(range(len(rows)),key=lambda i:(-rows[i]['scorer_score'],tuple(map(tuple,rows[i]['deleted_intervals']))))
    for rank,i in enumerate(order,1):rows[i]['scorer_rank']=rank
    return rows


def plan_from_embeddings(runtime,protein_id,sequence,embedding,*,budget=.2,seed=13,profile=None):
    validate_sequence(sequence)
    if not 0<float(budget)<1:raise ValueError('Deletion fraction must be strictly between 0 and 1')
    if tuple(embedding.shape)!=(len(sequence),int(runtime.config['hidden_size'])) or not torch.isfinite(embedding).all():
        raise ValueError('Embedding shape/values do not match this WT')
    # Preserve output dtype. The formal residue cache is float32 after a bf16 encoder forward.
    embedding=embedding.detach().cpu()
    profile=profile or {'representatives':[]}
    for r in profile.get('representatives',[]):
        if len(r['projected'])!=len(sequence):raise ValueError('Profile projection length mismatch')
    derived_seed=condition_seed(seed,protein_id,sequence,budget)
    with torch.inference_mode():
        logits=runtime.model.propose_from_embeddings([embedding],[budget])[0].residue_logits.cpu().double().numpy()
    original,cost=planning.policy_proposals(logits,budget,derived_seed,k=8,
        perturbation_scale=float(runtime.config.get('perturbation_scale',1.)),max_attempts=int(runtime.config.get('max_attempts',128)))
    if len(original)!=8:raise RuntimeError('Could not obtain eight distinct feasible Policy candidates')
    base=_score(runtime,protein_id,sequence,budget,original,logits,derived_seed,embedding)
    retrieved,attempts=planning.retrieved_proposals(logits,budget,profile,derived_seed)
    mixed,fallback=planning.mix_candidates(original,retrieved)
    guided=[dict(r) for r in base] if fallback else _score(runtime,protein_id,sequence,budget,mixed,logits,derived_seed,embedding)
    if len(guided)!=8:raise RuntimeError('Guided pool must have exactly eight distinct masks')
    selected={}
    for label,pool in [('base',base),('guided',guided)]:
        best=dict(min(pool,key=lambda r:r['scorer_rank']))
        mask=planning.validate_plan(len(sequence),budget,tuple(map(tuple,best['deleted_intervals'])))
        edited=''.join(a for a,removed in zip(sequence,mask) if not removed)
        best.update(edited_sequence=edited,edited_sequence_sha256=sequence_sha(edited))
        selected[label]=best
    return dict(schema='delepref-result-v1',protein_id=protein_id,WT_sha256=sequence_sha(sequence),WT_length=len(sequence),
        budget=float(budget),seed=int(seed),condition_seed=derived_seed,coordinate_system='0_based_closed_inclusive',
        training_span_scale=3,inference_max_spans=None,weights_sha256=runtime.weights_sha256,
        fallback_exact=fallback,original_generation=cost,retrieval_attempts=attempts,
        candidates={'base':base,'guided':guided},selected=selected)


def encode_sequences(records,snapshot,*,device='cuda',dtype='bfloat16',microbatch_size=32,progress=None):
    from .preference_core import load_esmc_official_safetensors_model
    snapshot=resolve_path(snapshot,environment='DELEPREF_ESMC_SNAPSHOT',exists=True)
    if device!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Fresh ESMC inference requires CUDA; use the explicitly cached CPU demo otherwise')
    descriptor=json.loads(packaged_asset('model.json').read_text())
    if file_sha(snapshot/'model.safetensors')!=descriptor['encoder_weights_sha256']:
        raise ValueError('ESMC weights differ from the pinned encoder; no silent encoder replacement')
    backend=load_esmc_official_safetensors_model(str(snapshot),dtype=dtype,device=device,max_length=1022,require_cuda=True)
    if progress:progress('ESMC_MODEL_LOADED')
    encoder=FrozenWTEncoder(backend,backend.tokenizer,hidden_size=1152,max_length=1022)
    for pid,seq in records:validate_sequence(seq)
    if progress:progress('WT_FORWARD_START')
    with torch.inference_mode():encoded=encoder.encode_unique_wildtypes([p for p,s in records],[s for p,s in records],microbatch_size=microbatch_size)
    if progress:progress('WT_FORWARD_COMPLETE')
    return {key.protein_id:h.detach().cpu() for key,h in encoded.embeddings.items()}
