"""Apply pre-existing mechanism controls only to reviewed queries and evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('reports/recomb_benign_mechanisms_20261017')
FREEZE=Path('configs/recomb_benign_mechanisms_freeze_20261017.json')


def prepare(root):
    allowed={r['protein_id'] for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    old_sub=read(root/'reports/delepref_v8_evaluation_preparation/subsets.json')['completion_ablation']
    old_panel=read(root/'reports/delepref_v8_native_panel/main_panel.json')
    scarcity=[]
    for source in ['cath_domain','pdb_chain']:
        rows=[r for r in old_panel if r['protein_id'] in old_sub and r['stratum']==source]
        rows.sort(key=lambda r:hashlib.sha256(('recomb-scarcity-20261017|'+r['sequence_sha256']).encode()).hexdigest())
        scarcity.extend(r['protein_id'] for r in rows[:5] if r['protein_id'] in allowed)
    selected=[pid for pid in old_sub if pid in allowed]
    out=root/BASE;out.mkdir(exist_ok=False)
    paths=['analysis/recomb_benign_mechanisms_20261017.py','analysis/recomb_general100_mechanisms_20261017.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
        'reports/recomb_benign_scope_20261017/general_panel.json','reports/delepref_v8_evaluation_preparation/subsets.json',
        'reports/recomb_benign_plans_20261017/candidate_scores.json','reports/recomb_benign_plans_20261017/planning_completion.json',
        'retrieval/recomb_benign_general_20261017/profiles.json','configs/recomb_benign_plans_freeze_20261017.json']
    write(root/FREEZE,dict(status='SCREENED_CONTROLS_FROZEN_BEFORE_SELECTION',files={p:sha(root/p) for p in paths},
        mechanism_ids=selected,scarcity_ids=scarcity,subset_rule='intersection_of_original_input_hash_subsets_with_positive_scope',
        K=8,budget=.2,seed=13,null_seed=20261017,new_training=False,structure_prediction_allowed=False))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE),mechanism_WT=len(selected),scarcity_WT=len(scarcity))))


def run(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if sha(root/FREEZE)!=checksum:raise ValueError('freeze changed')
    frozen=read(root/FREEZE)
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('input/source changed')
    import torch
    import recomb_general100_mechanisms_20261017 as old
    import delepref_v8_main as parent
    import delepref_v8_runtime as rt
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    parent.verify(root,read(root/'configs/recomb_benign_plans_freeze_20261017.json')['parent_main_freeze_sha256'])
    allowed={r['protein_id']:r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    ids=frozen['mechanism_ids'];scarcity=set(frozen['scarcity_ids'])
    if not set(ids).issubset(allowed) or not scarcity.issubset(ids):raise ValueError('unreviewed control query')
    records=[dict(protein_id=pid,wildtype_sequence=allowed[pid]['sequence'],budget=.2) for pid in ids]
    path=root/'reports/delepref_v8_main/wt_cache.pt'
    if sha(path)!=read(root/'reports/delepref_v8_main/encoding_completion.json')['cache_sha256']:raise ValueError('cache changed')
    cache=torch.load(path,map_location='cpu',weights_only=True)
    for k in ['embeddings','sequences']:cache[k]={p:v for p,v in cache[k].items() if p in ids}
    torch.set_num_threads(4);runtime,api=rt.load(root,device='cpu',cached_heads_only=True);chunk=old.cached_chunk(api,cache,records)
    profiles=read(root/'retrieval/recomb_benign_general_20261017/profiles.json');groups=defaultdict(list)
    for row in read(root/'reports/recomb_benign_plans_20261017/candidate_scores.json'):
        if row['protein_id'] in ids and float(row['budget'])==.2 and row['generation_seed']==13:groups[(row['protein_id'],row['method'])].append(row)
    selections=[];candidates=[];diagnostics=[]
    with torch.inference_mode():
        for pid in ids:
            original=sorted(groups[(pid,'delepref_unlimited')],key=lambda r:r['raw_candidate_rank'])
            rag=sorted(groups[(pid,'rag_unlimited')],key=lambda r:r['raw_candidate_rank'])
            template=original[0];sequence=allowed[pid]['sequence'];seed=int(template['condition_seed'])
            embedding=chunk.encoded.embeddings[chunk.encoded.key_for(pid,sequence)]
            logits=runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
            native=[dict(spans=tuple(map(tuple,json.loads(r['deleted_intervals']))),kind=r['generation_kind'],row=r) for r in original]
            marginal,md=old.marginal_profile(sequence,profiles[pid],seed);null,nd=old.null_profile(sequence,profiles[pid],20261017+seed)
            variants=[('rag_marginal_gap',marginal,md),('rag_cooccurrence_null',null,nd)]
            if pid in scarcity:
                reps=sorted(profiles[pid]['representatives'],key=lambda r:hashlib.sha256(('recomb-scarcity-homolog-20261017|'+r['target']).encode()).hexdigest())
                variants.append(('rag_evidence_min8',dict(profiles[pid],representatives=reps[:8]),dict(full_N=len(reps),reduced_N=min(8,len(reps)))))
                zero,flag=planning.mix_candidates(native,[])
                if not flag or [r['row'] for r in zero]!=original:raise ValueError('zero evidence mismatch')
                selections.append(old.select_rows(api,sequence,original,'rag_evidence_zero'))
            for method,profile,intervention in variants:
                retrieved,attempts=planning.retrieved_proposals(logits,.2,profile,seed);mixed,fallback=planning.mix_candidates(native,retrieved)
                scored=[dict(r) for r in original] if fallback else scoring.score_candidates(runtime,api,chunk,old.make_rows(api,template,mixed,len(sequence),logits))
                chosen=old.select_rows(api,sequence,scored,method)
                chosen['evidence_is_counterfactual']=method in ('rag_marginal_gap','rag_cooccurrence_null')
                if method=='rag_cooccurrence_null' and not nd['effective_intervention'] and chosen['deletion_mask_sha256']!=old.select_rows(api,sequence,rag,'reference')['deletion_mask_sha256']:raise ValueError('unperturbed null changed winner')
                selections.append(chosen);candidates.extend(dict(r,method=method) for r in scored)
                diagnostics.append(dict(protein_id=pid,method=method,intervention=intervention,fallback=fallback,retrieved_proposals=len(retrieved)))
            selections.append(old.select_rows(api,sequence,rag,'rag_no_scorer',use_policy=True))
    if len(selections)!=3*len(ids)+2*len(scarcity):raise ValueError('incomplete controls')
    out=root/BASE;write(out/'all_selections.json',selections);write(out/'candidate_scores.json',candidates);write(out/'diagnostics.json',diagnostics)
    report=dict(status='SCREENED_MECHANISM_CONTROLS_COMPLETE',WT=len(ids),scarcity_WT=len(scarcity),selected_outputs=len(selections),
        effective_null_WT=sum(d['intervention'].get('effective_intervention',False) for d in diagnostics if d['method']=='rag_cooccurrence_null'),
        effective_min8_WT=sum(d['intervention']['full_N']!=d['intervention']['reduced_N'] for d in diagnostics if d['method']=='rag_evidence_min8'),
        no_effective_intervention_is_not_evidence_for_or_against_joint_mechanism=True,new_encoder_forwards=0,new_structure_predictions=0,
        freeze_sha256=checksum,job_id=os.environ['SLURM_JOB_ID'],outputs={n:sha(out/n) for n in ['all_selections.json','candidate_scores.json','diagnostics.json']})
    write(out/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args();prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256)
