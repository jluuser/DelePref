"""Score already generated, reviewed SCISOR K8 pools with unchanged DelePref heads."""
import argparse
import json
import os
from pathlib import Path
from recomb_scope_screen_20261017 import read, sha, write

BASE=Path('reports/recomb_benign_scisor_score_20261017')
FREEZE=Path('configs/recomb_benign_scisor_score_freeze_20261017.json')


def prepare(root):
    (root/BASE).mkdir(exist_ok=False)
    paths=['reports/recomb_benign_baselines_20261017/scisor/all_conditions.json',
           'reports/recomb_benign_baselines_20261017/scisor/completion.json',
           'reports/recomb_benign_scope_20261017/general_panel.json',
           'reports/recomb_benign_scope_20261017/retained_subsets.json',
           'reports/recomb_benign_plans_20261017/candidate_scores.json',
           'reports/recomb_benign_plans_20261017/planning_completion.json',
           'configs/recomb_benign_plans_freeze_20261017.json',
           'analysis/recomb_benign_scisor_score_20261017.py']
    write(root/FREEZE,dict(status='REVIEWED_SCISOR_K8_SCORING_FROZEN',files={p:sha(root/p) for p in paths},
        selector='unchanged_DelePref_Scorer',new_generation=False,new_structure_predictions=0))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE))))


def run(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm required')
    if sha(root/FREEZE)!=checksum:raise ValueError('freeze changed')
    for p,h in read(root/FREEZE)['files'].items():
        if sha(root/p)!=h:raise ValueError('input changed')
    import torch
    import delepref_v8_main as parent
    import delepref_v8_runtime as rt
    import delepref_unlimited_scorer as scoring
    import delepref_unlimited_planning as planning
    parent.verify(root,read(root/'configs/recomb_benign_plans_freeze_20261017.json')['parent_main_freeze_sha256'])
    allowed={r['protein_id']:r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    expected=set(read(root/'reports/recomb_benign_scope_20261017/retained_subsets.json')['completion_ablation'])
    tasks=[t for t in read(root/'reports/recomb_benign_baselines_20261017/scisor/all_conditions.json') if t['mode']=='same_scorer_K8']
    if {t['protein_id'] for t in tasks}!=expected or len(tasks)!=7:raise ValueError('wrong screened K8 scope')
    for t in tasks:
        if t['sequence']!=allowed[t['protein_id']]['sequence'] or len(t['candidates'])!=8:raise ValueError('WT/K mismatch')
    cp=root/'reports/delepref_v8_main/wt_cache.pt'
    if sha(cp)!=read(root/'reports/delepref_v8_main/encoding_completion.json')['cache_sha256']:raise ValueError('cache changed')
    cache=torch.load(cp,map_location='cpu',weights_only=True)
    for key in ['embeddings','sequences']:cache[key]={p:v for p,v in cache[key].items() if p in expected}
    templates={}
    for row in read(root/'reports/recomb_benign_plans_20261017/candidate_scores.json'):
        if row['method']=='delepref_unlimited' and row['budget']==.2 and row['generation_seed']==13 and row['protein_id'] in expected:
            templates.setdefault(row['protein_id'],row)
    torch.set_num_threads(4);runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    chunk=parent.cached_chunk(api,cache,list(templates.values()))
    scored,selections=[],[]
    with torch.inference_mode():
        for task in tasks:
            pid=task['protein_id'];wt=task['sequence'];template=dict(templates[pid],method='SCISOR_K8_same_scorer')
            embedding=chunk.encoded.embeddings[chunk.encoded.key_for(pid,wt)]
            logits=runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
            items=[dict(spans=c['deleted_intervals'],kind='scisor_native_sampling') for c in task['candidates']]
            rows=parent.make_rows(api,template,items,len(wt),logits)
            for row,c in zip(rows,task['candidates']):
                mask=planning.validate_plan(len(wt),.2,c['deleted_intervals'])
                edited=''.join(a for a,d in zip(wt,mask) if not d)
                if edited!=c['edited_sequence'] or row['deletion_mask_sha256']!=c['deletion_mask_sha256']:raise ValueError('native reconstruction mismatch')
                row.update(candidate_index=c['candidate_index'],edited_sequence=edited,
                    edited_sequence_sha256=c['edited_sequence_sha256'],decoder='SCISOR_native_explicit_preserved_indices',
                    original_sampler_seed=c.get('condition_seed'),reused_native_candidate=c.get('reused_native',False))
            pool=scoring.score_candidates(runtime,api,chunk,rows)
            best=dict(min(pool,key=lambda r:r['scorer_rank']),effective_unique_K=len({r['deletion_mask_sha256'] for r in pool}))
            scored.extend(pool);selections.append(best)
    out=root/BASE
    write(out/'candidate_scores.json',scored);write(out/'all_selections.json',selections)
    write(out/'completion.json',dict(status='SCREENED_SCISOR_COMMON_SCORER_COMPLETE',WT=7,candidates=len(scored),selected=len(selections),
        effective_unique_K=[s['effective_unique_K'] for s in selections],all_native_reconstructions_checked=True,
        freeze_sha256=checksum,job_id=os.environ['SLURM_JOB_ID'],extra_PLM_forwards=0,new_sampler_calls=0,new_structure_predictions=0,
        outputs={p.name:sha(p) for p in out.glob('*.json') if p.name!='completion.json'}))
    print(json.dumps(read(out/'completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256)
