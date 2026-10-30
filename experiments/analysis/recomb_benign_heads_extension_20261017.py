"""Compute only newly required budget/direct-only conditions from reviewed cached WTs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('reports/recomb_benign_heads_extension_20261017')
FREEZE=Path('configs/recomb_benign_heads_extension_freeze_20261017.json')


def prepare(root):
    sub=read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json')
    previous=read(root/'reports/recomb_benign_scope_20261017/retained_subsets.json')
    added=[p for p in sub['completion_ablation'] if p not in previous['completion_ablation']]
    if len(added)!=13:raise ValueError('unexpected delta roster')
    (root/BASE).mkdir(exist_ok=False)
    paths=['analysis/recomb_benign_heads_extension_20261017.py',
        'reports/recomb_benign_final_subsets_20261017/subsets.json','reports/recomb_benign_scope_20261017/general_panel.json',
        'reports/recomb_benign_plans_20261017/candidate_scores.json','configs/recomb_benign_plans_freeze_20261017.json',
        'retrieval/recomb_benign_general_20261017/profiles.json']
    write(root/FREEZE,dict(status='REVIEWED_HEAD_DELTA_FROZEN',files={p:sha(root/p) for p in paths},
        added_WT=added,budgets=[.1,.3,.4],direct_only_budget=.2,seed=13,new_encoder_forwards=0,new_structure_predictions=0))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE))))


def run(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if sha(root/FREEZE)!=checksum:raise ValueError('freeze changed')
    frozen=read(root/FREEZE)
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('input changed')
    import torch
    import delepref_v8_main as parent
    import delepref_v8_runtime as rt
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    import recomb_general100_mechanisms_20261017 as controls
    parent.verify(root,read(root/'configs/recomb_benign_plans_freeze_20261017.json')['parent_main_freeze_sha256'])
    allowed={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')};ids=frozen['added_WT']
    if not set(ids).issubset(allowed):raise ValueError('unreviewed delta query')
    records=[dict(protein_id=p,wildtype_sequence=allowed[p]['sequence'],budget=.2) for p in ids]
    cache_path=root/'reports/delepref_v8_main/wt_cache.pt'
    if sha(cache_path)!=read(root/'reports/delepref_v8_main/encoding_completion.json')['cache_sha256']:raise ValueError('cache changed')
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    for k in ['embeddings','sequences']:cache[k]={p:v for p,v in cache[k].items() if p in ids}
    torch.set_num_threads(4);runtime,api=rt.load(root,device='cpu',cached_heads_only=True);chunk=parent.cached_chunk(api,cache,records)
    profiles=read(root/'retrieval/recomb_benign_general_20261017/profiles.json');old=defaultdict(list)
    for row in read(root/'reports/recomb_benign_plans_20261017/candidate_scores.json'):
        if row['protein_id'] in ids and row['method']=='delepref_unlimited' and row['budget']==.2 and row['generation_seed']==13:old[row['protein_id']].append(row)
    candidates=[];selections=[];diagnostics=[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id'];seq=record['wildtype_sequence']
            for beta in frozen['budgets']:
                pool,selected,diagnostic=parent.score_condition(runtime,api,chunk,dict(record,budget=beta),profiles[pid],13)
                candidates.extend(pool);selections.extend(selected);diagnostics.append(diagnostic)
            original=sorted(old[pid],key=lambda r:r['raw_candidate_rank']);template=original[0]
            native=[dict(spans=tuple(map(tuple,json.loads(r['deleted_intervals']))),kind=r['generation_kind'],row=r) for r in original]
            direct=[dict(spans=s['spans'],source_target=s['target'],seed_spans=s['spans'],kind='retrieval_direct') for s in planning.seed_sets(profiles[pid],len(seq),.2) if s['complete_natural_plan']][:4]
            mixed,fallback=planning.mix_candidates(native,direct)
            if fallback:pool=[dict(r) for r in original]
            else:
                embedding=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
                logits=runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
                pool=scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,mixed,len(seq),logits))
            candidates.extend(dict(r,method='rag_direct_only') for r in pool)
            selections.append(controls.select_rows(api,seq,pool,'rag_direct_only'))
    if len(selections)!=91 or len(diagnostics)!=39:raise ValueError('incomplete delta')
    out=root/BASE;write(out/'all_selections.json',selections);write(out/'candidate_scores.json',candidates);write(out/'generation_diagnostics.json',diagnostics)
    report=dict(status='REVIEWED_HEAD_DELTA_COMPLETE',additional_WT=13,budget_conditions=39,selected_outputs=91,
        previous_109_conditions_not_repeated=True,primary_original_K8_pools_reused=True,
        new_encoder_forwards=0,new_training=False,new_structure_predictions=0,freeze_sha256=checksum,
        job_id=os.environ['SLURM_JOB_ID'],outputs={n:sha(out/n) for n in ['all_selections.json','candidate_scores.json','generation_diagnostics.json']})
    write(out/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args();prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256)
