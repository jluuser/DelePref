"""Use unchanged trained heads and real cached embeddings on reviewed inputs only."""
import argparse
import json
import os
from pathlib import Path
import time
from recomb_scope_screen_20261017 import read, sha, write

BASE = Path('reports/recomb_benign_plans_20261017')
FREEZE = Path('configs/recomb_benign_plans_freeze_20261017.json')


def prepare(root):
    out = root / BASE
    out.mkdir(exist_ok=False)
    paths = ['reports/recomb_benign_scope_20261017/general_panel.json',
             'reports/recomb_benign_scope_20261017/retained_subsets.json',
             'retrieval/recomb_benign_general_20261017/profiles.json',
             'configs/recomb_benign_general_profiles_freeze_20261017.json',
             'reports/delepref_v8_main/encoding_completion.json',
             'reports/delepref_v8_evaluation_preparation/main_encoder_inputs.json',
             'analysis/recomb_benign_plans_20261017.py',
             'configs/recomb_benign_baselines_freeze_20261017.json',
             'slurm/recomb_benign_plans_20261017.sbatch']
    accepted = read(root/'configs/recomb_benign_baselines_freeze_20261017.json')
    if sha(root/paths[0]) != accepted['files'][paths[0]]:
        raise ValueError('positive scope differs from reviewed baseline roster')
    previous = root / 'configs/delepref_v8_main_freeze.json'
    profile_freeze = read(root / paths[3])
    if sha(root / paths[2]) != profile_freeze['profiles_sha256']:
        raise ValueError('screened profiles changed')
    write(root / FREEZE, dict(status='REVIEWED_HEADS_ONLY_INPUTS_FROZEN',
        files={p: sha(root/p) for p in paths}, parent_main_freeze_sha256=sha(previous),
        WT=44, primary_budget=.2, primary_seed=13, stability_seed=29,
        stress_budgets=[.1,.3,.4], only_preexisting_screened_nested_subset=True,
        new_encoder_forwards=0, new_training=False, structure_prediction_allowed=False))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE))))


def plans(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only Slurm required')
    if sha(root/FREEZE) != checksum:
        raise ValueError('freeze changed')
    frozen = read(root/FREEZE)
    for path, digest in frozen['files'].items():
        if sha(root/path) != digest:
            raise ValueError('source/input changed: '+path)
    import torch
    import delepref_v8_main as parent
    import delepref_v8_runtime as runtime
    parent.verify(root, frozen['parent_main_freeze_sha256'])
    roster = read(root/'reports/recomb_benign_scope_20261017/general_panel.json')
    allowed = {p['protein_id']:p['sequence'] for p in roster}
    records = [r for r in read(root/'reports/delepref_v8_evaluation_preparation/main_encoder_inputs.json') if r['protein_id'] in allowed]
    if len(records)!=44 or any(r['wildtype_sequence']!=allowed[r['protein_id']] for r in records):
        raise ValueError('reviewed query identity mismatch')
    profiles = read(root/'retrieval/recomb_benign_general_20261017/profiles.json')
    if set(profiles)!=set(allowed):
        raise ValueError('profile query scope mismatch')
    subsets = read(root/'reports/recomb_benign_scope_20261017/retained_subsets.json')
    cache_path = root/'reports/delepref_v8_main/wt_cache.pt'
    provenance = read(root/'reports/delepref_v8_main/encoding_completion.json')
    if sha(cache_path)!=provenance['cache_sha256']:
        raise ValueError('real encoder cache identity mismatch')
    cache = torch.load(cache_path,map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=runtime.CHECKPOINT_SHA:
        raise ValueError('wrong encoder provenance')
    for key in ['embeddings','sequences']:
        cache[key]={pid:v for pid,v in cache[key].items() if pid in allowed}
    out=root/BASE
    claim=out/'execution_claim.json'
    if claim.exists():raise FileExistsError('already claimed; inspect job')
    write(claim,dict(job_id=os.environ['SLURM_JOB_ID']))
    torch.set_num_threads(4)
    start=time.monotonic()
    model,api=runtime.load(root,device='cpu',cached_heads_only=True)
    chunk=parent.cached_chunk(api,cache,records)
    scores,selections,diagnostics=[],[],[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id']
            conditions=[(.2,13),(.2,29)]
            if pid in subsets['completion_ablation']:
                conditions.extend((b,13) for b in [.1,.3,.4])
            for budget,seed in conditions:
                s,w,d=parent.score_condition(model,api,chunk,dict(record,budget=budget),profiles[pid],seed,
                    ablation=(budget==.2 and seed==13 and pid in subsets['completion_ablation']))
                for row in w:
                    row['evaluation_role']='CPU_stability_only' if seed==29 else 'screened_generation'
                scores.extend(s);selections.extend(w);diagnostics.append(d)
    if len(diagnostics)!=109:
        raise ValueError('incomplete reviewed conditions')
    write(out/'candidate_scores.json',scores)
    write(out/'all_selections.json',selections)
    write(out/'generation_diagnostics.json',diagnostics)
    write(out/'planning_completion.json',dict(status='SCREENED_ORIGINAL_RAG_HEADS_COMPLETE_NOT_STRUCTURE_EVALUATED',
        WT=44,conditions=len(diagnostics),selected_outputs=len(selections),scored_candidates=len(scores),
        seconds=time.monotonic()-start,freeze_sha256=checksum,job_id=os.environ['SLURM_JOB_ID'],
        original_checkpoint_sha256=runtime.CHECKPOINT_SHA,cache_sha256=provenance['cache_sha256'],
        extra_PLM_forwards=0,new_training=False,new_structure_predictions=0,
        exact_fallback_checked=True,structure_prediction_allowed=False,
        outputs={p.name:sha(p) for p in out.glob('*.json') if p.name!='planning_completion.json'}))
    print(json.dumps(read(out/'planning_completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--freeze-sha256')
    args=p.parse_args()
    (prepare(args.root) if args.prepare else plans(args.root,args.freeze_sha256))
