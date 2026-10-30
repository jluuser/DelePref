"""Score the already frozen eight SCISOR-L draws with the unchanged original Scorer."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_SCISOR_L_K8_score')
FZ=Path('configs/recomb_confirm_v2_General100_SCISOR_L_K8_score_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('SCISOR L K8 scoring already frozen')
    names=['analysis/recomb_confirm_v2_scisor_K8_score.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_runtime.py','analysis/delepref_v8_main.py','analysis/delepref_v8_rag_unlimited.py',
        'analysis/delepref_unlimited_scorer.py','analysis/delepref_unlimited_planning.py',
        'configs/delepref_v8_baseline_freeze.json','configs/recomb_confirm_v2_General100_scisor_L_freeze.json',
        'reports/recomb_confirm_v2_General100_scisor_L/tasks.json',
        'reports/recomb_confirm_v2_General100_plans/candidate_scores.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'slurm/recomb_confirm_v2_scisor_K8_score.sbatch']
    write(root/FZ,dict(status='EXISTING_SCISOR_L_K8_DRAWS_ORIGINAL_SCORER_FROZEN',
        files={p:sha(root/p) for p in names},WT=20,draws_per_WT=8,
        duplicates='keep all eight predeclared draws, report independent masks, never resample to force unique K8',
        new_SCISOR_calls=0,new_encoder_forwards=0,new_training=False,structure_prediction_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen K8 Scorer input/source changed')
    source=root/'reports/recomb_confirm_v2_General100_scisor_L';finished=read(source/'completion.json')
    if sha(source/'all_conditions.json')!=finished['outputs_sha256']:raise ValueError('SCISOR-L completed pool changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('K8 scoring claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],source_completion_sha256=sha(source/'completion.json')))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as main
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    cachepath=root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt';encoded=read(cachepath.parent/'encoding_completion.json')
    if sha(cachepath)!=encoded['cache_sha256']:raise ValueError('True WT cache changed')
    cache=torch.load(cachepath,map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:raise ValueError('Not original real WT cache')
    baseline=read(root/'configs/delepref_v8_baseline_freeze.json')
    for p,h in {**baseline['inputs'],**baseline['code']}.items():
        if sha(root/p)!=h:raise ValueError('Original immutable model dependency changed')
    torch.set_num_threads(4);start=time.monotonic();runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    records=read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json');chunk=main.cached_chunk(api,cache,records)
    templates={}
    for p in read(root/'reports/recomb_confirm_v2_General100_plans/candidate_scores.json'):
        if p['budget']==.2 and p['generation_seed']==13 and p['method']=='delepref_unlimited':templates.setdefault(p['protein_id'],p)
    pools=[];selected=[];counts=[]
    with torch.inference_mode():
        for condition in read(source/'all_conditions.json'):
            if condition['mode']!='same_scorer_K8':continue
            pid=condition['protein_id'];seq=condition['wildtype_sequence'];candidates=condition['candidates']
            if len(candidates)!=8:raise ValueError('SCISOR-L predeclared draw count differs')
            emb=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
            logits=runtime.model.propose_from_embeddings([emb],[.2])[0].residue_logits.detach().cpu().double().numpy()
            items=[]
            for c in candidates:
                spans=tuple(map(tuple,c['deleted_intervals']));mask=planning.validate_plan(len(seq),.2,spans)
                edited=''.join(aa for aa,d in zip(seq,mask) if not d)
                if edited!=c['edited_sequence']:raise ValueError('SCISOR native explicit mapping reconstruction differs')
                items.append(dict(spans=spans,kind='official_SCISOR_L_native_draw'))
            rows=main.make_rows(api,templates[pid],items,len(seq),logits)
            rows=[dict(r,method='SCISOR_U90_L_K8_same_scorer',candidate_index=i) for i,r in enumerate(rows)]
            scored=scoring.score_candidates(runtime,api,chunk,rows)
            if sorted(p['scorer_rank'] for p in scored)!=list(range(1,9)):raise ValueError('Common Scorer ranks incomplete')
            best=dict(min(scored,key=lambda p:p['scorer_rank']));mask=planning.validate_plan(len(seq),.2,json.loads(best['deleted_intervals']))
            edited=''.join(aa for aa,d in zip(seq,mask) if not d)
            import hashlib
            best.update(edited_sequence=edited,edited_sequence_sha256=hashlib.sha256(edited.encode()).hexdigest(),
                actual_unique_masks=len({p['deletion_mask_sha256'] for p in scored}),predeclared_draws=8)
            pools.extend(scored);selected.append(best);counts.append(dict(protein_id=pid,unique_masks=best['actual_unique_masks']))
    if len(selected)!=20 or len(pools)!=160:raise ValueError('SCISOR-L common Scorer matrix incomplete')
    write(out/'candidate_scores.json',pools);write(out/'all_selections.json',selected);write(out/'mask_diversity.json',counts)
    done=dict(status='OFFICIAL_L_FIXED_K8_DRAW_POOLS_COMMON_ORIGINAL_SCORER_COMPLETE',WT=20,candidates=160,
        selected_outputs=20,unique_masks_per_pool=counts,new_model_encodings=0,new_SCISOR_calls=0,
        new_structure_predictions=0,seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],
        outputs={p:sha(out/p) for p in ['candidate_scores.json','all_selections.json','mask_diversity.json']})
    write(out/'completion.json',done);print({k:v for k,v in done.items() if k!='unique_masks_per_pool'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
