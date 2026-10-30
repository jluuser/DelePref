"""Independently verify all native500 pools, top50 ordering and final Raygun choices."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write


def run(root,collection):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    if collection=='General100':
        base=root/'reports/recomb_confirm_v2_General100_raygun_v2'
        freeze=root/'configs/recomb_confirm_v2_General100_raygun_freeze.json'
    else:
        base=root/'reports/recomb_confirm_v2_application_baselines'
        freeze=root/'configs/recomb_confirm_v2_application_baselines_freeze.json'
    out=root/('reports/recomb_confirm_v2_'+collection+'_Raygun_native_audit')
    if (out/'completion.json').exists():raise FileExistsError('Native pool audit already complete')
    f=read(freeze)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen native Raygun inputs changed: '+p)
    tasks=read(base/'raygun_tasks.json');results=[];sources={}
    if len(tasks)!=220:raise ValueError('Unexpected native condition roster')
    for shard in range(4):
        p=base/'raygun'/('completion_%d.json'%shard)
        if read(p)['conditions']!=sum(i%4==shard for i in range(len(tasks))):raise ValueError('Native array incomplete')
        sources[str(p.relative_to(root))]=sha(p)
    for index,task in enumerate(tasks):
        path=base/'raygun'/('condition_%03d.json'%index);row=read(path)
        for key,value in task.items():
            if row[key]!=value:raise ValueError('Native task/result identity differs: '+key)
        if hashlib.sha256(row['sequence'].encode()).hexdigest()!=row['WT_sha256']:raise ValueError('WT SHA differs')
        raw=base/'raygun'/('raw_%03d.jsonl.gz'%index)
        if sha(raw)!=row['raw_sha256']:raise ValueError('Raw native candidates changed')
        with gzip.open(raw,'rt') as h:pool=[json.loads(line) for line in h]
        if len(pool)!=500 or row['raw_candidate_count']!=500 or row['PLL_ranked_count']!=50 or len(row['native_top50'])!=50:
            raise ValueError('Native500/top50 cardinality differs')
        for i,candidate in enumerate(pool):
            if candidate['generation_index']!=i or len(candidate['sequence'])!=task['target_length']:
                raise ValueError('Raw native candidate length/index differs')
            if set(candidate['sequence'])-set('ACDEFGHIKLMNPQRSTVWYX') or not math.isfinite(candidate['adjusted_PLL']):
                raise ValueError('Invalid amino-acid candidate or PLL')
        order=sorted(range(500),key=lambda i:(-pool[i]['adjusted_PLL'],i))[:50]
        for rank,(i,saved) in enumerate(zip(order,row['native_top50']),1):
            candidate=pool[i]
            if saved['generation_index']!=i or saved['sequence']!=candidate['sequence'] or saved['adjusted_PLL']!=candidate['adjusted_PLL'] or saved['pll_rank']!=rank:
                raise ValueError('Stored native top50 is not the original PLL ordering')
            if saved['sequence_sha256']!=hashlib.sha256(saved['sequence'].encode()).hexdigest():raise ValueError('Top50 candidate SHA differs')
        if row['edited_sequence']!=pool[order[0]]['sequence'] or row['edited_sequence_sha256']!=row['native_top50'][0]['sequence_sha256']:
            raise ValueError('Stored native winner differs')
        sources[str(path.relative_to(root))]=sha(path);sources[str(raw.relative_to(root))]=sha(raw)
        results.append(dict(protein_id=row['protein_id'],budget=row['budget'],WT_sha256=row['WT_sha256'],
            edited_sequence_sha256=row['edited_sequence_sha256'],source=str(path.relative_to(root)),
            raw_pool_sha256=row['raw_sha256'],unique_raw_sequences=len({c['sequence'] for c in pool}),
            generation_seconds=row['generation_seconds'],PLL_seconds=row['pll_seconds'],actual_GPU=row['actual_GPU']))
    write(out/'per_condition_audit.json',results)
    done=dict(status='ALL_NATIVE_RAYGUN_RAW500_TOP50_AND_FINAL_WINNERS_VERIFIED',collection=collection,
        conditions=220,raw_candidates_audited=110000,top50_records_audited=11000,
        source_sha256=sources,per_condition_sha256=sha(out/'per_condition_audit.json'),
        science_freeze_sha256=sha(freeze),audit_source_sha256=sha(Path(__file__)),
        new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print({k:v for k,v in done.items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--collection',choices=['General100','application'],required=True)
    a=p.parse_args();run(a.root,a.collection)
