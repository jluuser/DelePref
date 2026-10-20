"""Original DelePref real-sequence inference and reusable WT cache, never training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import delepref_v8_runtime as rt

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = 'reports/delepref_v8'
FREEZE = 'configs/delepref_v8_baseline_freeze.json'


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def prepare(root):
    registry = json.loads((root/'provenance/original_delepref_assets_20261016.json').read_text())
    raw = subprocess.check_output(['git', '--git-dir='+registry['source_version']['git_dir'],
        'show', '239942a:configs/final_readiness/deletion_preference_student_v3_final.yaml'])
    config = root/'configs/delepref_original_restored.yaml'
    if config.exists():
        if config.read_bytes() != raw:
            raise ValueError('existing restored config differs')
    else:
        config.write_bytes(raw)
    if rt.sha(config) != rt.CONFIG_SHA:
        raise ValueError('restored configuration identity mismatch')
    queries = json.loads((root/'retrieval/rag_v6/prepared/query_map.json').read_text())
    train = sorted([q for q in queries if q['split']=='train'], key=lambda q:q['protein_id'])[:6]
    inputs = [dict(protein_id=q['protein_id'], wildtype_sequence=q['wt_sequence'], budget=b)
        for q in train for b in (.1,.2)]
    out = root/BASE
    out.mkdir(exist_ok=False)
    write(out/'inputs.json', inputs)
    public = Path(registry['public_code'])
    code = {str(p):rt.sha(p) for p in list((public/'src/delepref').glob('*.py'))+
            [public/'scripts/evaluation_runtime.py',public/'scripts/train_student_pilot.py']}
    code.update({p:rt.sha(root/p) for p in ['analysis/delepref_v8_runtime.py',
        'analysis/delepref_v8_baseline.py','slurm/delepref_v8_baseline.sbatch']})
    pins = {str(config):rt.sha(config), str(out/'inputs.json'):rt.sha(out/'inputs.json'),
        str(root/'provenance/original_delepref_assets_20261016.json'):rt.sha(root/'provenance/original_delepref_assets_20261016.json')}
    freeze = dict(status='ORIGINAL_DELEPREF_BASELINE_FROZEN',inputs=pins,code=code,
        checkpoint_sha256=rt.CHECKPOINT_SHA,WT=6,budgets=[.1,.2],seeds=[13,29],K=8,
        original_heads=True,training=False,new_structure_predictions=0,paper_changes=False,
        stage='engineering_development_not_independent_confirmation')
    write(root/FREEZE, freeze)
    return dict(freeze_sha256=rt.sha(root/FREEZE),WT=6,conditions=24)


def execute(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU allocation required')
    if rt.sha(root/FREEZE) != checksum:
        raise ValueError('freeze checksum mismatch')
    freeze = json.loads((root/FREEZE).read_text())
    for p,h in {**freeze['inputs'],**freeze['code']}.items():
        if rt.sha(root/p) != h:
            raise ValueError('frozen input/source changed: '+p)
    out=root/BASE
    write(out/'baseline_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    start=time.time(); torch.set_num_threads(4)
    runtime, api = rt.load(root)
    print(json.dumps(dict(event='ORIGINAL_MODEL_LOADED',step=4000,policy_tensors=18,scorer_tensors=14)),flush=True)
    records=json.loads((out/'inputs.json').read_text())
    with torch.inference_mode():
        chunk=api.encode_chunk(runtime,records)
    print(json.dumps(dict(event='REAL_WT_ENCODING_COMPLETE',unique_WT=chunk.encoded.unique_count,
        batches=runtime.model.encoder.forward_batches)),flush=True)
    cache=dict(embeddings={key.protein_id:value.detach().cpu() for key,value in chunk.encoded.embeddings.items()},
        sequences={key.protein_id:value for key,value in chunk.encoded.sequences.items()},
        checkpoint_sha256=rt.CHECKPOINT_SHA,config_sha256=runtime.checkpoint['config_sha256'],
        encoder_profile=dict(dtype='bfloat16',snapshot=runtime.config['model']['esmc_snapshot'],max_length=1022),
        real_encoder=True,unique_WT=chunk.encoded.unique_count)
    torch.save(cache,out/'original_wt_cache.pt')
    scored=[]
    for seed in freeze['seeds']:
        raw=api.generate_raw_candidates(runtime,chunk,global_generation_seed=seed)
        result=api.score_candidates(runtime,chunk,raw)
        scored.extend(result)
        print(json.dumps(dict(event='ORIGINAL_PLANNING_COMPLETE',seed=seed,candidates=len(result))),flush=True)
    write(out/'baseline_scores.json',scored)
    selected=api.select_scorer_top1(scored)
    write(out/'baseline_selected.json',selected)
    report=dict(status='ORIGINAL_DELEPREF_REAL_INFERENCE_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,WT=6,conditions=len(selected),candidates=len(scored),
        checkpoint_sha256=rt.CHECKPOINT_SHA,cache_sha256=rt.sha(out/'original_wt_cache.pt'),
        scores_sha256=rt.sha(out/'baseline_scores.json'),source_freeze_sha256=checksum,
        WT_encodings=chunk.encoded.unique_count,encoder_batches=runtime.model.encoder.forward_batches,
        mutant_encodings=0,trained_policy_used=True,trained_scorer_used=True,
        new_training=False,new_structure_predictions=0,paper_changes=False)
    write(out/'baseline_completion.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare: print(json.dumps(prepare(a.root)),flush=True)
    else: execute(a.root,a.freeze_sha256)
