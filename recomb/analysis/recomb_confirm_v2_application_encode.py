"""Single original WT encoding for each explicitly released application construct."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import torch
import delepref_v8_runtime as rt
from recomb_confirm_v2_diagnostics import read,sha,write

COHORTS={'Binder60':dict(N=60,base='reports/recomb_confirm_v2_Binder60_constructs',panel='Binder60_panel.json'),
         'Enzyme80':dict(N=80,base='reports/recomb_confirm_v2_Enzyme80_constructs',panel='Enzyme80_panel.json')}


def paths(cohort):return Path('reports/recomb_confirm_v2_'+cohort+'_model'),Path('configs/recomb_confirm_v2_'+cohort+'_encoder_freeze.json')


def prepare(root,cohort):
    spec=COHORTS[cohort];out,FZ=paths(cohort)
    if (root/FZ).exists():raise FileExistsError('Application encoding already released')
    base=Path(spec['base']);panel=read(root/base/spec['panel']);done=read(root/base/'completion.json')
    if len(panel)!=spec['N'] or sha(root/base/spec['panel'])!=done['outputs'][spec['panel']]:raise ValueError('Released cohort panel differs')
    for row in panel:
        if not row['WT_encoding_allowed'] or row['structure_prediction_allowed']:raise ValueError('Wrong operation release')
        if hashlib.sha256(row['sequence'].encode()).hexdigest()!=row['WT_sha256']:raise ValueError('Released WT SHA differs')
    names=['analysis/recomb_confirm_v2_application_encode.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_runtime.py','configs/delepref_original_restored.yaml',
        'configs/delepref_v8_baseline_freeze.json','provenance/original_delepref_assets_20261016.json',
        str(base/spec['panel']),str(base/'completion.json'),str(base/'encoder_inputs.json'),
        'slurm/recomb_confirm_v2_application_encode.sbatch']
    write(root/FZ,dict(status='QUALIFIED_APPLICATION_REAL_WT_ENCODING_ONLY_FROZEN',cohort=cohort,WT=spec['N'],
        files={n:sha(root/n) for n in names},original_checkpoint_sha256=rt.CHECKPOINT_SHA,
        allowed_GPU=['A40','A100'],new_training=False,mutant_encoding_allowed=False,
        model_operation='encode only released design WT, no variants or folding',
        deletion_generation_allowed=False,structure_prediction_allowed=False))
    (root/out).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root,cohort):
    spec=COHORTS[cohort];out,FZ=paths(cohort);f=read(root/FZ)
    for n,h in f['files'].items():
        if sha(root/n)!=h:raise ValueError('Frozen application input/source differs')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    device=torch.cuda.get_device_name(0)
    if not any(x in device for x in f['allowed_GPU']):raise ValueError('Unapproved GPU kind')
    if (root/out/'encoding_completion.json').exists() or (root/out/'execution_claim.json').exists():raise FileExistsError('Application encoder claimed; inspect existing job')
    write(root/out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    old=read(root/'configs/delepref_v8_baseline_freeze.json')
    for p,h in {**old['inputs'],**old['code']}.items():
        if sha(root/p)!=h:raise ValueError('Original immutable model dependency changed')
    records=read(root/spec['base']/'encoder_inputs.json')
    torch.set_num_threads(4);start=time.monotonic();runtime,api=rt.load(root)
    print('APPLICATION_ORIGINAL_MODEL_LOADED '+cohort+' '+device,flush=True)
    print('APPLICATION_REAL_WT_ENCODING_START '+cohort,flush=True)
    with torch.inference_mode():chunk=api.encode_chunk(runtime,records)
    if chunk.encoded.unique_count!=spec['N']:raise ValueError('Application WT count differs')
    cache=dict(embeddings={k.protein_id:v.detach().cpu() for k,v in chunk.encoded.embeddings.items()},
        sequences={k.protein_id:v for k,v in chunk.encoded.sequences.items()},
        checkpoint_sha256=rt.CHECKPOINT_SHA,real_encoder=True,unique_WT=spec['N'],
        encoder_profile=dict(dtype='bfloat16',hidden_size=1152,max_length=1022,
            snapshot=runtime.config['model']['esmc_snapshot'],actual_GPU=device))
    torch.save(cache,root/out/'wt_cache.pt')
    done=dict(status='APPLICATION_REAL_ORIGINAL_WT_ENCODING_COMPLETE',cohort=cohort,WT=spec['N'],
        cache_sha256=sha(root/out/'wt_cache.pt'),seconds=time.monotonic()-start,actual_GPU=device,
        job_id=os.environ['SLURM_JOB_ID'],checkpoint_sha256=rt.CHECKPOINT_SHA,freeze_sha256=sha(root/FZ),
        mutant_encodings=0,new_training=False,new_structure_predictions=0)
    write(root/out/'encoding_completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cohort',choices=COHORTS,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root,a.cohort) if a.prepare else run(a.root,a.cohort)
