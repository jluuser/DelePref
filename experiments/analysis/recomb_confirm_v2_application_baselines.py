"""Reuse verified native Raygun and official SCISOR-L for frozen ordinary applications."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import sys
import time
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_application_baselines')
FZ=Path('configs/recomb_confirm_v2_application_baselines_freeze.json')


def tasks(root):
    result=[]
    for cohort,n,budgets,subset_n in [('Binder60',60,[.1],20),('Enzyme80',80,[.1,.3],30)]:
        base=root/('reports/recomb_confirm_v2_'+cohort+'_constructs')
        panel=read(base/(cohort+'_panel.json'));done=read(base/'completion.json')
        if len(panel)!=n or sha(base/(cohort+'_panel.json'))!=done['outputs'][cohort+'_panel.json']:
            raise ValueError('Frozen application panel identity differs')
        subs=read(root/('reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json'))['budget10pct_and_three_structure_seeds']
        if len(subs)!=subset_n:raise ValueError('Application fixed subset differs')
        extra={r['protein_id'] for r in subs}
        for row in panel:
            seq=row['sequence']
            if not row['WT_encoding_allowed'] or not row['protein_assembly_complete'] or hashlib.sha256(seq.encode()).hexdigest()!=row['WT_sha256']:
                raise ValueError('Application lacks released ordinary WT/protein construct')
            for b in [.2]+(budgets if row['protein_id'] in extra else []):
                d=max(1,min(len(seq)-1,round(len(seq)*b)))
                result.append(dict(protein_id=row['protein_id'],sequence=seq,WT_sha256=row['WT_sha256'],cohort=cohort,
                    dataset=cohort+'_confirm_v2',budget=b,target_length=len(seq)-d,mode='native',candidate_count=500))
    if len(result)!=220:raise ValueError('Incorrect application baseline matrix')
    return result


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application native baselines already frozen')
    out=root/BASE;rows=tasks(root)
    prior=read(root/'configs/recomb_confirm_v2_General100_raygun_freeze.json')
    contract=root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_checkpoint_load_contract.json'
    if sha(contract)!=prior['files']['reports/recomb_confirm_v2_General100_raygun/raygun_checkpoint_load_contract.json']:
        raise ValueError('Completed native Raygun load contract differs')
    write(out/'raygun_tasks.json',rows);write(out/'raygun_checkpoint_load_contract.json',read(contract))
    inv=read(root/'reports/delepref_v8_scisor_inventory.json');pre=read(root/'reports/recomb_confirm_v2_scisor_L/preflight_v4.json')
    if pre['status']!='OFFICIAL_SCISOR_L_OFFLINE_ALL_STATE_TOKENIZER_PASS':raise ValueError('Official L preflight incomplete')
    names=['analysis/recomb_confirm_v2_application_baselines.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_general100_baselines_20261017.py','analysis/recomb_general100_common_20261017.py',
        'analysis/recomb_confirm_v2_scisor_L_loader_v2.py','analysis/delepref_v8_scisor_mapping.py',
        'analysis/delepref_v8_scisor_compat.py','analysis/delepref_v8_scisor_inference_import_v2.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_v8_controls.py',
        'reports/recomb_confirm_v2_scisor_L/preflight_v4.json','reports/delepref_v8_scisor_inventory.json',
        'configs/recomb_confirm_v2_General100_raygun_freeze.json',str(BASE/'raygun_tasks.json'),
        str(BASE/'raygun_checkpoint_load_contract.json'),
        'slurm/recomb_confirm_v2_application_raygun.sbatch','slurm/recomb_confirm_v2_application_scisor_L.sbatch']
    for cohort in ['Binder60','Enzyme80']:
        names += ['reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json',
            'reports/recomb_confirm_v2_'+cohort+'_constructs/completion.json',
            'reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json']
    freeze=dict(status='ORDINARY_APPLICATION_NATIVE_RAYGUN_AND_OFFICIAL_L_FROZEN',
        files={p:sha(root/p) for p in names},assets=prior['assets'],external_code=prior['external_code'],
        protocol=prior['protocol'],SCISOR_code=inv['code_sha256'],SCISOR_p0=inv['assets']['p0'],
        L_checkpoint_sha256=pre['checkpoint_sha256'],conditions=220,
        cohort_conditions=dict(Binder60=80,Enzyme80=140),qmax=None,
        Raygun='unchanged official native500 -> native adjusted PLL top50 -> top1',
        SCISOR='official L native K1, temperature1, alpha_rate1/1.1, common exact rounded deletion count',
        no_functional_site_protection=True,no_new_training=True,structure_prediction_allowed=False,
        GPU_coordination='Raygun array throttle1 plus L native1; both wait completed General L and application WT encoders; do not raise throttle without auditing all task GPUs')
    write(root/FZ,freeze);print(sha(root/FZ),flush=True)


def verify(root):
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen application native input/source changed: '+p)
    if read(root/BASE/'raygun_tasks.json')!=tasks(root):raise ValueError('Application scope/identity task reconstruction differs')
    return f


def raygun(root,shard,shards):
    f=verify(root)
    for p,h in f['external_code'].items():
        if sha(p)!=h:raise ValueError('Verified native Raygun source changed')
    for k in ['raygun_checkpoint','input_encoder']:
        if sha(f['assets'][k]['path'])!=f['assets'][k]['sha256']:raise ValueError('Raygun model asset changed')
    from recomb_general100_common_20261017 import require_gpu
    require_gpu()
    import recomb_general100_baselines_20261017 as runner
    runner.BASE=BASE;runner.run_raygun(root,f,shard,shards)


def scisor(root):
    import numpy as np
    import torch
    from delepref_v8_scisor_mapping import mapped_plan
    from delepref_v8_controls import condition_seed,sequence_sha
    from recomb_general100_common_20261017 import require_gpu
    from recomb_confirm_v2_scisor_L_loader_v2 import load
    import delepref_unlimited_planning as planning
    f=verify(root)
    for p,h in f['SCISOR_code'].items():
        if sha(p)!=h:raise ValueError('Original SCISOR source changed')
    if sha(f['SCISOR_p0']['path'])!=f['SCISOR_p0']['sha256']:raise ValueError('SCISOR prior changed')
    require_gpu();out=root/BASE/'scisor_L';out.mkdir(parents=True,exist_ok=True)
    if (out/'execution_claim.json').exists():raise FileExistsError('Application L already claimed; inspect job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    model,payload,provenance=load(root);del payload
    model=model.to('cuda').eval();model.p0=torch.load(f['SCISOR_p0']['path'],map_location='cuda',weights_only=True)
    rate=1/1.1;model.alpha=lambda t:(1-t)**rate;model.beta=lambda t:rate/(1-t);model.log_alpha=lambda t:-torch.log(model.alpha(t))
    torch.set_num_threads(4);start=time.monotonic();rows=[]
    print('APPLICATION_OFFICIAL_L_MODEL_LOADED '+torch.cuda.get_device_name(),flush=True)
    with torch.inference_mode():
        for i,task in enumerate(read(root/BASE/'raygun_tasks.json')):
            seq=task['sequence'];pid=task['protein_id'];b=task['budget'];seed=condition_seed(13,pid,seq,b)
            random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            ids=model.tokenizer(seq).input_ids
            if len(ids)!=len(seq)+2:raise ValueError('Application L token length differs')
            x=torch.tensor(ids,device='cuda').unsqueeze(0);d=planning.delete_count(len(seq),b);before=time.monotonic()
            print('APPLICATION_L_FORWARD_START '+str(i)+' '+pid+' '+str(b),flush=True)
            y,preserved=model.shrink_sequence(x,torch.tensor([d],device='cuda'),temperature=1.)
            item=mapped_plan(seq,b,preserved[0],y[0].detach().cpu().tolist(),model.tokenizer)
            mask=planning.validate_plan(len(seq),b,item['deleted_intervals'])
            if len(item['edited_sequence'])!=task['target_length']:raise ValueError('Application L exact budget differs')
            item.update(task,method='SCISOR_U90_L_native',candidate_count=1,condition_seed=seed,generation_seed=13,
                edited_sequence_sha256=sequence_sha(item['edited_sequence']),
                deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                seconds=time.monotonic()-before,actual_GPU=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'])
            write(out/('condition_%03d.json'%i),item);rows.append(item)
    write(out/'all_conditions.json',rows)
    done=dict(status='APPLICATION_OFFICIAL_SCISOR_L_NATIVE_COMPLETE',conditions=len(rows),new_sampler_calls=len(rows),
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_structure_predictions=0,
        outputs_sha256=sha(out/'all_conditions.json'),K=1,not_equal_compute_to_K8=True)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--method',choices=['raygun','scisor_L']);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.method=='raygun':raygun(a.root,a.shard_index,a.shards)
    elif a.method=='scisor_L':scisor(a.root)
    else:p.error('--method required for execution')
