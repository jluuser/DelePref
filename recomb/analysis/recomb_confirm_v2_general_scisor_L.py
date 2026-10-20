"""Official L native deletion sampling and fixed K8 control on ordinary General100."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_scisor_L')
FZ=Path('configs/recomb_confirm_v2_General100_scisor_L_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('General100 L generation already frozen')
    preflight=root/'reports/recomb_confirm_v2_scisor_L/preflight_v4.json'
    report=read(preflight)
    if report['status']!='OFFICIAL_SCISOR_L_OFFLINE_ALL_STATE_TOKENIZER_PASS' or not report['all_state_tensor_values_equal']:
        raise ValueError('L full-state preflight not passed')
    panel=read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    subsets=read(root/'reports/recomb_confirm_v2_General100_subsets/subsets.json')
    stress={r['protein_id'] for r in subsets['budget40']};control={r['protein_id'] for r in subsets['cost_and_SCISOR_K8_20']}
    scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    tasks=[]
    for r in panel:
        if scope[r['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':raise ValueError('Input not approved')
        if hashlib.sha256(r['sequence'].encode()).hexdigest()!=r['WT_sha256']:raise ValueError('WT differs')
        for b in ([.2,.1,.3,.4] if r['protein_id'] in stress else [.2]):
            tasks.append(dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],WT_sha256=r['WT_sha256'],
                budget=b,mode='native',candidate_count=1))
    tasks += [dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],WT_sha256=r['WT_sha256'],
        budget=.2,mode='same_scorer_K8',candidate_count=8) for r in panel if r['protein_id'] in control]
    if len(tasks)!=240:raise ValueError('Incorrect native/control matrix')
    write(root/BASE/'tasks.json',tasks)
    names=['analysis/recomb_confirm_v2_general_scisor_L.py','analysis/recomb_confirm_v2_scisor_L_loader_v2.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/delepref_v8_scisor_mapping.py',
        'analysis/delepref_v8_scisor_compat.py','analysis/delepref_v8_scisor_inference_import_v2.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_v8_controls.py',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json',
        'reports/recomb_confirm_v2_scisor_L/preflight_v4.json','reports/delepref_v8_scisor_inventory.json',
        'configs/recomb_confirm_v2_scisor_L_preflight_freeze_v4.json',str(BASE/'tasks.json'),
        'slurm/recomb_confirm_v2_general_scisor_L.sbatch']
    assets=read(root/'reports/delepref_v8_scisor_inventory.json')['assets']
    write(root/FZ,dict(status='QUALIFIED_GENERAL100_L_NATIVE_AND_K8_GENERATION_FROZEN',
        files={p:sha(root/p) for p in names},L_checkpoint_sha256=report['checkpoint_sha256'],
        p0=assets['p0'],trained_step_or_parameters_changed=False,temperature=1.,alpha_rate=1/1.1,
        seed=13,qmax=None,common_exact_budget='round(beta*WT_length), explicit deletion count',
        native_conditions=220,control_K8_conditions=20,actual_new_sampler_calls=360,
        K8_candidate0='same L native output reused',new_training=False,structure_prediction_allowed=False,
        not_equal_compute_to_K8_DelePref=True,wait_for_Raygun_array_before_GPU=True))
    print(sha(root/FZ),flush=True)


def run(root):
    import delepref_unlimited_planning as planning
    from delepref_v8_controls import condition_seed,sequence_sha
    from delepref_v8_scisor_mapping import mapped_plan
    from recomb_general100_common_20261017 import require_gpu
    from recomb_confirm_v2_scisor_L_loader_v2 import load
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen L generation source/input differs')
    inventory=read(root/'reports/delepref_v8_scisor_inventory.json')
    for p,h in inventory['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Original native SCISOR source changed')
    if sha(f['p0']['path'])!=f['p0']['sha256']:raise ValueError('SCISOR prior changed')
    require_gpu();out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('L generation already claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    model,payload,provenance=load(root);del payload
    model=model.to('cuda').eval();model.p0=torch.load(f['p0']['path'],map_location='cuda',weights_only=True)
    rate=f['alpha_rate'];model.alpha=lambda t:(1-t)**rate;model.beta=lambda t:rate/(1-t)
    model.log_alpha=lambda t:-torch.log(model.alpha(t));torch.set_num_threads(4)
    print('CONFIRM_GENERAL100_OFFICIAL_L_MODEL_LOADED '+torch.cuda.get_device_name(0),flush=True)
    rows=[];native={};calls=0;start=time.monotonic()
    with torch.inference_mode():
        for index,task in enumerate(read(out/'tasks.json')):
            seq=task['wildtype_sequence'];pid=task['protein_id'];b=task['budget'];pool=[]
            for rank in range(task['candidate_count']):
                if task['mode']=='same_scorer_K8' and rank==0:
                    pool.append(dict(native[(pid,b)],candidate_index=0,reused_native=True));continue
                seed=condition_seed(13,pid,seq,b)+rank*1_000_003
                random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                ids=model.tokenizer(seq).input_ids
                if len(ids)!=len(seq)+2:raise ValueError('L tokenizer changed length')
                x=torch.tensor(ids,device='cuda').unsqueeze(0);d=planning.delete_count(len(seq),b);before=time.monotonic()
                print('CONFIRM_GENERAL100_L_FORWARD_START '+str(index)+' '+str(rank),flush=True)
                y,preserved=model.shrink_sequence(x,torch.tensor([d],device='cuda'),temperature=1.)
                item=mapped_plan(seq,b,preserved[0],y[0].detach().cpu().tolist(),model.tokenizer)
                mask=planning.validate_plan(len(seq),b,item['deleted_intervals'])
                if len(item['edited_sequence'])!=len(seq)-d:raise ValueError('L exact budget not met')
                item.update(candidate_index=rank,condition_seed=seed,generation_seed=13,
                    edited_sequence_sha256=sequence_sha(item['edited_sequence']),
                    deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                    seconds=time.monotonic()-before,reused_native=False)
                calls+=1;pool.append(item)
            row=dict(task,candidates=pool,method='SCISOR_U90_L_native' if task['mode']=='native' else 'SCISOR_U90_L_K8_same_scorer',
                actual_GPU=torch.cuda.get_device_name(0),job_id=os.environ['SLURM_JOB_ID'])
            if task['mode']=='native':row.update(pool[0]);native[(pid,b)]=dict(pool[0])
            rows.append(row);write(out/('condition_%03d.json'%index),row)
            print('CONFIRM_GENERAL100_L_CONDITION_DONE '+str(index),flush=True)
    if len(rows)!=240 or calls!=360:raise ValueError('L native/control matrix incomplete')
    write(out/'all_conditions.json',rows)
    report=dict(status='GENERAL100_OFFICIAL_L_NATIVE_AND_FIXED_K8_CONTROL_COMPLETE',
        WT=100,native_conditions=220,K8_control_conditions=20,new_sampler_calls=calls,
        exact_budgets_and_explicit_correspondence_pass=True,new_structure_predictions=0,new_training=False,
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],checkpoint_sha256=f['L_checkpoint_sha256'],
        outputs_sha256=sha(out/'all_conditions.json'),structure_prediction_allowed=False)
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
