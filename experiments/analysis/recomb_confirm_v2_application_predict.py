"""Frozen full-complex application inference; no implicit MSA or chemistry substitution."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_general_predict import read_predicted_chains
from recomb_confirm_v2_application_catalog_v2 import digest

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE=Path('reports/recomb_confirm_v2_application_structures')
CAT=Path('reports/recomb_confirm_v2_application_catalog_v2')
FREEZE=Path('configs/recomb_confirm_v2_application_confirmation_freeze.json')
sys.path.insert(0,str(ROOT/'code'))
from prediction_journal import PredictionJournal


def verify_chemical_output(path,model_input):
    import gemmi
    structure=gemmi.read_structure(str(path));result=[]
    if len(structure)!=1:raise ValueError('Expected one predicted complex model')
    for item in model_input['sequences']:
        if 'ligand' not in item:continue
        spec=item['ligand'];chain=structure[0].find_chain(spec['id'])
        if chain is None:raise ValueError('Predicted complex omitted a frozen ligand chain')
        residues=list(chain);expected=spec['ccd'] if isinstance(spec['ccd'],list) else [spec['ccd']]
        if [r.name for r in residues]!=expected:raise ValueError('Predicted CCD unit order/composition differs')
        heavy=0
        for r in residues:
            for atom in r:
                if atom.element.name in ('H','D'):continue
                if not np.isfinite([atom.pos.x,atom.pos.y,atom.pos.z]).all():raise ValueError('Nonfinite predicted chemical atom')
                heavy+=1
        if not heavy:raise ValueError('Predicted ligand has no resolved heavy atom')
        result.append(dict(chain=spec['id'],CCD_units=expected,heavy_atoms=heavy))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--phase',choices=['WT','main','extension','all'],required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Application GPU only through Slurm')
    if sha(ROOT/FREEZE)!=a.freeze_sha256:raise ValueError('Application final confirmation freeze changed')
    freeze=read(ROOT/FREEZE)
    if not freeze.get('fold_allowed') or not freeze.get('all_selected_outputs_frozen'):raise ValueError('Complete final application release required')
    for item in freeze['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen application source/input changed: '+item['path'])
    preflight=read(ROOT/'reports/delepref_v8_predictors/boltz2_preflight.json')
    for item in preflight['files']+preflight['implementation']+preflight.get('standard_amino_acid_assets',[]):
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen predictor implementation/weight changed')
    requests=read(ROOT/CAT/'physical_requests.json');logical=read(ROOT/CAT/'logical_requests.json');profiles=read(ROOT/CAT/'predictor_profiles.json')
    wt={r['physical_id'] for r in logical if r['method']=='WT' and r['seed']==20261017}
    primary={r['physical_id'] for r in logical if r['evaluation_role']=='primary20pct' and r['seed']==20261017}
    if a.phase=='WT':requests=[r for r in requests if r['task_id'] in wt]
    elif a.phase=='main':requests=[r for r in requests if r['task_id'] in primary-wt]
    elif a.phase=='extension':requests=[r for r in requests if r['task_id'] not in wt|primary]
    if not 0<=a.shard_index<a.shards:raise ValueError('Invalid application prediction shard')
    requests=[r for i,r in enumerate(requests) if i%a.shards==a.shard_index]
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('Frozen application profile requires A100')
    from boltz.main import predict
    checkpoint=Path(next(f['path'] for f in preflight['files'] if f['path'].endswith('boltz2_conf.ckpt')))
    out=ROOT/BASE;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    journal=PredictionJournal(ROOT/CAT/'statistical_protocol.json',out/'prediction_journal.jsonl')
    print('APPLICATION_FROZEN_PREDICTOR_LOADED '+torch.cuda.get_device_name(),flush=True)
    failures=0
    for index,r in enumerate(requests):
        profile=profiles[r['profile_name']];input_path=ROOT/r['model_input_relative_path']
        if digest(profile)!=r['profile_sha256'] or sha(input_path)!=r['model_input_sha256']:raise ValueError('Actual input/profile fingerprint differs')
        model_input=read(input_path)
        for c in r['chains']:
            m=ROOT/c['msa_relative_path']
            if sha(m)!=c['msa_csv_sha256']:raise ValueError('Frozen physical-chain MSA changed')
            with m.open(newline='') as handle:query=next(csv.DictReader(handle))['sequence']
            if query!=c['sequence']:raise ValueError('Actual model MSA query differs from physical chain')
        task=r['task_id'];reservation=journal.reserve(task_id=task,sequence_sha=r['sequence_sha256'],wt_variant=r['wt_variant'],
            predictor_weights_sha=r['weights_sha256'],profile_sha=r['profile_sha256'],seed=r['seed'],sample=r['sample'],
            category=r['category'],job_id=os.environ['SLURM_JOB_ID'])
        if not reservation['new_reservation']:
            if reservation['state'] not in ('succeeded','cache_hit','failed'):raise RuntimeError('Unfinished previous dispatch requires audit')
            continue
        random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
        torch.backends.cuda.matmul.allow_tf32=profile.get('tf32_allowed',False)
        torch.cuda.reset_peak_memory_stats();journal.start(task);before=time.monotonic()
        print('APPLICATION_STRUCTURE_FORWARD_START '+task+' '+str(index+1)+'/'+str(len(requests)),flush=True)
        try:
            directory=out/'boltz_runs'/task;directory.mkdir(parents=True,exist_ok=True)
            predict.callback(data=input_path,out_dir=directory,cache=checkpoint.parent,checkpoint=checkpoint,
                accelerator='gpu',devices=1,model='boltz2',recycling_steps=profile['recycling_steps'],sampling_steps=profile['sampling_steps'],
                diffusion_samples=1,max_parallel_samples=1,num_workers=0,seed=r['seed'],step_scale=profile['step_scale'],
                output_format='mmcif',preprocessing_threads=1,use_msa_server=False,no_trifast=True)
            files=list(directory.rglob('*_model_0.cif'))
            if len(files)!=1:raise ValueError('Expected exactly one predicted complex')
            chains=read_predicted_chains(files[0],r['chains']);chemistry=verify_chemical_output(files[0],model_input)
            primary_chain=chains[r['primary_design_chain_id']];coords=np.asarray(primary_chain['coordinates'],float)
            if coords.shape!=(len(r['sequence']),3) or not np.isfinite(coords).all(1).any():raise ValueError('Invalid primary design CA array')
            torch.cuda.synchronize();artifact=out/'artifacts'/(task+'.npz');artifact.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(artifact,coordinates=coords,sequence=np.array(r['sequence']),sequence_sha256=np.array(r['sequence_sha256']),
                chains_json=np.array(json.dumps(chains,allow_nan=False,separators=(',',':'))),
                chemical_output_json=np.array(json.dumps(chemistry,allow_nan=False,separators=(',',':'))))
            raw=out/'artifacts'/(task+'.cif');shutil.copyfile(files[0],raw);seconds=time.monotonic()-before
            journal.finish(task,succeeded=True,seconds=seconds,max_memory=float(torch.cuda.max_memory_allocated()),output_path=artifact)
            write(out/'records'/(task+'.json'),dict(r,status='SUCCEEDED',output_path=str(artifact),output_sha256=sha(artifact),
                raw_structure_path=str(raw),raw_structure_sha256=sha(raw),chemical_output=chemistry,
                seconds=seconds,actual_GPU=torch.cuda.get_device_name(),confirmation_freeze_sha256=a.freeze_sha256,job_id=os.environ['SLURM_JOB_ID']))
            failures=0;print('APPLICATION_STRUCTURE_FORWARD_COMPLETE '+task,flush=True)
        except Exception as error:
            journal.finish(task,succeeded=False,seconds=time.monotonic()-before,max_memory=float(torch.cuda.max_memory_allocated()))
            write(out/'records'/(task+'.json'),dict(r,status='FAILED',error=repr(error),confirmation_freeze_sha256=a.freeze_sha256,job_id=os.environ['SLURM_JOB_ID']))
            failures+=1;print('APPLICATION_STRUCTURE_FORWARD_FAILED '+task+' '+repr(error),flush=True)
            if failures>=2:raise RuntimeError('Two consecutive application failures, stop for technical audit') from error
        finally:gc.collect();torch.cuda.empty_cache()
    write(out/('boltz_'+a.phase+'_shard_'+str(a.shard_index)+'_completion.json'),dict(status='FROZEN_APPLICATION_DISPATCH_FINISHED',
        phase=a.phase,physical_requests=len(requests),shards=a.shards,shard_index=a.shard_index,freeze_sha256=a.freeze_sha256))


if __name__=='__main__':main()
