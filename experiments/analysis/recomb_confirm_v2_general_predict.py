"""General100 frozen predictors; explicit primary/extended phases and MSA-aware Boltz inputs."""
import argparse
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

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
sys.path.insert(0,str(ROOT/'code'))
from prediction_journal import PredictionJournal

BASE=Path('reports/recomb_confirm_v2_General100_structures')
CAT=Path('reports/recomb_confirm_v2_General100_catalog')
FREEZE=Path('configs/recomb_confirm_v2_General100_confirmation_freeze.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def read_predicted_chains(path,chains):
    import gemmi
    structure=gemmi.read_structure(str(path));result={}
    for spec in chains:
        chain=structure[0].find_chain(spec['id']);seq=spec['sequence']
        if chain is None:raise ValueError('missing canonical prediction chain')
        ca=[None]*len(seq);selected={};seen=set()
        for residue in chain:
            i=residue.seqid.num-1
            if not 0<=i<len(seq):raise ValueError('prediction position outside input chain')
            aa=gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
            if aa!=seq[i] and seq[i]!='X':raise ValueError('prediction residue/input mismatch')
            if i in seen:raise ValueError('multiple prediction residues at one position')
            seen.add(i)
            for atom in residue:
                if atom.element.name in ('H','D'):continue
                xyz=[atom.pos.x,atom.pos.y,atom.pos.z]
                if not np.isfinite(xyz).all():raise ValueError('invalid predicted atom')
                key=i,atom.name;rank=(-atom.occ,atom.altloc)
                if key not in selected or rank<selected[key][0]:
                    selected[key]=rank,dict(residue_index=i,name=atom.name,element=atom.element.name,xyz=xyz)
        atoms=[a for _,a in (selected[k] for k in sorted(selected))]
        for atom in atoms:
            if atom['name']=='CA':ca[atom['residue_index']]=atom['xyz']
        if not any(c is not None for c in ca):raise ValueError('prediction chain has no CA')
        result[spec['id']]=dict(sequence=seq,coordinates=ca,heavy_atoms=atoms)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictor',choices=['esmfold','boltz2'],required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--phase',choices=['main','extended','all'],default='all')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU only via Slurm')
    if sha(ROOT/FREEZE)!=a.freeze_sha256:raise ValueError('confirmation freeze changed')
    frozen=json.loads((ROOT/FREEZE).read_text())
    if not frozen['fold_allowed'] or not frozen['all_selected_outputs_frozen']:raise ValueError('final release required')
    for f in frozen['files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('frozen experiment file changed: '+f['path'])
    preflight=json.loads((ROOT/'reports/delepref_v8_predictors'/(a.predictor+'_preflight.json')).read_text())
    for f in preflight['files']+preflight['implementation']+preflight.get('standard_amino_acid_assets',[]):
        if sha(f['path'])!=f['sha256']:raise ValueError('predictor identity changed')
    profiles=json.loads((ROOT/CAT/'predictor_profiles.json').read_text())
    cache=json.loads((ROOT/CAT/'compatible_cache.json').read_text())
    requests=[r for r in json.loads((ROOT/CAT/'physical_requests.json').read_text()) if r['predictor']==a.predictor and r['task_id'] not in cache]
    logical=json.loads((ROOT/CAT/'logical_requests.json').read_text())
    main_ids={r['physical_id'] for r in logical if r['predictor']=='esmfold' and r['evaluation_role'] in ('primary20pct','WT_reference')}
    if a.predictor=='esmfold' and a.phase=='main':requests=[r for r in requests if r['task_id'] in main_ids]
    elif a.predictor=='esmfold' and a.phase=='extended':requests=[r for r in requests if r['task_id'] not in main_ids]
    elif a.predictor=='boltz2' and a.phase!='all':raise ValueError('Boltz phase must retain all fixed second-predictor seeds')
    if not 0<=a.shard_index<a.shards:raise ValueError('invalid shard')
    requests=[r for i,r in enumerate(requests) if i%a.shards==a.shard_index]
    journal=PredictionJournal(ROOT/CAT/'statistical_protocol.json',ROOT/BASE/'prediction_journal.jsonl')
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('frozen profiles require A100')
    out=ROOT/BASE;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if a.predictor=='esmfold':
        from transformers import AutoTokenizer,EsmForProteinFolding
        directory=str(Path(next(f['path'] for f in preflight['files'] if f['path'].endswith('pytorch_model.bin'))).parent)
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        model=EsmForProteinFolding.from_pretrained(directory,local_files_only=True).eval()
        model.esm=model.esm.half();model.trunk.set_chunk_size(profiles['esmfold']['chunk_size']);model=model.cuda()
    else:
        from boltz.main import predict
        checkpoint=Path(preflight['files'][0]['path']);model_cache=checkpoint.parent
    print('FROZEN_PREDICTOR_LOADED '+a.predictor,flush=True)
    failures=0
    for n,r in enumerate(requests):
        task=r['task_id'];profile=profiles[r['profile_name']]
        if hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=r['profile_sha256']:raise ValueError('request profile changed')
        reservation=journal.reserve(task_id=task,sequence_sha=r['sequence_sha256'],wt_variant=r['wt_variant'],
            predictor_weights_sha=r['weights_sha256'],profile_sha=r['profile_sha256'],seed=r['seed'],sample=r['sample'],
            category=r['category'],job_id=os.environ['SLURM_JOB_ID'])
        if not reservation['new_reservation']:
            if reservation['state'] not in ('succeeded','cache_hit','failed'):raise RuntimeError('unfinished dispatch needs audit')
            continue
        random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
        torch.backends.cuda.matmul.allow_tf32=profile.get('tf32_allowed',False)
        torch.cuda.reset_peak_memory_stats();journal.start(task);started=time.monotonic()
        print('STRUCTURE_FORWARD_START '+task+' '+str(n+1)+'/'+str(len(requests)),flush=True)
        try:
            confidence=None;raw_record={}
            if a.predictor=='esmfold':
                inputs=tokenizer([r['sequence']],return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode():output=model(**inputs,num_recycles=profile['num_recycles'])
                torch.cuda.synchronize()
                coords=output.positions[-1,0,:,1].detach().float().cpu().numpy()
                coords[~output.atom14_atom_exists[0,:,1].detach().bool().cpu().numpy()]=np.nan
                confidence=float(output.ptm.detach().float().cpu().reshape(-1)[0]);chains=None
                del output,inputs
            else:
                directory=out/'boltz_inputs'/task;directory.mkdir(parents=True,exist_ok=True)
                yaml=directory/'input.yaml'
                import csv
                proteins=[]
                for c in r['chains']:
                    msa_path=ROOT/c['msa_relative_path']
                    if sha(msa_path)!=c['msa_csv_sha256']:raise ValueError('Frozen variant MSA changed')
                    with msa_path.open(newline='') as handle:query=next(csv.DictReader(handle))['sequence']
                    if query!=c['sequence']:raise ValueError('MSA query differs from selected output')
                    proteins.append(dict(protein=dict(id=c['id'],sequence=c['sequence'],msa=str(msa_path))))
                write(yaml,dict(version=1,sequences=proteins))
                predict.callback(data=yaml,out_dir=directory/'output',cache=model_cache,checkpoint=checkpoint,
                    accelerator='gpu',devices=1,model='boltz2',recycling_steps=profile['recycling_steps'],sampling_steps=profile['sampling_steps'],
                    diffusion_samples=1,max_parallel_samples=1,num_workers=0,seed=r['seed'],step_scale=profile['step_scale'],
                    output_format='mmcif',preprocessing_threads=1,use_msa_server=False,no_trifast=True)
                files=list((directory/'output').rglob('*_model_0.cif'))
                if len(files)!=1:raise ValueError('expected one Boltz sample')
                chains=read_predicted_chains(files[0],r['chains'])
                coords=np.asarray(chains['A']['coordinates'],dtype=float)
                raw=out/'artifacts'/(task+'.cif');raw.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(files[0],raw)
                raw_record=dict(raw_structure_path=str(raw),raw_structure_sha256=sha(raw))
                torch.cuda.synchronize()
            if coords.shape!=(len(r['chains'][0]['sequence']),3) or not np.isfinite(coords).all(1).any():raise ValueError('invalid output CA array')
            artifact=out/'artifacts'/(task+'.npz');artifact.parent.mkdir(parents=True,exist_ok=True)
            fields=dict(coordinates=coords,sequence=np.array(r['chains'][0]['sequence']),sequence_sha256=np.array(r['sequence_sha256']))
            if chains is not None:fields['chains_json']=np.array(json.dumps(chains,allow_nan=False,separators=(',',':')))
            np.savez_compressed(artifact,**fields)
            seconds=time.monotonic()-started
            journal.finish(task,succeeded=True,seconds=seconds,max_memory=float(torch.cuda.max_memory_allocated()),output_path=artifact)
            write(out/'records'/(task+'.json'),dict(r,status='SUCCEEDED',output_path=str(artifact),output_sha256=sha(artifact),
                seconds=seconds,confirmation_freeze_sha256=a.freeze_sha256,job_id=os.environ['SLURM_JOB_ID'],confidence=confidence,**raw_record))
            failures=0;print('STRUCTURE_FORWARD_COMPLETE '+task,flush=True)
        except Exception as error:
            journal.finish(task,succeeded=False,seconds=time.monotonic()-started,max_memory=float(torch.cuda.max_memory_allocated()))
            write(out/'records'/(task+'.json'),dict(r,status='FAILED',error=repr(error),confirmation_freeze_sha256=a.freeze_sha256,job_id=os.environ['SLURM_JOB_ID']))
            failures+=1;print('STRUCTURE_FORWARD_FAILED '+task+' '+repr(error),flush=True)
            if failures>=2:raise RuntimeError('two consecutive failures; stop for audit') from error
        finally:
            gc.collect();torch.cuda.empty_cache()
    write(out/(a.predictor+'_'+a.phase+'_shard_'+str(a.shard_index)+'_completion.json'),dict(status='FROZEN_DISPATCH_FINISHED',
        predictor=a.predictor,physical_requests=len(requests),shards=a.shards,shard_index=a.shard_index,freeze_sha256=a.freeze_sha256))


if __name__=='__main__':main()
