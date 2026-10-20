"""WT-only calibration on positively reviewed constructs; no variant selection."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_audit_msa import records

REMOTE=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE=Path('reports/recomb_confirm_v2_WT_calibration')
MSA=Path('external/recomb_confirm_v2_WT_MSA_reviewed')
FREEZE=Path('configs/recomb_confirm_v2_WT_calibration_freeze.json')
SEEDS=(20261017,20261018,20261019)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root):
    out=root/BASE
    if (out/'prepare_completion.json').exists():raise FileExistsError('WT inputs already prepared')
    panel=read(root/'reports/recomb_benign_application_constructs_20261017/panel.json')
    allowed={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if len(panel)!=23 or {p['protein_id'] for p in panel}!=allowed:raise ValueError('Wrong reviewed WT roster')
    review=read(root/MSA/'completion.json');queries={r['query_sha256']:r for r in review['files']}
    query_manifest={r['sequence_sha256']:r for r in read(root/'external/recomb_confirm_v2_WT_MSA_staging/request_manifest.json')['queries']}
    csvs=[]
    for h,r in queries.items():
        source=root/MSA/'a3m'/(h+'.a3m')
        if sha(source)!=r['sha256']:raise ValueError('Reviewed MSA changed')
        pairs=records(source);sequence=query_manifest[h]['sequence']
        if pairs[0][1]!=sequence or hashlib.sha256(sequence.encode()).hexdigest()!=h:raise ValueError('MSA query identity changed')
        dest=out/'msa_csv'/(h+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['key','sequence'])
            for _,aligned in pairs:
                if len(''.join(c for c in aligned if not c.islower()))!=len(sequence):raise ValueError('MSA aligned width differs')
                writer.writerow([-1,aligned])
        csvs.append(dict(sequence_sha256=h,relative_path=str(dest.relative_to(root)).replace('\\','/'),
                         sha256=sha(dest),source_rows=r['rows']))
    bysha={r['sequence_sha256']:r for r in csvs}
    profiles={}
    for recycling in (3,6):
        name='source_filtered_MSA_r'+str(recycling)+'_s200'
        profiles[name]=dict(software_version='2.0.2',device_class='A100_80GB',
            recycling_steps=recycling,sampling_steps=200,diffusion_samples=1,
            max_parallel_samples=1,step_scale=1.5,templates='none',affinity=False,
            msa='source_filtered_MMseqs2_quality_prefix128',msa_pairing='unpaired_only_key_minus1',
            raw_standard_MSA_not_used=True,no_artificial_pairing=True,
            source_policy_sha256=sha(root/MSA/'review_policy.json'),source_review_completion_sha256=sha(root/MSA/'completion.json'),
            precision='bf16-mixed',num_workers=0,preprocessing_threads=1,no_trifast=True,
            existing_verified_constructs_only=True,additional_ligands='none_context_limitation_disclosed')
    logical=[];physical={};contexts=[]
    weights=read(root/'reports/delepref_v8_predictors/boltz2_preflight.json')['files'][0]['sha256']
    for p in panel:
        specs=[dict(id='A',sequence=p['sequence'],role='design',native_chain_id=p['design_chain'])]
        specs += [dict(id=chr(66+i),sequence=c['sequence'],role='partner',native_chain_id=c['chain_id']) for i,c in enumerate(p.get('partners',[]))]
        for c in specs:
            h=hashlib.sha256(c['sequence'].encode()).hexdigest();r=bysha[h]
            c.update(sequence_sha256=h,msa_relative_path=r['relative_path'],msa_csv_sha256=r['sha256'],source_MSA_rows=r['source_rows'])
        context=dict(protein_id=p['protein_id'],cohort='binding_interface' if p['dataset']=='Binder20' else 'enzyme',
            chains=specs,native_chain_construct_identity='reuses_frozen_reviewed_native_mapping',
            cofactor_and_additional_assembly_context='not_augmented_in_this_WT_only_calibration',
            no_functional_validation=True)
        contexts.append(context)
        canonical=[{k:c[k] for k in ('id','sequence','sequence_sha256','msa_relative_path',
                                     'msa_csv_sha256','source_MSA_rows')} for c in specs]
        for profile_name,profile in profiles.items():
            for seed in SEEDS:
                signature=dict(chains_sha256=digest(canonical),weights_sha256=weights,
                    profile_sha256=digest(profile),seed=seed,sample=0)
                task='WTcal_'+digest(signature)
                if task not in physical:
                    physical[task]=dict(signature,task_id=task,profile_name=profile_name,chains=canonical,
                        sequence_sha256=digest([dict(id=c['id'],sequence=c['sequence']) for c in specs]),
                        wt_variant='WT',predictor='boltz2',category='WT_calibration')
                logical.append(dict(protein_id=p['protein_id'],cohort=context['cohort'],physical_id=task,
                    profile_name=profile_name,seed=seed,native_to_prediction_chains=[
                        dict(native_chain_id=c['native_chain_id'],prediction_chain_id=c['id'],role=c['role']) for c in specs]))
    write(out/'msa_csv_manifest.json',csvs);write(out/'profiles.json',profiles)
    write(out/'contexts.json',contexts);write(out/'physical_requests.json',list(physical.values()))
    write(out/'logical_requests.json',logical)
    write(out/'protocol.json',dict(status='WT_ONLY_PROSPECTIVE_CALIBRATION',methods_compared=False,
        predeclared_seeds=SEEDS,default_profile='source_filtered_MSA_r3_s200',
        higher_recycling_profile='source_filtered_MSA_r6_s200',
        selection='WT_only_review_of_absolute_quality_coverage_failure_sampling_variation_and_cost_before_variants',
        no_weighted_RAG_objective=True,no_WT_survivor_only_roster=True,no_variant_results_used=True,
        MSA_query_only_queries=review['query_only'],scope='23 previously reviewed WT contexts, all results retained',
        unpaired_MSA_and_omitted_extra_cofactors_are_disclosed_limits=True,
        compute=dict(structure_call_limit=None,structure_retries_max=1,structure_categories=dict(WT_calibration=None))))
    names=['analysis/recomb_confirm_v2_wt_calibration.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_confirm_v2_audit_msa.py','analysis/recomb_benign_fold_worker_20261017.py',
        'code/prediction_journal.py','code/budget_ledger.py',
        'slurm/recomb_confirm_v2_WT_preflight.sbatch','slurm/recomb_confirm_v2_WT_calibration.sbatch',
        'reports/recomb_benign_application_constructs_20261017/panel.json',
        'reports/recomb_benign_application_scope_20261017/audit.json',
        'reports/delepref_v8_predictors/boltz2_preflight.json']
    names += [str(p.relative_to(root)).replace('\\','/') for p in out.glob('*.json')]
    names += [r['relative_path'] for r in csvs]
    write(root/FREEZE,dict(status='WT_CALIBRATION_INPUTS_AND_PROTOCOL_FROZEN_NOT_GPU_RELEASED',
        WT=23,logical_predictions=len(logical),physical_predictions=len(physical),structure_call_limit=None,
        files={p:sha(root/p) for p in names},weights_sha256=weights))
    write(out/'prepare_completion.json',dict(status='WT_CALIBRATION_PREPARED',WT=23,
        logical=len(logical),physical=len(physical),reviewed_MSA_queries=len(csvs),
        freeze_sha256=sha(root/FREEZE),new_structure_calls=0))
    print(read(out/'prepare_completion.json'))


def verify(root):
    f=read(root/FREEZE)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen WT input/source differs: '+p)
    return f


def preflight(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU preflight only')
    verify(root)
    from boltz.data.parse.csv import parse_csv
    from boltz.main import predict
    out=root/BASE;parsed=[]
    for r in read(out/'msa_csv_manifest.json'):
        path=root/r['relative_path'];msa=parse_csv(path)
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        width=len(rows[0]['sequence'])
        if hashlib.sha256(rows[0]['sequence'].encode()).hexdigest()!=r['sequence_sha256']:raise ValueError('CSV query SHA changed')
        if len(msa.sequences)<1 or any(int(s['res_end'])-int(s['res_start'])!=width for s in msa.sequences):
            raise ValueError('Boltz parsed MSA width differs')
        if any(int(s['taxonomy'])!=-1 for s in msa.sequences):raise ValueError('Unexpected artificial pairing key')
        parsed.append(dict(sequence_sha256=r['sequence_sha256'],parsed_unique_rows=len(msa.sequences),width=width))
    required={'recycling_steps','sampling_steps','seed','checkpoint','use_msa_server','no_trifast'}
    if not required<=set(inspect.signature(predict.callback).parameters):raise ValueError('Boltz callback contract differs')
    write(out/'preflight.json',dict(status='CPU_PARSER_PREFLIGHT_PASSED',parsed_queries=len(parsed),
        parsed_MSA=parsed,freeze_sha256=sha(root/FREEZE),job_id=os.environ['SLURM_JOB_ID'],new_model_forwards=0))
    print('WT_CALIBRATION_CPU_PREFLIGHT_PASSED',flush=True)


def worker(root,index):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU only via Slurm')
    verify(root);out=root/BASE
    pf=read(out/'preflight.json')
    if pf['status']!='CPU_PARSER_PREFLIGHT_PASSED' or pf['freeze_sha256']!=sha(root/FREEZE):raise ValueError('CPU release not passed')
    old=read(root/'reports/delepref_v8_predictors/boltz2_preflight.json')
    for f in old['files']+old['implementation']+old.get('standard_amino_acid_assets',[]):
        if sha(f['path'])!=f['sha256']:raise ValueError('Predictor asset identity changed')
    import torch
    from boltz.main import predict
    from recomb_benign_fold_worker_20261017 import read_predicted_chains
    import sys
    sys.path.insert(0,str(root/'code'))
    from prediction_journal import PredictionJournal
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('Frozen A100 profile required')
    torch.set_num_threads(4)
    profiles=read(out/'profiles.json');names=sorted(profiles)
    if not 0<=index<6:raise ValueError('Invalid profile/seed array shard')
    profile_name=names[index//3];seed=SEEDS[index%3];profile=profiles[profile_name]
    requests=[p for p in read(out/'physical_requests.json') if p['profile_name']==profile_name and p['seed']==seed]
    journal=PredictionJournal(out/'protocol.json',out/'prediction_journal.jsonl')
    checkpoint=Path(old['files'][0]['path']);failures=0
    for i,r in enumerate(requests):
        key=r['task_id'];record_path=out/'records'/(key+'.json')
        reservation=journal.reserve(task_id=key,sequence_sha=r['sequence_sha256'],wt_variant='WT',
            predictor_weights_sha=r['weights_sha256'],profile_sha=r['profile_sha256'],
            seed=seed,sample=0,category='WT_calibration',job_id=os.environ['SLURM_JOB_ID'])
        if not reservation['new_reservation']:
            if reservation['state'] not in ('succeeded','cache_hit','failed'):raise RuntimeError('Unfinished WT call needs audit')
            continue
        directory=out/'inputs'/key;directory.mkdir(parents=True,exist_ok=True)
        chains=[dict(id=c['id'],sequence=c['sequence']) for c in r['chains']]
        yaml=directory/'input.yaml'
        write(yaml,dict(version=1,sequences=[dict(protein=dict(id=c['id'],sequence=c['sequence'],
            msa=str(root/c['msa_relative_path']))) for c in r['chains']]))
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        journal.start(key);start=time.monotonic()
        print('WT_CALIBRATION_FORWARD_START '+key+' '+str(i+1)+'/'+str(len(requests)),flush=True)
        try:
            predict.callback(data=yaml,out_dir=directory/'output',cache=checkpoint.parent,checkpoint=checkpoint,
                accelerator='gpu',devices=1,model='boltz2',recycling_steps=profile['recycling_steps'],
                sampling_steps=profile['sampling_steps'],diffusion_samples=1,max_parallel_samples=1,
                num_workers=0,seed=seed,step_scale=1.5,output_format='mmcif',preprocessing_threads=1,
                use_msa_server=False,no_trifast=True)
            files=list((directory/'output').rglob('*_model_0.cif'))
            if len(files)!=1:raise ValueError('Expected exactly one calibrated prediction')
            predicted=read_predicted_chains(files[0],chains)
            raw=out/'artifacts'/(key+'.cif');raw.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(files[0],raw)
            artifact=out/'artifacts'/(key+'.npz')
            np.savez_compressed(artifact,coordinates=np.asarray(predicted['A']['coordinates'],dtype=float),
                chains_json=np.array(json.dumps(predicted,allow_nan=False,separators=(',',':'))),
                sequence_sha256=np.array(r['sequence_sha256']))
            journal.finish(key,succeeded=True,seconds=time.monotonic()-start,max_memory=float(torch.cuda.max_memory_allocated()),output_path=artifact)
            write(record_path,dict(r,status='SUCCEEDED',output_path=str(artifact),output_sha256=sha(artifact),
                raw_structure_path=str(raw),raw_structure_sha256=sha(raw),seconds=time.monotonic()-start,
                job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=sha(root/FREEZE)))
            failures=0;print('WT_CALIBRATION_FORWARD_COMPLETE '+key,flush=True)
        except Exception as error:
            journal.finish(key,succeeded=False,seconds=time.monotonic()-start,max_memory=float(torch.cuda.max_memory_allocated()))
            write(record_path,dict(r,status='FAILED',error=repr(error),job_id=os.environ['SLURM_JOB_ID']))
            failures+=1
            if failures>=2:raise RuntimeError('Two consecutive WT failures; inspect before continuing') from error
        finally:torch.cuda.empty_cache()
    write(out/('shard_'+str(index)+'_completion.json'),dict(status='WT_CALIBRATION_SHARD_FINISHED',
        index=index,profile=profile_name,seed=seed,requests=len(requests),job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REMOTE)
    p.add_argument('--stage',choices=['prepare','preflight','worker'],required=True);p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='preflight':preflight(a.root)
    else:worker(a.root,a.index)
