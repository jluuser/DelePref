"""Create exact prediction identities and verify compatible caches without folding."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from recomb_scope_screen_20261017 import read, sha, write

BASE = Path('reports/recomb_benign_prediction_catalog_20261017')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def identity_mapping(sequence):
    return dict(method='identity_WT',wt_to_edit=list(range(len(sequence))),position_states=['stable']*len(sequence),
        stable_positions=len(sequence),certain_deleted_positions=0,ambiguous_positions=0)


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU Slurm required')
    out = root/BASE
    if out.exists():
        raise FileExistsError('prediction catalog exists; do not overwrite')
    sources = {}
    def load(p):
        sources[p] = sha(root/p); return read(root/p)
    general = {p['protein_id']:p for p in load('reports/recomb_benign_scope_20261017/general_panel.json')}
    applications = {p['protein_id']:p for p in load('reports/recomb_benign_application_constructs_20261017/panel.json')}
    sub = load('reports/recomb_benign_final_subsets_20261017/subsets.json')
    b = 'reports/recomb_benign_final_outputs_20261017/'
    done = load(b+'completion.json')
    for n,h in done['outputs'].items():
        if sha(root/(b+n)) != h:
            raise ValueError('final output matrix changed')
    core = load(b+'core_selected_outputs.json'); mechanisms = load(b+'mechanism_selected_outputs.json')
    apps_path = 'reports/recomb_benign_application_outputs_20261017/all_selected_outputs.json'
    if sha(root/apps_path) != load('reports/recomb_benign_application_outputs_20261017/completion.json')['output_sha256']:
        raise ValueError('application matrix changed')
    app_rows = load(apps_path)
    for name in ['binding','enzyme_native_v2']:
        folder = 'recomb_benign_binding_native_20261017' if name=='binding' else 'recomb_benign_enzyme_native_v2_20261017'
        reference_path = 'reports/'+folder+'/native_references.json'
        if sha(root/reference_path) != load('reports/'+folder+'/completion.json')['output_sha256']:
            raise ValueError('reviewed application reference changed')
        sources[reference_path] = sha(root/reference_path)
    for p in general.values():
        path = Path(p['wt_native_json'])
        if sha(path) != p['wt_native_json_sha256']:
            raise ValueError('general native reference changed')
        native = read(path)
        if native['wt_sequence_sha256'] != p['sequence_sha256']:
            raise ValueError('general native/WT mismatch')
        sources[str(path)] = sha(path)
    old_profiles = load('reports/delepref_v8_structure_preparation/predictor_profiles.json')
    profiles = dict(esmfold=dict(old_profiles['esmfold']),boltz_general=dict(old_profiles['boltz2']),
        boltz_binding=dict(old_profiles['boltz2'],seed=20260726,recycling_steps=1,sampling_steps=50),
        boltz_enzyme=dict(old_profiles['boltz2'],seed=20260908,recycling_steps=3,sampling_steps=200))
    reports = {name:load('reports/delepref_v8_predictors/'+name+'_preflight.json') for name in ['esmfold','boltz2']}
    weights = {name: next(f['sha256'] for f in report['files'] if f['path'].endswith(('pytorch_model.bin','boltz2_conf.ckpt')))
        for name,report in reports.items()}
    logical = []; physical = {}
    def add(row,profile_name,cohort,role):
        pid = row['protein_id']; p = general[pid] if cohort=='general' else applications[pid]
        if row['WT_sha256'] != p['sequence_sha256']:
            raise ValueError('logical WT identity changed')
        is_app = cohort != 'general'
        raw_chains = row.get('chains') if is_app else None
        raw_chains = raw_chains or [dict(native_chain_id=p.get('reference_chain',p.get('design_chain','A')),role='design',sequence=row['edited_sequence'])]
        chains = [dict(id=chr(ord('A')+i),sequence=c['sequence']) for i,c in enumerate(raw_chains)]
        if len(chains)>26 or chains[0]['sequence'] != row['edited_sequence']:
            raise ValueError('canonical chain identity mismatch')
        seq_sha = hashlib.sha256(chains[0]['sequence'].encode()).hexdigest() if len(chains)==1 else digest(chains)
        predictor = 'esmfold' if profile_name=='esmfold' else 'boltz2'
        profile = profiles[profile_name]
        signature = dict(sequence_sha256=seq_sha,weights_sha256=weights[predictor],
            profile_sha256=digest(profile),seed=profile['seed'],sample=profile['sample'])
        task = predictor+'_'+digest(signature)
        if task not in physical:
            physical[task] = dict(signature,task_id=task,predictor=predictor,profile_name=profile_name,
                sequence=chains[0]['sequence'] if len(chains)==1 else None,chains=chains,canonical_chains_sha256=digest(chains),
                category='main_or_application_final_output',wt_variant='WT' if row['method']=='WT' else 'variant',
                requires_all_atom_output=cohort=='binding_interface')
        elif physical[task]['chains'] != chains:
            raise ValueError('prediction identity collision')
        request = dict(protein_id=pid,cohort=cohort,method=row['method'],budget=row['budget'],
            evaluation_role=role,profile_name=profile_name,predictor=predictor,physical_id=task,
            WT_sha256=p['sequence_sha256'],edited_sequence_sha256=row['edited_sequence_sha256'],
            correspondence=row['correspondence'],native_to_prediction_chains=[dict(native_chain_id=c['native_chain_id'],prediction_chain_id=chains[i]['id'],role=c['role']) for i,c in enumerate(raw_chains)])
        request['request_id'] = digest(request); logical.append(request)
    for row in core+mechanisms:
        add(row,'esmfold','general',row['evaluation_role'])
    for p in general.values():
        row = dict(protein_id=p['protein_id'],WT_sha256=p['sequence_sha256'],edited_sequence=p['sequence'],
            edited_sequence_sha256=p['sequence_sha256'],method='WT',budget=0.,correspondence=identity_mapping(p['sequence']))
        add(row,'esmfold','general','WT_control')
    for row in app_rows:
        cohort = 'binding_interface' if row['dataset']=='Binder20' else 'enzyme'
        add(row,'boltz_binding' if cohort=='binding_interface' else 'boltz_enzyme',cohort,'application_primary20pct')
    for p in applications.values():
        chains = [dict(native_chain_id=p['design_chain'],role='design',sequence=p['sequence'])]
        chains += [dict(native_chain_id=c['chain_id'],role='partner',sequence=c['sequence']) for c in p['partners']]
        row = dict(protein_id=p['protein_id'],WT_sha256=p['sequence_sha256'],edited_sequence=p['sequence'],
            edited_sequence_sha256=p['sequence_sha256'],method='WT',budget=0.,correspondence=identity_mapping(p['sequence']),chains=chains)
        cohort = 'binding_interface' if p['dataset']=='Binder20' else 'enzyme'
        add(row,'boltz_binding' if cohort=='binding_interface' else 'boltz_enzyme',cohort,'WT_control')
    main_methods = {'delepref_unlimited','rag_unlimited','terminal','scisor_native','raygun_native'}
    for row in core:
        if row['protein_id'] in sub['second_predictor'] and row['budget']==.2 and row['method'] in main_methods:
            add(row,'boltz_general','general','second_predictor')
    for pid in sub['second_predictor']:
        p = general[pid]
        row = dict(protein_id=pid,WT_sha256=p['sequence_sha256'],edited_sequence=p['sequence'],edited_sequence_sha256=p['sequence_sha256'],
            method='WT',budget=0.,correspondence=identity_mapping(p['sequence']))
        add(row,'boltz_general','general','second_predictor_WT')
    if len(logical)!=867 or len({r['request_id'] for r in logical})!=867:
        raise ValueError('final logical request roster differs')
    reuse = {}; rejected = []
    cache_dirs = ['reports/delepref_v8_structures','reports/recomb_task_structures_20261017_v2']
    for request in physical.values():
        if request['requires_all_atom_output'] or request['profile_name'] in ('boltz_binding','boltz_enzyme'):
            continue
        # Open only files named by the currently approved requested sequence hash.
        old_id = request['predictor']+'_'+request['sequence_sha256']
        for folder in cache_dirs:
            path = root/folder/'records'/(old_id+'.json')
            if not path.exists():
                continue
            record = read(path)
            keys = ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']
            if record.get('status')!='SUCCEEDED' or any(record.get(k)!=request[k] for k in keys) or record.get('sequence')!=request['sequence']:
                rejected.append(dict(task_id=request['task_id'],cache_record=str(path),reason='prediction_identity_mismatch')); continue
            artifact = Path(record['output_path'])
            if not artifact.is_file() or sha(artifact)!=record['output_sha256']:
                raise ValueError('cache artifact missing or SHA mismatch')
            with np.load(artifact,allow_pickle=False) as z:
                if str(z['sequence'])!=request['sequence'] or z['coordinates'].shape!=(len(request['sequence']),3):
                    raise ValueError('cached coordinate/sequence identity mismatch')
            reuse[request['task_id']] = dict(record_path=str(path),record_sha256=sha(path),artifact_path=str(artifact),artifact_sha256=record['output_sha256'])
            break
    out.mkdir()
    write(out/'logical_requests.json',logical); write(out/'physical_requests.json',list(physical.values()))
    write(out/'compatible_cache.json',reuse); write(out/'cache_rejections.json',rejected); write(out/'predictor_profiles.json',profiles)
    profile_counts = Counter(r['profile_name'] for r in physical.values())
    new_counts = Counter(r['profile_name'] for r in physical.values() if r['task_id'] not in reuse)
    sources['analysis/recomb_benign_prediction_catalog_20261017.py'] = sha(Path(__file__))
    report = dict(status='FINAL_SEQUENCE_AND_PROFILE_CATALOG_NOT_RELEASED_FOR_FOLDING',logical_requests=867,
        general_logical=704,application_logical=115,second_predictor_logical=48,unique_physical=len(physical),
        physical_by_profile=dict(profile_counts),verified_compatible_cache=len(reuse),missing_final_predictions=len(physical)-len(reuse),
        missing_by_profile=dict(new_counts),all_binding_application_caches_rechecked_by_new_full_construct=False,
        legacy_application_cache_policy='not reused without complete prediction identity; predict fixed final outputs afresh',
        native_chain_mapping_retained_per_logical_record=True,structure_prediction_limit=None,
        fold_allowed=False,new_structure_predictions=0,source_sha256=sources,
        pending=['freeze executable metric/evaluator and worker identity, final statistics and predictor checks before dispatch'],
        outputs={n:sha(out/n) for n in ['logical_requests.json','physical_requests.json','compatible_cache.json','cache_rejections.json','predictor_profiles.json']})
    write(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
