"""All fixed application outputs, biological identity, uncertainty and seed-level statistics."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_general_evaluate import effect
from recomb_confirm_v2_structure_alignment import aligned_geometry
from recomb_benign_metrics_20261017 import geometry,catalytic_geometry
from recomb_benign_catalytic_identity_20261017 import retained_identity
from recomb_confirm_v2_application_metrics import (coordinates,choose_role_map,interface_contacts,
    backbone_components,enzyme_catalytic_geometry)

CAT=Path('reports/recomb_confirm_v2_application_catalog_v2')
STRUCT=Path('reports/recomb_confirm_v2_application_structures')
OUT=Path('reports/recomb_confirm_v2_application_evaluation')
CONFIRM=Path('configs/recomb_confirm_v2_application_confirmation_freeze.json')


@lru_cache(maxsize=160)
def native_reference(path):return read(path)


def metric_one(item):
    root_s,key,row,artifact,allowed,required=item;root=Path(root_s)
    native=native_reference(row['native_assembly_reference']);predicted={}
    if artifact:
        with np.load(artifact,allow_pickle=False) as z:predicted=json.loads(str(z['chains_json']))
    mapping,map_audit=choose_role_map(native,predicted,row['correspondence'],allowed)
    primary=next(c for c in native['chains'] if c['predictor_chain_id']==native['primary_target_chain'])
    output=predicted.get(mapping[primary['predictor_chain_id']],{})
    prediction=coordinates(output['coordinates']) if output else np.full((len(row['edited_sequence']),3),np.nan)
    reference=coordinates(primary['native_CA']);binary=root/'provenance/usalign_20261017/USalign'
    result=dict(role_mapping=map_audit,target_gene_copies=sum(c['target_gene_copy'] for c in native['chains']),
        primary_native_design_instance=native['primary_target_chain'],
        common_structure=aligned_geometry(reference,prediction,binary),
        fixed_identity_geometry=geometry(reference,prediction,row['correspondence'],binary))
    if row['cohort']=='Binder60':
        result['interface']=interface_contacts(native,predicted,row['correspondence'],mapping,required)
        result['H_light_interface']=interface_contacts(native,predicted,row['correspondence'],mapping,required,'fixed_light_chain')
        result['backbone_components']=backbone_components(native,predicted,row['correspondence'],mapping)
    else:
        sites=[dict(status='PASS',wt_index=s['wt_index'],wt_residue=s['WT_residue']) for s in native['catalytic']]
        result['catalytic_identity']=retained_identity(sites,row['edited_sequence'],row['correspondence'])
        old=dict(native_CA=primary['native_CA'],catalytic=sites)
        result['catalytic_position_geometry']=catalytic_geometry(old,prediction,row['correspondence'])
        result['catalytic_identity_geometry']=enzyme_catalytic_geometry(native,predicted,row['correspondence'],mapping)
    return key,result


def endpoint_bounds(row,name):
    if name=='complete_catalytic_AA_identity':
        b=row['catalytic_identity'];a=b['all_verified_identities_retained_lower'];z=b['all_verified_identities_retained_upper']
    elif name=='catalytic_neighbor_contact_recovery':
        b=row['catalytic_position_geometry'];a=b['catalytic_neighbor_contact_recovery_lower'];z=b['catalytic_neighbor_contact_recovery_upper']
    elif name=='identity_aware_catalytic_neighbor_contact_recovery':
        b=row['catalytic_identity_geometry'];a=b['identity_aware_catalytic_neighbor_contact_recovery_lower'];z=b['identity_aware_catalytic_neighbor_contact_recovery_upper']
    elif name in ['interface_contact_recovery','interface_F1']:
        b=row['interface'];field='contact_recovery' if name=='interface_contact_recovery' else name
        a=b[field+'_lower'];z=b[field+'_upper']
    elif name in ['native_interface_backbone_RMSD_A','antigen_fit_H_backbone_RMSD_A']:
        a=z=row['backbone_components'][name]
    else:a=z=row['common_structure'].get(name)
    return (float(a),float(z)) if a is not None and z is not None else (None,None)


def seed_mean(bounds,expected=3):
    if len(bounds)!=expected or any(a is None or b is None for a,b in bounds):return None,None
    return float(np.mean([a for a,b in bounds])),float(np.mean([b for a,b in bounds]))


def summary_entry(rows,endpoint):
    vals=[endpoint_bounds(r,endpoint) for r in rows]
    defined=sum(a is not None and b is not None for a,b in vals)
    return dict(N_all=len(rows),N_defined=defined,
        mean_lower=float(np.mean([a for a,b in vals])) if defined==len(rows) and rows else None,
        mean_upper=float(np.mean([b for a,b in vals])) if defined==len(rows) and rows else None,
        no_silent_case_exclusion=True)


def paired_bounds(ids,method,baseline,by,endpoint,components):
    lower=[];upper=[]
    for pid in sorted(ids):
        a,b=endpoint_bounds(by[pid,method],endpoint);c,d=endpoint_bounds(by[pid,baseline],endpoint)
        lower.append((pid,a-d if a is not None and d is not None else None))
        upper.append((pid,b-c if b is not None and c is not None else None))
    lo=effect(lower,components);hi=effect(upper,components)
    interval=[lo['CI95'][0],hi['CI95'][1]] if 'CI95' in lo and 'CI95' in hi else None
    return dict(method=method,versus=baseline,endpoint=endpoint,lower_effect=lo,upper_effect=hi,CI95_conservative_envelope=interval)


def evaluate(root,phase):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    for p in read(root/CONFIRM)['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen application input/source changed')
    out=root/OUT/phase
    if (out/'completion.json').exists():raise FileExistsError('Application evaluation phase already complete')
    logical=read(root/CAT/'logical_requests.json');specs={r['task_id']:r for r in read(root/CAT/'physical_requests.json')}
    if phase=='WT':logical=[r for r in logical if r['method']=='WT' and r['seed']==20261017]
    elif phase=='main':logical=[r for r in logical if r['seed']==20261017 and r['evaluation_role'] in ['primary20pct','WT_reference']]
    expected=dict(WT=140,main=980,all=2160)[phase]
    if len(logical)!=expected:raise ValueError('Application fixed logical roster differs')
    protocol=read(root/CAT/'statistical_protocol.json');components=protocol['component_by_protein_id']
    meta=root/'reports/recomb_confirm_v2_application_metric_preflight'
    required=read(meta/'required_protein_heavy_atoms.json');allowed={r['protein_id']:r['allowed_maps'] for r in read(meta/'allowed_role_maps.json')}
    statuses={};artifacts={};audit=[]
    for task in {r['physical_id'] for r in logical}:
        p=root/STRUCT/'records'/(task+'.json');saved=read(p) if p.exists() else {};r=specs[task]
        status=saved.get('status','NOT_PRODUCED');artifact=Path(saved['output_path']) if status=='SUCCEEDED' else None
        if artifact:
            if any(saved.get(k)!=r[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample','model_input_sha256']):raise ValueError('Full context prediction identity changed')
            if sha(artifact)!=saved['output_sha256'] or sha(saved['raw_structure_path'])!=saved['raw_structure_sha256']:raise ValueError('Complex prediction artifact changed')
            with np.load(artifact,allow_pickle=False) as z:
                if str(z['sequence'])!=r['sequence'] or str(z['sequence_sha256'])!=r['sequence_sha256']:raise ValueError('Stored sequence/full-context digest differs')
                chains=json.loads(str(z['chains_json']))
                if set(chains)!={c['id'] for c in r['chains']} or any(chains[c['id']]['sequence']!=c['sequence'] for c in r['chains']):raise ValueError('Predicted protein copies/sequence differs')
        statuses[task]=status;artifacts[task]=str(artifact) if artifact else None
        audit.append(dict(physical_id=task,status=status,artifact_sha256=saved.get('output_sha256'),raw_structure_sha256=saved.get('raw_structure_sha256')))
    jobs={}
    for row in logical:
        key=(row['protein_id'],row['physical_id'],json.dumps(row['correspondence'],sort_keys=True))
        jobs.setdefault(key,(str(root),key,row,artifacts[row['physical_id']],allowed[row['protein_id']],required))
    with ProcessPoolExecutor(max_workers=8) as executor:metrics=dict(executor.map(metric_one,jobs.values()))
    result=[]
    for row in logical:
        key=(row['protein_id'],row['physical_id'],json.dumps(row['correspondence'],sort_keys=True))
        result.append(dict(row,prediction_status=statuses[row['physical_id']],**metrics[key]))
    groups=defaultdict(list)
    for row in result:
        if row['seed']==20261017:groups[row['cohort'],row['method'],row['budget']].append(row)
    shape=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM']
    endpoints={'Binder60':['interface_contact_recovery','interface_F1','native_interface_backbone_RMSD_A','antigen_fit_H_backbone_RMSD_A']+shape,
        'Enzyme80':['complete_catalytic_AA_identity','catalytic_neighbor_contact_recovery','identity_aware_catalytic_neighbor_contact_recovery']+shape}
    tables=[]
    for (cohort,method,budget),rows in sorted(groups.items()):
        tables.append(dict(cohort=cohort,method=method,budget=budget,N_all=len(rows),
            endpoints={e:summary_entry(rows,e) for e in endpoints[cohort]}))
    comparisons=[]
    for cohort in ['Binder60','Enzyme80']:
        for budget in sorted({r['budget'] for r in result if r['cohort']==cohort and r['method']!='WT' and r['seed']==20261017}):
            selected=[r for r in result if r['cohort']==cohort and r['budget']==budget and r['method']!='WT' and r['seed']==20261017]
            by={(r['protein_id'],r['method']):r for r in selected};ids={r['protein_id'] for r in selected}
            for method in protocol['methods']:
                if method=='delepref_unlimited':continue
                for endpoint in endpoints[cohort]:
                    comparisons.append(dict(cohort=cohort,budget=budget,
                        primary_comparison=budget==.2 and method=='rag_unlimited' and endpoint==protocol['primary_endpoint'][cohort],
                        **paired_bounds(ids,method,'delepref_unlimited',by,endpoint,components)))
    seed_rows=[]
    if phase=='all':
        for cohort,ids in protocol['repeat_subset_by_cohort'].items():
            grouped=defaultdict(list)
            for row in result:
                if row['cohort']==cohort and row['protein_id'] in ids and (row['budget']==.2 or row['method']=='WT'):
                    grouped[row['protein_id'],row['method']].append(row)
            for (pid,method),rows in grouped.items():
                if sorted(r['seed'] for r in rows)!=protocol['structure_seeds']:raise ValueError('Lost or duplicated fixed structure seed')
                values={}
                for endpoint in endpoints[cohort]:
                    bounds=[endpoint_bounds(r,endpoint) for r in rows];lower,upper=seed_mean(bounds)
                    exact=all(a is not None and a==b for a,b in bounds)
                    values[endpoint]=dict(mean_lower=lower,mean_upper=upper,
                        point_seed_SD=float(np.std([a for a,b in bounds],ddof=1)) if exact else None,
                        point_worst_seed=(max if 'RMSD' in endpoint else min)(a for a,b in bounds) if exact else None,
                        uncertain_seeds_not_replaced_by_point_estimates=not exact)
                seed_rows.append(dict(protein_id=pid,cohort=cohort,method=method,seed_count=3,endpoints=values))
    outputs=dict(per_output_metrics=result,mean_tables=tables,paired_comparisons=comparisons,
        within_WT_three_seed_summaries=seed_rows,prediction_identity_audit=audit)
    for name,value in outputs.items():write(out/(name+'.json'),value)
    done=dict(status='APPLICATION_FROZEN_METRICS_COMPLETE_NOT_FUNCTIONAL_VALIDATION',phase=phase,
        logical=len(result),physical=len(artifacts),missing_or_failed=sum(s!='SUCCEEDED' for s in statuses.values()),
        full_rosters_and_unknown_bounds_preserved=True,primary_identity_not_inferred_from_shape=True,
        all_seed_statistics_within_WT_no_pseudo_replication=True,
        seed_mean_component_comparisons_and_human_readable_report_pending=phase=='all',
        outputs={n+'.json':sha(out/(n+'.json')) for n in outputs},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['WT','main','all'],required=True)
    a=p.parse_args();evaluate(a.root,a.phase)
