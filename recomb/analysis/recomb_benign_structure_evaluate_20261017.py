"""Whole-roster multi-endpoint evaluation; no selector or sample updates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_metrics_20261017 import geometry,binding_contacts,catalytic_geometry

CAT=Path('reports/recomb_benign_prediction_catalog_20261017')
STRUCT=Path('reports/recomb_benign_structures_20261017')
OUT=Path('reports/recomb_benign_structure_evaluation_20261017')


def coords(values):
    out=np.full((len(values),3),np.nan)
    for i,v in enumerate(values):
        if v is not None:out[i]=v
    return out


def paired_summary(pairs,components,seed=20261016):
    if not pairs:return dict(status='EMPTY',N=0)
    if any(p['difference_lower'] is None or p['difference_upper'] is None for p in pairs):
        return dict(status='UNDEFINED_METRIC_PRESENT_NO_SILENT_CASE_REMOVAL',N=len(pairs),
            N_defined=sum(p['difference_lower'] is not None and p['difference_upper'] is not None for p in pairs))
    lower=np.array([p['difference_lower'] for p in pairs]);upper=np.array([p['difference_upper'] for p in pairs])
    labels=[components[p['protein_id']] for p in pairs];unique=sorted(set(labels));rng=np.random.default_rng(seed)
    sums_l=np.array([lower[np.array(labels)==label].sum() for label in unique])
    sums_u=np.array([upper[np.array(labels)==label].sum() for label in unique])
    counts=np.array([labels.count(label) for label in unique])
    sample=rng.integers(0,len(unique),size=(10000,len(unique)))
    boot_l=sums_l[sample].sum(1)/counts[sample].sum(1);boot_u=sums_u[sample].sum(1)/counts[sample].sum(1)
    tolerance=1e-8
    return dict(status='COMPONENT_BOOTSTRAP_WITH_METRIC_UNCERTAINTY',N=len(pairs),components=len(unique),
        construct_macro_difference_lower=float(lower.mean()),construct_macro_difference_upper=float(upper.mean()),
        median_difference_lower=float(np.median(lower)),median_difference_upper=float(np.median(upper)),
        uncertainty_and_bootstrap_interval95=[float(np.quantile(boot_l,.025)),float(np.quantile(boot_u,.975))],
        confirmed_improved=int((lower>tolerance).sum()),confirmed_degraded=int((upper < -tolerance).sum()),
        numerically_unchanged=int(((lower>=-tolerance)&(upper<=tolerance)).sum()),
        unresolved=int((~((lower>tolerance)|(upper < -tolerance)|((lower>=-tolerance)&(upper<=tolerance)))).sum()),
        bootstrap_seed=seed,bootstrap_replicates=10000)


def evaluate(root,checksum):
    freeze=root/'configs/recomb_benign_confirmation_freeze_20261017.json'
    if sha(freeze)!=checksum:raise ValueError('confirmation freeze changed')
    for f in read(freeze)['files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('frozen input/source changed')
    out=root/OUT
    if out.exists():raise FileExistsError('evaluation exists; preserve it')
    logical=read(root/CAT/'logical_requests.json');physical={p['task_id']:p for p in read(root/CAT/'physical_requests.json')}
    reuse=read(root/CAT/'compatible_cache.json');protocol=read(root/CAT/'statistical_protocol.json')
    general={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    bindings={p['protein_id']:p for p in read(root/'reports/recomb_benign_binding_native_20261017/native_references.json')}
    enzymes={p['protein_id']:p for p in read(root/'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json')}
    binary=root/'provenance/usalign_20261017/USalign'
    predictions={};statuses={};audit=[]
    for task,p in physical.items():
        if task in reuse:
            old=reuse[task];path=Path(old['artifact_path']);expected=old['artifact_sha256'];status='VERIFIED_CACHE'
        else:
            record_path=root/STRUCT/'records'/(task+'.json')
            record=read(record_path) if record_path.exists() else {}
            status=record.get('status','NOT_PRODUCED')
            path=Path(record['output_path']) if status=='SUCCEEDED' else None
            expected=record.get('output_sha256')
            if status=='SUCCEEDED' and any(record[k]!=p[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):
                raise ValueError('prediction record identity mismatch')
            if status=='SUCCEEDED' and record.get('raw_structure_path') and sha(record['raw_structure_path'])!=record['raw_structure_sha256']:
                raise ValueError('raw all-atom structure changed')
        if path is not None:
            if sha(path)!=expected:raise ValueError('prediction artifact changed')
            with np.load(path,allow_pickle=False) as z:
                if str(z['sequence_sha256'])!=p['sequence_sha256']:raise ValueError('prediction input SHA changed')
                if 'chains_json' in z:chains=json.loads(str(z['chains_json']))
                else:chains={'A':dict(sequence=p['chains'][0]['sequence'],coordinates=z['coordinates'].tolist(),heavy_atoms=[])}
                for chain in p['chains']:
                    if chains[chain['id']]['sequence']!=chain['sequence']:raise ValueError('prediction chain identity mismatch')
                if p['requires_all_atom_output'] and not chains['A']['heavy_atoms']:raise ValueError('binding predictor output lacks heavy atoms')
        else:
            chains={c['id']:dict(sequence=c['sequence'],coordinates=[None]*len(c['sequence']),heavy_atoms=[]) for c in p['chains']}
        predictions[task]=chains;statuses[task]=status
        audit.append(dict(physical_id=task,status=status,artifact_path=str(path) if path else None,artifact_sha256=expected))
    result=[]
    for row in logical:
        pid=row['protein_id'];chains=predictions[row['physical_id']];prediction=coords(chains['A']['coordinates'])
        if row['cohort']=='general':
            reference=read(Path(general[pid]['wt_native_json']));native=np.full((len(general[pid]['sequence']),3),np.nan)
            for i,v in reference['coordinates'].items():native[int(i)]=v
        elif row['cohort']=='binding_interface':native=coords(bindings[pid]['chains'][0]['native_CA'])
        else:native=coords(enzymes[pid]['native_CA'])
        value=dict(row,prediction_status=statuses[row['physical_id']],geometry=geometry(native,prediction,row['correspondence'],binary))
        if row['cohort']=='binding_interface':value['interface']=binding_contacts(bindings[pid],chains,row['correspondence'],row['native_to_prediction_chains'])
        if row['cohort']=='enzyme':value['catalytic']=catalytic_geometry(enzymes[pid],prediction,row['correspondence'])
        result.append(value)
    if len(result)!=867:raise ValueError('logical roster lost')
    def bounds(row,endpoint):
        if endpoint=='full_WT_TM':return row['geometry']['full_WT_TM']['lower'],row['geometry']['full_WT_TM']['upper']
        block='interface' if endpoint.startswith('interface_') else 'catalytic' if endpoint.startswith('catalytic_') else 'geometry'
        key='contact_recovery' if endpoint=='interface_contact_recovery' else endpoint
        return row[block].get(key+'_lower'),row[block].get(key+'_upper')
    subsets=read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json')
    groups={}
    groups['general_primary']=[r for r in result if r['cohort']=='general' and r['profile_name']=='esmfold' and r['budget']==.2 and r['evaluation_role']=='comparison']
    groups['general_budget_curve']=[r for r in result if r['cohort']=='general' and r['profile_name']=='esmfold' and r['protein_id'] in subsets['completion_ablation'] and r['evaluation_role']=='comparison']
    groups['general_mechanisms']=[r for r in result if r['cohort']=='general' and r['profile_name']=='esmfold' and r['budget']==.2 and r['protein_id'] in subsets['completion_ablation']]
    groups['general_second_predictor']=[r for r in result if r['profile_name']=='boltz_general']
    groups['binding_interface']=[r for r in result if r['cohort']=='binding_interface']
    groups['enzyme']=[r for r in result if r['cohort']=='enzyme']
    all_pairs=[];summaries=[]
    for name,rows in groups.items():
        methods=sorted({r['method'] for r in rows}-{'WT','delepref_unlimited'})
        endpoints=['retained_ca_lddt','contact_recovery','full_WT_TM']
        if name=='binding_interface':endpoints+=['interface_contact_recovery','interface_jaccard']
        if name=='enzyme':endpoints+=['catalytic_neighbor_contact_recovery']
        baseline={(r['protein_id'],r['budget']):r for r in rows if r['method']=='delepref_unlimited'}
        for method in methods:
            by_budget=defaultdict(list)
            for r in rows:
                if r['method']==method:by_budget[r['budget']].append(r)
            for budget,selected in sorted(by_budget.items()):
                for endpoint in endpoints:
                    pairs=[]
                    for r in selected:
                        a,b=bounds(r,endpoint);c,d=bounds(baseline[(r['protein_id'],budget)],endpoint)
                        pairs.append(dict(experiment=name,method=method,budget=budget,endpoint=endpoint,protein_id=r['protein_id'],
                            difference_lower=a-d if a is not None and d is not None else None,
                            difference_upper=b-c if b is not None and c is not None else None))
                    all_pairs+=pairs
                    summaries.append(dict(experiment=name,method=method,budget=budget,endpoint=endpoint,
                        **paired_summary(pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    out.mkdir();write(out/'per_output_metrics.json',result);write(out/'per_protein_paired_differences.json',all_pairs)
    write(out/'paired_comparisons.json',summaries);write(out/'prediction_identity_audit.json',audit)
    report=dict(status='WHOLE_ROSTER_STRUCTURAL_PROXY_EVALUATION_COMPLETE',logical_outputs=867,physical_predictions=len(physical),
        failures_or_missing=sum(s not in ('SUCCEEDED','VERIFIED_CACHE') for s in statuses.values()),
        confirmation_freeze_sha256=checksum,no_model_or_output_reselection=True,not_functional_validation=True,
        outputs={n:sha(out/n) for n in ['per_output_metrics.json','per_protein_paired_differences.json','paired_comparisons.json','prediction_identity_audit.json']})
    write(out/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--freeze-sha256',required=True);a=p.parse_args();evaluate(a.root,a.freeze_sha256)
