"""All-WT, all-seed calibration summary; no variant data or profile search."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_benign_metrics_20261017 import geometry,binding_contacts,catalytic_geometry
from recomb_benign_structure_evaluate_20261017 import paired_summary


def coordinates(values):
    return np.asarray([v if v is not None else [np.nan]*3 for v in values],dtype=float)


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    frozen=read(root/'configs/recomb_confirm_v2_WT_statistics_freeze.json')
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('WT metric/source input changed: '+p)
    if sha(root/'provenance/usalign_20261017/USalign')!=frozen['USalign_sha256']:
        raise ValueError('USalign binary changed')
    base=root/'reports/recomb_confirm_v2_WT_calibration'
    out=root/'reports/recomb_confirm_v2_WT_evaluation'
    if (out/'completion.json').exists():raise FileExistsError('WT summary already exists')
    logical=read(base/'logical_requests.json')
    physical={p['task_id']:p for p in read(base/'physical_requests.json')}
    bindings={r['protein_id']:r for r in read(root/'reports/recomb_benign_binding_native_20261017/native_references.json')}
    enzymes={r['protein_id']:r for r in read(root/'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json')}
    states={};predicted={};integrity=[]
    for key,p in physical.items():
        path=base/'records'/(key+'.json')
        record=read(path) if path.exists() else {}
        status=record.get('status','NOT_PRODUCED');states[key]=status
        if status=='SUCCEEDED':
            for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']:
                if record[k]!=p[k]:raise ValueError('WT prediction identity changed')
            if sha(record['output_path'])!=record['output_sha256'] or sha(record['raw_structure_path'])!=record['raw_structure_sha256']:
                raise ValueError('WT prediction artifact bytes changed')
            with np.load(record['output_path'],allow_pickle=False) as data:
                chains=json.loads(str(data['chains_json']))
            for c in p['chains']:
                if chains[c['id']]['sequence']!=c['sequence']:raise ValueError('WT predicted chain sequence differs')
            predicted[key]=chains
        integrity.append(dict(physical_id=key,status=status,seconds=record.get('seconds'),
            output_sha256=record.get('output_sha256'),job_id=record.get('job_id')))
    metrics=[]
    for r in logical:
        pid=r['protein_id'];p=physical[r['physical_id']]
        row={k:r[k] for k in ['protein_id','cohort','physical_id','profile_name','seed']}
        row['status']=states[r['physical_id']]
        if row['status']=='SUCCEEDED':
            chains=predicted[r['physical_id']];prediction=coordinates(chains['A']['coordinates'])
            native=coordinates(bindings[pid]['chains'][0]['native_CA'] if pid in bindings else enzymes[pid]['native_CA'])
            correspondence=dict(wt_to_edit=list(range(len(native))),position_states=['stable']*len(native))
            g=geometry(native,prediction,correspondence,root/'provenance/usalign_20261017/USalign')
            row.update(retained_ca_lddt=g['retained_ca_lddt_lower'],contact_recovery=g['contact_recovery_lower'],
                full_WT_TM_lower=g['full_WT_TM']['lower'],full_WT_TM_upper=g['full_WT_TM']['upper'],
                native_CA_coverage=float(np.isfinite(native).all(1).mean()))
            if pid in bindings:
                interface=binding_contacts(bindings[pid],chains,correspondence,r['native_to_prediction_chains'])
                row['interface_contact_recovery']=interface['contact_recovery_lower']
                row['interface_jaccard']=interface['interface_jaccard_lower']
            else:
                cat=catalytic_geometry(enzymes[pid],prediction,correspondence)
                row['catalytic_neighbor_contact_recovery']=cat['catalytic_neighbor_contact_recovery_lower']
            row['MSA_source_rows_by_chain']={c['id']:c['source_MSA_rows'] for c in p['chains']}
        metrics.append(row)
    endpoints=['retained_ca_lddt','contact_recovery','full_WT_TM_lower','full_WT_TM_upper',
        'interface_contact_recovery','interface_jaccard','catalytic_neighbor_contact_recovery']
    groups=defaultdict(list)
    for r in metrics:groups[r['protein_id'],r['profile_name']].append(r)
    cases=[]
    for (pid,profile),rows in sorted(groups.items()):
        if len(rows)!=3:raise ValueError('Logical WT seed roster incomplete')
        case=dict(protein_id=pid,cohort=rows[0]['cohort'],profile_name=profile,
            predeclared_seeds=sorted(r['seed'] for r in rows),successful_seeds=sum(r['status']=='SUCCEEDED' for r in rows))
        for endpoint in endpoints:
            values=[r.get(endpoint) for r in rows]
            case[endpoint]=float(np.mean(values)) if all(v is not None for v in values) else None
            case[endpoint+'_seed_sd']=float(np.std(values,ddof=1)) if all(v is not None for v in values) else None
        cases.append(case)
    table=[]
    for cohort in ['binding_interface','enzyme']:
        for profile in sorted({r['profile_name'] for r in cases}):
            selected=[r for r in cases if r['cohort']==cohort and r['profile_name']==profile]
            row=dict(cohort=cohort,profile_name=profile,N=len(selected),
                successful_logical_predictions=sum(r['successful_seeds'] for r in selected))
            for endpoint in endpoints:
                values=[r[endpoint] for r in selected]
                row[endpoint]=float(np.mean(values)) if values and all(v is not None for v in values) else None
                row[endpoint+'_defined_N']=sum(v is not None for v in values)
            table.append(row)
    components=read(root/'reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json')['component_by_protein_id']
    bycase={(r['protein_id'],r['profile_name']):r for r in cases};names=sorted({r['profile_name'] for r in cases})
    contrasts=[]
    for cohort in ['binding_interface','enzyme']:
        ids=sorted({r['protein_id'] for r in cases if r['cohort']==cohort})
        wanted=['retained_ca_lddt','contact_recovery']+(['interface_contact_recovery'] if cohort=='binding_interface' else ['catalytic_neighbor_contact_recovery'])
        for endpoint in wanted:
            pairs=[]
            for pid in ids:
                a,b=bycase[pid,names[1]][endpoint],bycase[pid,names[0]][endpoint]
                pairs.append(dict(protein_id=pid,difference_lower=a-b if a is not None and b is not None else None,
                                  difference_upper=a-b if a is not None and b is not None else None))
            contrasts.append(dict(cohort=cohort,comparison=names[1]+' minus '+names[0],endpoint=endpoint,
                **paired_summary(pairs,components,20261018)))
    write(out/'prediction_integrity.json',integrity);write(out/'per_seed_WT_metrics.json',metrics)
    write(out/'per_case_seed_means.json',cases);write(out/'cohort_profile_table.json',table)
    write(out/'paired_profile_contrasts.json',contrasts)
    with (out/'cohort_profile_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report=dict(status='ALL_WT_ALL_SEED_CALIBRATION_COMPLETE' if all(v=='SUCCEEDED' for v in states.values()) else 'CALIBRATION_MISSING_OR_FAILED_CALLS_RETAINED',
        logical=len(metrics),physical=len(physical),physical_status=dict(Counter(states.values())),
        recorded_prediction_seconds=sum(r['seconds'] or 0 for r in integrity),
        profile_not_automatically_selected=True,variant_results_used=False,seeds_not_counted_as_independent_WT=True,
        frozen_construct_context_not_extended_with_unverified_cofactors=True,
        files={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.suffix in ('.json','.csv')},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
