"""Read the entire predefined General primary cohort using frozen metrics/statistics."""
import os
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_general_evaluate import metric_pair,effect

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
CAT=ROOT/'reports/recomb_confirm_v2_General100_catalog'
STRUCT=ROOT/'reports/recomb_confirm_v2_General100_structures'
OUT=ROOT/'reports/recomb_confirm_v2_General100_primary_readout'


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if (OUT/'completion.json').exists():raise FileExistsError('Primary readout already complete')
    frozen=ROOT/'configs/recomb_confirm_v2_General100_confirmation_freeze.json'
    for p in read(frozen)['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen metric/statistics/predictor/input changed')
    logical=[r for r in read(CAT/'logical_requests.json') if r['predictor']=='esmfold' and r['evaluation_role'] in ['primary20pct','WT_reference']]
    if len(logical)!=700:raise ValueError('Primary full-cohort roster changed')
    physical={p['task_id']:p for p in read(CAT/'physical_requests.json')};reuse=read(CAT/'compatible_cache.json')
    panel={r['protein_id']:r for r in read(ROOT/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    protocol=read(CAT/'statistical_protocol.json');artifacts={};statuses={};audit=[]
    for task in {r['physical_id'] for r in logical}:
        spec=physical[task]
        if task in reuse:
            old=reuse[task];path=Path(old['artifact_path']);expected=old['artifact_sha256'];status='VERIFIED_CACHE'
        else:
            p=STRUCT/'records'/(task+'.json');record=read(p) if p.exists() else {}
            status=record.get('status','NOT_PRODUCED');path=Path(record['output_path']) if status=='SUCCEEDED' else None;expected=record.get('output_sha256')
            if status=='SUCCEEDED' and any(record.get(k)!=spec[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):raise ValueError('Primary prediction identity mismatch')
        if path:
            if sha(path)!=expected:raise ValueError('Primary prediction artifact changed')
            with np.load(path,allow_pickle=False) as z:
                if str(z['sequence'])!=spec['sequence'] or z['coordinates'].shape!=(len(spec['sequence']),3):raise ValueError('Primary predicted sequence/shape changed')
        artifacts[task]=str(path) if path else None;statuses[task]=status
        audit.append(dict(physical_id=task,status=status,artifact_path=str(path) if path else None,artifact_sha256=expected))
    jobs={}
    for r in logical:
        key=(r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))
        jobs.setdefault(key,(str(ROOT),key,panel[r['protein_id']]['wt_native_json'],artifacts[r['physical_id']],len(r['edited_sequence']),r['correspondence']))
    with ProcessPoolExecutor(max_workers=8) as executor:values=dict(executor.map(metric_pair,jobs.values()))
    rows=[dict(r,prediction_status=statuses[r['physical_id']],**values[(r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))]) for r in logical]
    endpoints=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM'];table=[];comparisons=[];differences=[]
    for method in protocol['methods']+['WT']:
        group=[r for r in rows if r['method']==method]
        if len(group)!=100:raise ValueError('Primary method/WT does not retain all100 inputs')
        entry=dict(method=method,N=100)
        for endpoint in endpoints:
            vals=[r['common_structure'].get(endpoint) for r in group]
            entry[endpoint]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            entry[endpoint+'_defined_N']=sum(v is not None for v in vals)
        table.append(entry)
    by={(r['protein_id'],r['method']):r for r in rows}
    for method in protocol['methods']:
        if method=='delepref_unlimited':continue
        for endpoint in endpoints:
            pairs=[]
            for pid in panel:
                a=by[pid,method]['common_structure'].get(endpoint);b=by[pid,'delepref_unlimited']['common_structure'].get(endpoint)
                d=a-b if a is not None and b is not None else None;pairs.append((pid,d))
                differences.append(dict(protein_id=pid,method=method,endpoint=endpoint,difference=d))
            comparisons.append(dict(method=method,versus='delepref_unlimited',endpoint=endpoint,**effect(pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    write(OUT/'per_output_metrics.json',rows);write(OUT/'primary_table.json',table)
    write(OUT/'paired_comparisons.json',comparisons);write(OUT/'per_protein_differences.json',differences)
    write(OUT/'prediction_identity_audit.json',audit)
    done=dict(status='FULL_PREDEFINED_GENERAL100_PRIMARY_STRUCTURAL_READOUT_COMPLETE',logical=700,physical=len(artifacts),
        missing_or_failed=sum(v not in ('SUCCEEDED','VERIFIED_CACHE') for v in statuses.values()),
        new_predictions=0,all100_inputs_and_failures_preserved=True,no_new_selector_or_subset=True,
        reader_added_after_dispatch_but_metrics_and_endpoints_frozen_before_predictions=True,
        metric_source_sha256=sha(ROOT/'analysis/recomb_confirm_v2_general_evaluate.py'),reader_source_sha256=sha(Path(__file__)),
        confirmation_freeze_sha256=sha(frozen),not_functional_validation=True,job_id=os.environ['SLURM_JOB_ID'],
        outputs={p:sha(OUT/p) for p in ['per_output_metrics.json','primary_table.json','paired_comparisons.json','per_protein_differences.json','prediction_identity_audit.json']})
    write(OUT/'completion.json',done);print(done,flush=True)


if __name__=='__main__':main()
