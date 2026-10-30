"""Integrity first, then fixed-correspondence summaries for the development 2x2."""
import argparse
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_benign_metrics_20261017 import geometry
from recomb_benign_structure_evaluate_20261017 import paired_summary


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    out=root/'reports/recomb_confirm_v2_span_evaluation'
    if (out/'completion.json').exists():raise FileExistsError('Completed diagnostic immutable')
    cat=root/'reports/recomb_confirm_v2_span_catalog';struct=root/'reports/recomb_confirm_v2_span_structures'
    freeze=root/'configs/recomb_confirm_v2_span_prediction_freeze.json'
    for f in read(freeze)['files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('Frozen diagnostic source/input changed')
    logical=read(cat/'logical_requests.json');requests={p['task_id']:p for p in read(cat/'physical_requests.json')}
    cache=read(cat/'compatible_cache.json');arrays={};audit=[]
    for key,p in requests.items():
        if key in cache:
            record=cache[key];path=Path(record['artifact_path']);expected=record['artifact_sha256'];status='VERIFIED_CACHE'
        else:
            record=read(struct/'records'/(key+'.json'))
            if record['status']!='SUCCEEDED':raise ValueError('Diagnostic prediction missing/failed')
            for field in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']:
                if record[field]!=p[field]:raise ValueError('Prediction identity differs')
            path=Path(record['output_path']);expected=record['output_sha256'];status='SUCCEEDED'
        if sha(path)!=expected:raise ValueError('Prediction bytes differ')
        with np.load(path,allow_pickle=False) as z:
            if str(z['sequence_sha256'])!=p['sequence_sha256']:raise ValueError('Prediction sequence SHA differs')
            arrays[key]=z['coordinates']
        audit.append(dict(physical_id=key,status=status,artifact_path=str(path),artifact_sha256=expected))
    panel={r['protein_id']:r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    metric=[];ids=set()
    for r in logical:
        p=panel[r['protein_id']];native=np.full((len(p['sequence']),3),np.nan)
        for i,v in read(p['wt_native_json'])['coordinates'].items():native[int(i)]=v
        metric.append(dict(r,geometry=geometry(native,arrays[r['physical_id']],r['correspondence'],
                           root/'provenance/usalign_20261017/USalign')))
        ids.add(r['protein_id'])
    inherited=[r for r in read(root/'reports/recomb_benign_structure_evaluation_20261017/per_output_metrics.json')
        if r['protein_id'] in ids and r['profile_name']=='esmfold' and r['budget']==.2
        and r['method'] in ('delepref_unlimited','rag_unlimited')]
    if len(metric)!=40 or len(inherited)!=40:raise ValueError('Missing 2x2 arm')
    metric+=inherited;lookup={(r['protein_id'],r['method']):r for r in metric}
    protocol=read(root/'reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json')
    contrasts=[]
    for a,b in [('rag_q3_diagnostic','delepref_q3_diagnostic'),('rag_unlimited','delepref_unlimited'),
                ('delepref_q3_diagnostic','delepref_unlimited'),('rag_q3_diagnostic','rag_unlimited')]:
        for endpoint in ['retained_ca_lddt','contact_recovery']:
            pairs=[dict(protein_id=pid,difference_lower=lookup[pid,a]['geometry'][endpoint+'_lower']-lookup[pid,b]['geometry'][endpoint+'_upper'],
                        difference_upper=lookup[pid,a]['geometry'][endpoint+'_upper']-lookup[pid,b]['geometry'][endpoint+'_lower']) for pid in sorted(ids)]
            contrasts.append(dict(comparison=a+' minus '+b,endpoint=endpoint,
                **paired_summary(pairs,protocol['component_by_protein_id'],20261017)))
    table=[]
    for method in ['delepref_q3_diagnostic','rag_q3_diagnostic','delepref_unlimited','rag_unlimited']:
        selected=[r for r in metric if r['method']==method]
        table.append(dict(method=method,N=len(selected),
            retained_CA_lddt=float(np.mean([r['geometry']['retained_ca_lddt_lower'] for r in selected])),
            native_contact_recovery=float(np.mean([r['geometry']['contact_recovery_lower'] for r in selected])),
            full_WT_TM_lower=float(np.mean([r['geometry']['full_WT_TM']['lower'] for r in selected])),
            full_WT_TM_upper=float(np.mean([r['geometry']['full_WT_TM']['upper'] for r in selected]))))
    write(out/'prediction_integrity.json',audit);write(out/'per_output_metrics.json',metric)
    write(out/'paired_contrasts.json',contrasts);write(out/'four_arm_table.json',table)
    write(out/'completion.json',dict(status='DEVELOPMENT_CAP_ONLY_2X2_COMPLETE',WT=20,logical_outputs=80,
        verified_new_predictions=sum(r['status']=='SUCCEEDED' for r in audit),verified_cache=len(cache),
        main_task_still_unlimited=True,no_model_selection_by_diagnostic=True,
        files={p.name:sha(p) for p in out.glob('*.json')},job_id=os.environ['SLURM_JOB_ID']))
    print(table,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
