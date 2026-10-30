"""Post-hoc common structural comparison of the complete benign development round."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_structure_alignment import aligned_geometry


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm required')
    start=time.monotonic();out=root/'reports/recomb_confirm_v2_alignment'
    if (out/'completion.json').exists():raise FileExistsError('Evaluation already completed')
    frozen=read(root/'configs/recomb_confirm_v2_alignment_freeze.json')
    for name,h in frozen['files'].items():
        if sha(root/name)!=h:raise ValueError('Frozen input changed: '+name)
    binary=root/'provenance/usalign_20261017/USalign'
    if sha(binary)!=frozen['USalign_sha256']:raise ValueError('USalign identity changed')
    g={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    b={p['protein_id']:p for p in read(root/'reports/recomb_benign_binding_native_20261017/native_references.json')}
    e={p['protein_id']:p for p in read(root/'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json')}
    approved=set(g)|set(b)|set(e)
    source=root/'reports/recomb_benign_structure_evaluation_20261017'
    physical={p['physical_id']:p for p in read(source/'prediction_identity_audit.json')}
    rows=read(source/'per_output_metrics.json');cache={};results=[]
    for i,row in enumerate(rows):
        pid=row['protein_id']
        if pid not in approved:raise ValueError('Input outside positively reviewed scope')
        p=physical[row['physical_id']]
        if p['artifact_path'] is None:raise ValueError('Missing prediction remains unhandled')
        artifact=Path(p['artifact_path'])
        if sha(artifact)!=p['artifact_sha256']:raise ValueError('Prediction artifact changed')
        if pid in g:
            spec=g[pid];native=np.full((len(spec['sequence']),3),np.nan)
            for j,xyz in read(spec['wt_native_json'])['coordinates'].items():native[int(j)]=xyz
        else:
            values=b[pid]['chains'][0]['native_CA'] if pid in b else e[pid]['native_CA']
            native=np.array([xyz if xyz is not None else [np.nan]*3 for xyz in values],dtype=float)
        key=pid,row['physical_id']
        if key not in cache:
            with np.load(artifact,allow_pickle=False) as data:
                prediction=data['coordinates']
            cache[key]=aligned_geometry(native,prediction,binary)
        value={k:row[k] for k in ('protein_id','cohort','method','budget','evaluation_role','profile_name','physical_id')}
        value['common_structure']=cache[key];results.append(value)
        if (i+1)%50==0:
            print('COMMON_ALIGNMENT_COMPLETE '+str(i+1)+'/'+str(len(rows)),flush=True)
    if len(results)!=867:raise ValueError('Lost logical records')
    groups=defaultdict(list)
    for row in results:
        expected='esmfold' if row['cohort']=='general' else ('boltz_binding' if row['cohort']=='binding_interface' else 'boltz_enzyme')
        if row['budget']==.2 and row['profile_name']==expected and row['evaluation_role']=='comparison':
            groups[row['cohort'],row['method']].append(row)
    table=[]
    for (cohort,method),selected in sorted(groups.items()):
        entry=dict(cohort=cohort,method=method,N=len(selected))
        for endpoint in ('full_WT_TM','aligned_full_WT_coverage','structure_mapped_CA_lddt',
                         'structure_mapped_native_contact_recovery'):
            values=[r['common_structure'].get(endpoint) for r in selected]
            entry[endpoint+'_defined_N']=sum(v is not None for v in values)
            entry[endpoint]=float(np.mean(values)) if all(v is not None for v in values) else None
        table.append(entry)
    write(out/'per_output_common_structure.json',results);write(out/'primary_common_structure_table.json',table)
    write(out/'completion.json',dict(status='DEVELOPMENT_COMMON_STRUCTURE_EVALUATION_COMPLETE',
        logical_outputs=len(results),unique_native_prediction_pairs=len(cache),all_six_methods_same_protocol=True,
        post_hoc_development=True,not_residue_identity_or_function_validation=True,
        new_structure_predictions=0,seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],
        files={p.name:sha(p) for p in out.glob('*.json')}))
    print('COMMON_ALIGNMENT_ALL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
