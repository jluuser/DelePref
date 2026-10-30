"""Numerical consistency, native identity controls and independent CA parsing."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_metrics_20261017 import geometry


def array(values):
    xyz=np.full((len(values),3),np.nan)
    for i,p in enumerate(values):
        if p is not None:xyz[i]=p
    return xyz


def audit(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    sys.path.insert(0,str(root/'code'))
    from adapters.structure_adapter import read_coordinates
    out=root/'reports/recomb_benign_final_quality_20261017'
    if out.exists():raise FileExistsError('final quality audit exists')
    full=root/'reports/recomb_benign_structure_evaluation_20261017'
    final=read(full/'completion.json')
    for n,h in final['outputs'].items():
        if sha(full/n)!=h:raise ValueError('full evaluation changed')
    values=read(full/'per_output_metrics.json');by_id={r['request_id']:r for r in values}
    keys=['retained_ca_lddt_lower','retained_ca_lddt_upper','contact_recovery_lower','contact_recovery_upper','T_lower','T_upper']
    checked={}
    for folder in ['recomb_benign_primary_readout_20261017','recomb_benign_general_readout_20261017']:
        previous=read(root/'reports'/folder/'per_output_metrics.json')
        for r in previous:
            other=by_id[r['request_id']]
            for k in keys:
                a,b=r['geometry'][k],other['geometry'][k]
                if a is None or b is None:
                    if a is not b:raise ValueError('undefined metric mismatch')
                elif abs(a-b)>1e-11:raise ValueError('early/full metric mismatch')
            for k in ['lower','upper']:
                if abs(r['geometry']['full_WT_TM'][k]-other['geometry']['full_WT_TM'][k])>1e-11:raise ValueError('TM readout mismatch')
        checked[folder]=len(previous)
    refs=[]
    for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json'):
        native=read(Path(p['wt_native_json']));xyz=np.full((len(p['sequence']),3),np.nan)
        for i,v in native['coordinates'].items():xyz[int(i)]=v
        refs.append((p['protein_id'],xyz))
    for p in read(root/'reports/recomb_benign_binding_native_20261017/native_references.json'):refs.append((p['protein_id'],array(p['chains'][0]['native_CA'])))
    for p in read(root/'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json'):refs.append((p['protein_id'],array(p['native_CA'])))
    native_checks=[]
    for pid,xyz in refs:
        mp=dict(wt_to_edit=list(range(len(xyz))),position_states=['stable']*len(xyz))
        score=geometry(xyz,xyz,mp)
        for k in ['retained_ca_lddt_lower','contact_recovery_lower','T_lower']:
            if abs(score[k]-1)>1e-12:raise ValueError('native self-identity metric failed')
        native_checks.append(dict(protein_id=pid,known_CA=int(np.isfinite(xyz).all(1).sum()),native_self_identity_pass=True))
    struct=root/'reports/recomb_benign_structures_20261017';parsed=[]
    for path in (struct/'records').glob('*.json'):
        r=read(path)
        if r['predictor']!='boltz2' or r['status']!='SUCCEEDED':continue
        raw=Path(r['raw_structure_path'])
        if sha(raw)!=r['raw_structure_sha256']:raise ValueError('raw structure differs')
        independent,_=read_coordinates(raw,r['chains'][0]['sequence'],chain_id='A')
        with np.load(r['output_path'],allow_pickle=False) as z:
            ca=z['coordinates']
            if not np.allclose(ca,independent,atol=1e-6,equal_nan=True):raise ValueError('independent CA parser mismatch')
        parsed.append(dict(physical_id=r['task_id'],same_CA_as_independent_parser=True))
    if len(parsed)!=137:raise ValueError('not all new Boltz outputs audited')
    wt=defaultdict(list)
    for r in values:
        if r['method']=='WT':wt[(r['cohort'],r['profile_name'])].append(r)
    calibration=[]
    for (cohort,profile),rows in sorted(wt.items()):
        calibration.append(dict(cohort=cohort,profile_name=profile,N=len(rows),
            WT_retained_CA_lddt=float(np.mean([r['geometry']['retained_ca_lddt_lower'] for r in rows])),
            WT_native_contact_recovery=float(np.mean([r['geometry']['contact_recovery_lower'] for r in rows])),
            WT_full_TM_lower=float(np.mean([r['geometry']['full_WT_TM']['lower'] for r in rows])),
            WT_full_TM_upper=float(np.mean([r['geometry']['full_WT_TM']['upper'] for r in rows]))))
    out.mkdir();write(out/'native_self_checks.json',native_checks);write(out/'independent_Boltz_CA_checks.json',parsed)
    write(out/'WT_calibration.json',calibration)
    report=dict(status='FINAL_NUMERICAL_AND_COORDINATE_AUDIT_PASS_WITH_PREDICTOR_CALIBRATION_LIMITATIONS',
        early_and_full_numeric_results_match=checked,native_self_checks=len(native_checks),independent_Boltz_CA_parses=137,
        no_coordinate_parser_mismatch=True,WT_calibration_requires_report=True,
        limitations='Empty-MSA frozen Boltz profiles have weak WT structural agreement for enzymes and general8; paired signs are limited consistency evidence, not absolute quality or function validation',
        source_script_sha256=sha(Path(__file__)),full_evaluation_completion_sha256=sha(full/'completion.json'),new_GPU_calls=0,
        outputs={p.name:sha(p) for p in out.glob('*.json')},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    audit(p.parse_args().root)
