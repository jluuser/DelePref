"""Read the complete frozen primary44 panel while other fixed panels keep running."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_metrics_20261017 import geometry
from recomb_benign_structure_evaluate_20261017 import paired_summary

BASE=Path('reports/recomb_benign_primary_readout_20261017')


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    sys.path.insert(0,str(root/'code'))
    from prediction_journal import PredictionJournal
    cat=root/'reports/recomb_benign_prediction_catalog_20261017';struct=root/'reports/recomb_benign_structures_20261017'
    freeze=root/'configs/recomb_benign_confirmation_freeze_20261017.json'
    expected='7efd8b3589ffb625db933efacbc8142e2b048f08a585fce1b562d8d8cde9297b'
    if sha(freeze)!=expected:raise ValueError('frozen confirmation changed')
    for f in read(freeze)['files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('frozen source/input changed')
    out=root/BASE
    if out.exists():raise FileExistsError('primary readout already saved')
    binary=root/'provenance/usalign_20261017/USalign'
    if sha(binary)!='3b3e5bf0e10357447c1d66efe80494020cb0a3c768b04083968967eb7bf07c00':raise ValueError('TM implementation changed')
    for shard,n in [(0,71),(1,71),(2,70),(3,70)]:
        s=read(struct/('esmfold_shard_'+str(shard)+'_completion.json'))
        if s['physical_requests']!=n:raise ValueError('ESMFold primary backbone not fully complete')
    methods={'delepref_unlimited','rag_unlimited','terminal','scisor_native','raygun_native'}
    logical=[r for r in read(cat/'logical_requests.json') if r['profile_name']=='esmfold' and
        ((r['budget']==.2 and r['method'] in methods and r['evaluation_role']=='comparison') or r['method']=='WT')]
    if len(logical)!=264:raise ValueError('not the full primary44 panel plus WT')
    panel={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    for method in methods|{'WT'}:
        if {r['protein_id'] for r in logical if r['method']==method}!=set(panel):raise ValueError('method primary roster differs')
    reuse=read(cat/'compatible_cache.json');physical={p['task_id']:p for p in read(cat/'physical_requests.json')}
    snapshot=PredictionJournal(cat/'statistical_protocol.json',struct/'prediction_journal.jsonl').snapshot()
    journal={p['task_id']:p for p in snapshot['attempts']}
    predictions={};checks=[]
    for task in sorted({r['physical_id'] for r in logical}):
        request=physical[task]
        if task in reuse:
            item=reuse[task];path=Path(item['artifact_path']);digest=item['artifact_sha256'];state='VERIFIED_CACHE'
        else:
            item=read(struct/'records'/(task+'.json'));j=journal[task]
            if j['state']!='succeeded' or item['status']!='SUCCEEDED' or j['output_sha']!=item['output_sha256']:raise ValueError('primary prediction not successful/verified')
            if any(item[k]!=request[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):raise ValueError('primary prediction identity differs')
            path=Path(item['output_path']);digest=item['output_sha256'];state='NEW_SUCCEEDED'
        if sha(path)!=digest:raise ValueError('primary artifact SHA differs')
        with np.load(path,allow_pickle=False) as z:
            if str(z['sequence'])!=request['sequence'] or str(z['sequence_sha256'])!=request['sequence_sha256']:raise ValueError('primary output sequence differs')
            predictions[task]=z['coordinates'].copy()
        checks.append(dict(physical_id=task,state=state,artifact_path=str(path),artifact_sha256=digest))
    values=[]
    for row in logical:
        p=panel[row['protein_id']];native=read(Path(p['wt_native_json']));ref=np.full((len(p['sequence']),3),np.nan)
        for i,xyz in native['coordinates'].items():ref[int(i)]=xyz
        metric=geometry(ref,predictions[row['physical_id']],row['correspondence'],binary)
        values.append(dict(row,stratum=p['stratum'],geometry=metric))
    def interval(r,name):
        g=r['geometry']
        return (g['full_WT_TM']['lower'],g['full_WT_TM']['upper']) if name=='full_WT_TM' else (g[name+'_lower'],g[name+'_upper'])
    endpoints=['retained_ca_lddt','contact_recovery','full_WT_TM']
    baseline={r['protein_id']:r for r in values if r['method']=='delepref_unlimited'}
    protocol=read(cat/'statistical_protocol.json');paired=[];summary=[];absolute=[]
    for method in sorted(methods|{'WT'}):
        rows=[r for r in values if r['method']==method]
        for endpoint in endpoints:
            b=[interval(r,endpoint) for r in rows]
            absolute.append(dict(method=method,endpoint=endpoint,N=44,
                macro_lower=float(np.mean([x[0] for x in b])) if all(x[0] is not None for x in b) else None,
                macro_upper=float(np.mean([x[1] for x in b])) if all(x[1] is not None for x in b) else None))
            if method in ('WT','delepref_unlimited'):continue
            pairs=[]
            for r in rows:
                a,b=interval(r,endpoint);c,d=interval(baseline[r['protein_id']],endpoint)
                pairs.append(dict(protein_id=r['protein_id'],method=method,endpoint=endpoint,
                    difference_lower=a-d if a is not None and d is not None else None,
                    difference_upper=b-c if b is not None and c is not None else None))
            paired+=pairs;summary.append(dict(method=method,endpoint=endpoint,
                **paired_summary(pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    evidence={r['protein_id']:r for r in read(root/'reports/recomb_rag_decision_audit_20261017/per_condition.json') if r['cohort']=='general' and r['budget']==.2 and r['generation_seed']==13}
    rag_cases=[]
    for pid in sorted(panel):
        row=dict(evidence[pid]);row['differences']={p['endpoint']:dict(lower=p['difference_lower'],upper=p['difference_upper']) for p in paired if p['protein_id']==pid and p['method']=='rag_unlimited'}
        rag_cases.append(row)
    out.mkdir();write(out/'per_output_metrics.json',values);write(out/'absolute_metrics.json',absolute)
    write(out/'paired_comparisons.json',summary);write(out/'per_protein_differences.json',paired)
    write(out/'RAG_coverage_and_differences.json',rag_cases);write(out/'prediction_integrity.json',checks)
    report=dict(status='COMPLETE_FROZEN_PRIMARY44_READOUT_OTHER_EXPERIMENTS_STILL_RUNNING',WT=44,logical_outputs=264,
        physical_artifacts_verified=len(checks),all_methods_same44_WT=True,all_selected_outputs_prespecified=True,
        no_model_or_selector_or_sample_changes=True,not_functional_validation=True,not_full_experiment_completion=True,
        freeze_sha256=expected,bootstrap_seed=protocol['bootstrap_seed'],bootstrap_replicates=10000,
        source_sha256=sha(Path(__file__)),job_id=os.environ['SLURM_JOB_ID'],
        outputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'completion.json',report)
    print(json.dumps(dict(status=report['status'],RAG=[r for r in summary if r['method']=='rag_unlimited'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    run(p.parse_args().root)
