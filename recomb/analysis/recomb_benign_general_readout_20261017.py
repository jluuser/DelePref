"""Complete only the remaining frozen general-panel metrics while Boltz runs."""
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


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    sys.path.insert(0,str(root/'code'))
    from prediction_journal import PredictionJournal
    cat=root/'reports/recomb_benign_prediction_catalog_20261017';struct=root/'reports/recomb_benign_structures_20261017'
    out=root/'reports/recomb_benign_general_readout_20261017'
    if out.exists():raise FileExistsError('general readout already saved')
    primary=root/'reports/recomb_benign_primary_readout_20261017';pd=read(primary/'completion.json')
    for n,h in pd['outputs'].items():
        if sha(primary/n)!=h:raise ValueError('primary readout changed')
    freeze=root/'configs/recomb_benign_confirmation_freeze_20261017.json'
    if sha(freeze)!=pd['freeze_sha256']:raise ValueError('experiment freeze differs')
    for f in read(freeze)['files']:
        if f['path'].startswith(str(root)) and sha(f['path'])!=f['sha256']:raise ValueError('frozen scientific file changed')
    binary=root/'provenance/usalign_20261017/USalign'
    if sha(binary)!=read(root/'configs/recomb_benign_structure_dispatch_20261017.json')['metric_binary']['sha256']:raise ValueError('TM binary changed')
    values=read(primary/'per_output_metrics.json');already={v['request_id'] for v in values}
    logical=[r for r in read(cat/'logical_requests.json') if r['profile_name']=='esmfold']
    todo=[r for r in logical if r['request_id'] not in already]
    if len(logical)!=704 or len(values)!=264 or len(todo)!=440:raise ValueError('general/primary delta differs')
    physical={p['task_id']:p for p in read(cat/'physical_requests.json')};reuse=read(cat/'compatible_cache.json')
    panel={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    journal={p['task_id']:p for p in PredictionJournal(cat/'statistical_protocol.json',struct/'prediction_journal.jsonl').snapshot()['attempts']}
    predictions={};audit=[]
    for task in sorted({r['physical_id'] for r in todo}):
        p=physical[task]
        if task in reuse:path=Path(reuse[task]['artifact_path']);digest=reuse[task]['artifact_sha256']
        else:
            record=read(struct/'records'/(task+'.json'));j=journal[task]
            if record['status']!='SUCCEEDED' or j['state']!='succeeded' or j['output_sha']!=record['output_sha256']:raise ValueError('general output not successful')
            if any(record[k]!=p[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):raise ValueError('prediction identity differs')
            path=Path(record['output_path']);digest=record['output_sha256']
        if sha(path)!=digest:raise ValueError('artifact changed')
        with np.load(path,allow_pickle=False) as z:
            if str(z['sequence'])!=p['sequence'] or str(z['sequence_sha256'])!=p['sequence_sha256']:raise ValueError('sequence changed')
            predictions[task]=z['coordinates'].copy()
        audit.append(dict(physical_id=task,artifact_path=str(path),artifact_sha256=digest))
    for row in todo:
        p=panel[row['protein_id']];native=read(Path(p['wt_native_json']));reference=np.full((len(p['sequence']),3),np.nan)
        for i,xyz in native['coordinates'].items():reference[int(i)]=xyz
        values.append(dict(row,stratum=p['stratum'],geometry=geometry(reference,predictions[row['physical_id']],row['correspondence'],binary)))
    protocol=read(cat/'statistical_protocol.json');sub=read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json')
    comparisons=[];differences=[];absolute=[]
    def bounds(r,e):
        return (r['geometry']['full_WT_TM']['lower'],r['geometry']['full_WT_TM']['upper']) if e=='full_WT_TM' else (r['geometry'][e+'_lower'],r['geometry'][e+'_upper'])
    experiments=dict(general_primary=[r for r in values if r['budget']==.2 and r['evaluation_role']=='comparison'],
        general_budget_curve=[r for r in values if r['protein_id'] in sub['completion_ablation'] and r['evaluation_role']=='comparison'],
        general_mechanisms=[r for r in values if r['protein_id'] in sub['completion_ablation'] and r['budget']==.2 and r['method']!='WT'])
    for experiment,rows in experiments.items():
        base={(r['protein_id'],r['budget']):r for r in rows if r['method']=='delepref_unlimited'}
        groups=defaultdict(list)
        for r in rows:groups[(r['method'],r['budget'])].append(r)
        for (method,budget),group in sorted(groups.items()):
            for endpoint in ['retained_ca_lddt','contact_recovery','full_WT_TM']:
                intervals=[bounds(r,endpoint) for r in group]
                absolute.append(dict(experiment=experiment,method=method,budget=budget,endpoint=endpoint,N=len(group),
                    macro_lower=float(np.mean([v[0] for v in intervals])) if all(v[0] is not None for v in intervals) else None,
                    macro_upper=float(np.mean([v[1] for v in intervals])) if all(v[1] is not None for v in intervals) else None))
                if method=='delepref_unlimited':continue
                pairs=[]
                for r in group:
                    lo,hi=bounds(r,endpoint);bl,bu=bounds(base[(r['protein_id'],budget)],endpoint)
                    pairs.append(dict(experiment=experiment,method=method,budget=budget,endpoint=endpoint,protein_id=r['protein_id'],
                        difference_lower=lo-bu if lo is not None and bu is not None else None,
                        difference_upper=hi-bl if hi is not None and bl is not None else None))
                differences+=pairs;comparisons.append(dict(experiment=experiment,method=method,budget=budget,endpoint=endpoint,
                    **paired_summary(pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    selected={(r['protein_id'],r['method']):r for r in values if r['budget']==.2}
    identity=[]
    for pid in sub['completion_ablation']:
        full=selected[(pid,'rag_unlimited')]
        for method in ['rag_direct_only','rag_no_scorer','rag_marginal_gap','rag_cooccurrence_null','rag_position_gap_frequency']:
            row=selected[(pid,method)]
            identity.append(dict(protein_id=pid,method=method,identical_to_full_RAG_output_and_correspondence=
                row['physical_id']==full['physical_id'] and row['correspondence']==full['correspondence']))
    out.mkdir();write(out/'per_output_metrics.json',values);write(out/'absolute_metrics.json',absolute)
    write(out/'paired_comparisons.json',comparisons);write(out/'per_protein_differences.json',differences)
    write(out/'intervention_output_identity.json',identity);write(out/'prediction_integrity_delta.json',audit)
    report=dict(status='COMPLETE_FROZEN_GENERAL704_READOUT_APPLICATIONS_AND_SECOND_PREDICTOR_PENDING',logical_outputs=704,
        reused_primary_metrics=264,new_logical_metrics=440,new_GPU_calls=0,primary_readout_completion_sha256=sha(primary/'completion.json'),
        freeze_sha256=pd['freeze_sha256'],source_script_sha256=sha(Path(__file__)),job_id=os.environ['SLURM_JOB_ID'],
        not_full_experiment_completion=True,outputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k!='outputs'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    run(p.parse_args().root)
