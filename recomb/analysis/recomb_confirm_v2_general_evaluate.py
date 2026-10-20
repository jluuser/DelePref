"""Evaluate every frozen General prediction; preserve failures and both correspondence layers."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_benign_metrics_20261017 import geometry
from recomb_confirm_v2_structure_alignment import aligned_geometry

CAT=Path('reports/recomb_confirm_v2_General100_catalog')
STRUCT=Path('reports/recomb_confirm_v2_General100_structures')
OUT=Path('reports/recomb_confirm_v2_General100_evaluation')
CONFIRM=Path('configs/recomb_confirm_v2_General100_confirmation_freeze.json')


def metric_pair(item):
    root_s,key,native_path,artifact,pred_length,correspondence=item;root=Path(root_s)
    source=read(native_path);ref=np.full((len(correspondence['wt_to_edit']),3),np.nan)
    for i,xyz in source['coordinates'].items():ref[int(i)]=xyz
    if artifact:
        with np.load(artifact,allow_pickle=False) as z:pred=np.asarray(z['coordinates'],dtype=float)
    else:pred=np.full((pred_length,3),np.nan)
    binary=root/'provenance/usalign_20261017/USalign'
    return key,dict(common_structure=aligned_geometry(ref,pred,binary),fixed_identity_geometry=geometry(ref,pred,correspondence,binary))


def effect(pairs,components,seed=20261016,replicates=10000):
    if not pairs:return dict(status='EMPTY',N=0)
    if any(v is None or not np.isfinite(v) for _,v in pairs):
        return dict(status='UNDEFINED_OUTCOME_PRESENT_NO_SILENT_EXCLUSION',N=len(pairs),N_defined=sum(v is not None and np.isfinite(v) for _,v in pairs))
    values=np.asarray([v for _,v in pairs],dtype=float);labels=[components[pid] for pid,_ in pairs];unique=sorted(set(labels))
    sums=np.array([values[np.array(labels)==label].sum() for label in unique]);counts=np.array([labels.count(label) for label in unique])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(unique),size=(replicates,len(unique)))
    boot=sums[indices].sum(1)/counts[indices].sum(1);tol=1e-8
    loco=[float((values.sum()-sums[i])/(len(values)-counts[i])) for i in range(len(unique)) if len(values)>counts[i]]
    return dict(status='COMPONENT_PAIRED_BOOTSTRAP',N=len(values),components=len(unique),mean_difference=float(values.mean()),
        median_difference=float(np.median(values)),CI95=np.quantile(boot,[.025,.975]).tolist(),
        joint_Bonferroni_CI97_5=np.quantile(boot,[.0125,.9875]).tolist(),
        improved=int((values>tol).sum()),degraded=int((values< -tol).sum()),unchanged=int((np.abs(values)<=tol).sum()),
        LOCO_mean_range=[min(loco),max(loco)] if loco else None,bootstrap_seed=seed,replicates=replicates)


def evaluate(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/CONFIRM)
    for p in f['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen prediction/statistic input changed')
    out=root/OUT
    if (out/'completion.json').exists():raise FileExistsError('General evaluation already completed')
    logical=read(root/CAT/'logical_requests.json');physical={p['task_id']:p for p in read(root/CAT/'physical_requests.json')}
    reuse=read(root/CAT/'compatible_cache.json');protocol=read(root/CAT/'statistical_protocol.json')
    panel={p['protein_id']:p for p in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    artifacts={};statuses={};identity_audit=[]
    for key,p in physical.items():
        if key in reuse:
            info=reuse[key];path=Path(info['artifact_path']);expected=info['artifact_sha256'];status='VERIFIED_CACHE'
            if sha(info['record_path'])!=info['record_sha256']:raise ValueError('Verified cache record changed')
        else:
            record=root/STRUCT/'records'/(key+'.json');r=read(record) if record.exists() else {}
            status=r.get('status','NOT_PRODUCED');path=Path(r['output_path']) if status=='SUCCEEDED' else None;expected=r.get('output_sha256')
            if status=='SUCCEEDED' and any(r.get(k)!=p[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):raise ValueError('Prediction identity mismatch')
        if path:
            if sha(path)!=expected:raise ValueError('Prediction artifact SHA changed')
            with np.load(path,allow_pickle=False) as z:
                if str(z['sequence'])!=p['sequence'] or z['coordinates'].shape!=(len(p['sequence']),3):raise ValueError('Prediction sequence/coordinates changed')
        artifacts[key]=str(path) if path else None;statuses[key]=status
        identity_audit.append(dict(physical_id=key,status=status,artifact_path=str(path) if path else None,artifact_sha256=expected))
    jobs={}
    for r in logical:
        key=(r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))
        jobs.setdefault(key,(str(root),key,panel[r['protein_id']]['wt_native_json'],artifacts[r['physical_id']],len(r['edited_sequence']),r['correspondence']))
    with ProcessPoolExecutor(max_workers=8) as executor:metrics=dict(executor.map(metric_pair,jobs.values()))
    rows=[]
    for r in logical:
        key=(r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))
        rows.append(dict(r,prediction_status=statuses[r['physical_id']],**metrics[key]))
    if len(rows)!=2494:raise ValueError('Lost frozen General logical predictions')
    endpoints=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM']
    main=[r for r in rows if r['predictor']=='esmfold' and r['evaluation_role']=='primary20pct'];table=[];comparisons=[]
    for method in protocol['methods']:
        selected=[r for r in main if r['method']==method];entry=dict(method=method,N_all=len(selected))
        for endpoint in endpoints:
            vals=[r['common_structure'].get(endpoint) for r in selected]
            entry[endpoint]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            entry[endpoint+'_defined_N']=sum(v is not None for v in vals)
        table.append(entry)
    for predictor,role in [('esmfold','primary20pct'),('boltz2','second_predictor')]:
        selected=[r for r in rows if r['predictor']==predictor and r['evaluation_role']==role and r['method']!='WT']
        # All predefined seeds are first averaged within WT/method. A missing seed leaves the WT undefined.
        grouped=defaultdict(list)
        for r in selected:grouped[(r['protein_id'],r['method'])].append(r)
        for method in protocol['methods']:
            if method=='delepref_unlimited':continue
            for endpoint in endpoints:
                pairs=[]
                for (pid,m),values in grouped.items():
                    if m!=method:continue
                    base=grouped[(pid,'delepref_unlimited')]
                    a=[r['common_structure'].get(endpoint) for r in values];b=[r['common_structure'].get(endpoint) for r in base]
                    expected=3 if predictor=='boltz2' else 1
                    delta=float(np.mean(a)-np.mean(b)) if len(a)==expected and len(b)==expected and all(v is not None for v in a+b) else None
                    pairs.append((pid,delta))
                comparisons.append(dict(predictor=predictor,method=method,versus='delepref_unlimited',endpoint=endpoint,
                    **effect(pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    primary=[x for x in comparisons if x['predictor']=='esmfold' and x['method']=='rag_unlimited' and x['endpoint'] in protocol['co_primary_endpoints']]
    joint_supported=len(primary)==2 and all(x.get('joint_Bonferroni_CI97_5',[None])[0] is not None and x['joint_Bonferroni_CI97_5'][0]>0 for x in primary)
    for name,value in [('per_output_metrics.json',rows),('primary_table.json',table),('primary_and_second_predictor_comparisons.json',comparisons),('prediction_identity_audit.json',identity_audit)]:write(out/name,value)
    done=dict(status='GENERAL_FULL_ROSTER_METRICS_AND_MAIN_SECOND_PREDICTOR_STATISTICS_COMPLETE',
        logical_outputs=len(rows),physical_predictions=len(physical),missing_or_failed=sum(v not in ('SUCCEEDED','VERIFIED_CACHE') for v in statuses.values()),
        common_shape_dual_endpoint_joint_support=joint_supported,not_functional_validation=True,
        no_model_candidate_or_selector_reselection=True,additional_predeclared_budget_mechanism_and_pool_summaries_pending=True,
        outputs={p:sha(out/p) for p in ['per_output_metrics.json','primary_table.json','primary_and_second_predictor_comparisons.json','prediction_identity_audit.json']})
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();evaluate(a.root)
