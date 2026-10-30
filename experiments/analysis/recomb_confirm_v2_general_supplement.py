"""Read-only E1b/E4/E5/E7 summaries using frozen General metric functions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_general_evaluate import metric_pair,effect

BASE=Path('reports/recomb_confirm_v2_General100_ESM_supplement')
FZ=Path('configs/recomb_confirm_v2_General100_ESM_supplement_freeze.json')
CAT=Path('reports/recomb_confirm_v2_General100_catalog')
PRIMARY=Path('reports/recomb_confirm_v2_General100_primary_readout')
STRUCT=Path('reports/recomb_confirm_v2_General100_structures')
ENDPOINTS=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM']


def pool_summary(values,selected):
    if len(values)!=8:raise ValueError('A complete K8 pool is required for pool best/regret')
    if selected is None or any(v is None or not np.isfinite(v) for v in values):
        return dict(status='UNDEFINED_CANDIDATE_OR_SELECTION_NO_SILENT_EXCLUSION',candidate_N=8)
    best=float(max(values));regret=best-float(selected)
    if regret < -1e-8:raise ValueError('Selected value exceeds the complete candidate pool')
    return dict(status='COMPLETE_K8_POOL_DESCRIPTIVE_ORACLE_NOT_A_SELECTION',candidate_N=8,
        minimum=float(min(values)),mean=float(np.mean(values)),median=float(np.median(values)),
        best=best,selected=float(selected),regret=max(0.,regret))


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('General supplementary reader already frozen')
    names=['analysis/recomb_confirm_v2_general_supplement.py','slurm/recomb_confirm_v2_general_supplement.sbatch',
        'analysis/tests/test_recomb_confirm_v2_pool_summary.py','analysis/recomb_confirm_v2_general_evaluate.py',
        'analysis/recomb_confirm_v2_diagnostics.py','configs/recomb_confirm_v2_General100_confirmation_freeze.json',
        str(CAT/'logical_requests.json'),str(CAT/'physical_requests.json'),str(CAT/'statistical_protocol.json'),
        str(PRIMARY/'completion.json'),str(PRIMARY/'per_output_metrics.json'),
        'reports/recomb_confirm_v2_General100_mechanisms/diagnostics.json',
        'reports/recomb_confirm_v2_General100_mechanisms/subsets.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json']
    write(root/FZ,dict(status='READ_ONLY_PREDECLARED_ESM_EXTENSION_READER_FROZEN_AFTER_PRIMARY_READOUT',
        files={p:sha(root/p) for p in names},endpoints=ENDPOINTS,
        analyses=['same40_WT_budget_curve_including20pct','all40_and_input_valid_mechanism_controls',
            'nested20_evidence_scarcity','all12_complete_K8_pool_best_and_scorer_regret','quota_control12'],
        effects='frozen component paired bootstrap10000 seed20261016, 95%CI; secondary/descriptive',
        missing_outcomes='retain undefined rows and denominators, no case removal',
        no_model_candidate_selector_or_subgroup_reselection=True,new_structure_predictions=0))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def evaluate(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    out=root/BASE
    if (out/'completion.json').exists():raise FileExistsError('ESM supplementary evaluation already completed')
    for p,h in read(root/FZ)['files'].items():
        if sha(root/p)!=h:raise ValueError('Supplement reader/input changed: '+p)
    for p in read(root/'configs/recomb_confirm_v2_General100_confirmation_freeze.json')['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Original frozen scientific input changed')
    done=read(root/PRIMARY/'completion.json')
    if sha(root/PRIMARY/'per_output_metrics.json')!=done['outputs']['per_output_metrics.json']:
        raise ValueError('Completed primary metrics changed')
    logical=[r for r in read(root/CAT/'logical_requests.json') if r['predictor']=='esmfold']
    if len(logical)!=1864:raise ValueError('ESM extension roster differs')
    physical={r['task_id']:r for r in read(root/CAT/'physical_requests.json')}
    protocol=read(root/CAT/'statistical_protocol.json');components=protocol['component_by_protein_id']
    panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    old=read(root/PRIMARY/'per_output_metrics.json');metrics={}
    def key(r):return (r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))
    for r in old:metrics[key(r)]={n:r[n] for n in ['common_structure','fixed_identity_geometry']}
    artifacts={};statuses={};audit=[]
    for task in {r['physical_id'] for r in logical}:
        p=physical[task];record=root/STRUCT/'records'/(task+'.json');saved=read(record) if record.exists() else {}
        status=saved.get('status','NOT_PRODUCED');path=Path(saved['output_path']) if status=='SUCCEEDED' else None
        if status=='SUCCEEDED':
            if any(saved.get(k)!=p[k] for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']):raise ValueError('Prediction identity changed')
            if sha(path)!=saved['output_sha256']:raise ValueError('Prediction artifact SHA differs')
            with np.load(path,allow_pickle=False) as z:
                if str(z['sequence'])!=p['sequence'] or z['coordinates'].shape!=(len(p['sequence']),3):raise ValueError('Predicted sequence/CA count differs')
        artifacts[task]=str(path) if path else None;statuses[task]=status
        audit.append(dict(physical_id=task,status=status,artifact_sha256=saved.get('output_sha256')))
    jobs={}
    for r in logical:
        k=key(r)
        if k not in metrics:jobs.setdefault(k,(str(root),k,panel[r['protein_id']]['wt_native_json'],artifacts[r['physical_id']],len(r['edited_sequence']),r['correspondence']))
    with ProcessPoolExecutor(max_workers=8) as executor:metrics.update(executor.map(metric_pair,jobs.values()))
    rows=[dict(r,prediction_status=statuses[r['physical_id']],**metrics[key(r)]) for r in logical]
    by={(r['protein_id'],r['method'],r['budget']):r for r in rows if r['evaluation_role'] in ['primary20pct','budget_curve','mechanism_control','quota_control']}
    def comparison(ids,method,versus,beta,endpoint,**meta):
        pairs=[]
        for pid in sorted(ids):
            a=by[pid,method,beta]['common_structure'].get(endpoint);b=by[pid,versus,beta]['common_structure'].get(endpoint)
            pairs.append((pid,a-b if a is not None and b is not None else None))
        return dict(method=method,versus=versus,budget=beta,endpoint=endpoint,**meta,**effect(pairs,components,protocol['bootstrap_seed']))
    budget_ids={r['protein_id'] for r in rows if r['evaluation_role']=='budget_curve'}
    if len(budget_ids)!=40:raise ValueError('Fixed budget40 cohort differs')
    budget_table=[];budget_effects=[]
    for beta in [.1,.2,.3,.4]:
        for method in protocol['methods']:
            group=[by[pid,method,beta] for pid in sorted(budget_ids)];entry=dict(method=method,budget=beta,N=40)
            for endpoint in ENDPOINTS:
                vals=[r['common_structure'].get(endpoint) for r in group]
                entry[endpoint]=float(np.mean(vals)) if all(v is not None for v in vals) else None
                entry[endpoint+'_defined_N']=sum(v is not None for v in vals)
                if method!='delepref_unlimited':budget_effects.append(comparison(budget_ids,method,'delepref_unlimited',beta,endpoint,analysis='E1b_secondary'))
            budget_table.append(entry)
    diagnostics=read(root/'reports/recomb_confirm_v2_General100_mechanisms/diagnostics.json')
    e4={d['protein_id'] for d in diagnostics}
    valid_joint={d['protein_id'] for d in diagnostics if d['joint_null'].get('effective_intervention',False)}
    valid_shape={d['protein_id'] for d in diagnostics if any(x.get('effective_final_shape_matched_intervention',False) for x in d['morphology'])}
    control_effects=[]
    for method in ['rag_morphology_randomized','rag_joint_null','rag_direct_only','rag_no_scorer']:
        subsets=[('all_predefined40',e4)]
        if method=='rag_joint_null':subsets.append(('input_effective_joint_intervention',valid_joint))
        if method=='rag_morphology_randomized':subsets.append(('input_effective_shape_intervention',valid_shape))
        for name,ids in subsets:
            for endpoint in ENDPOINTS:
                control_effects.append(comparison(ids,method,'rag_unlimited',.2,endpoint,analysis='E4_secondary',cohort=name))
    scarcity=[]
    for method in ['rag_evidence_50','rag_evidence_25','rag_evidence_0']:
        ids={r['protein_id'] for r in rows if r['method']==method}
        if len(ids)!=20:raise ValueError('Predefined scarcity20 cohort differs')
        for pid in ids:
            if method=='rag_evidence_0' and by[pid,method,.2]['edited_sequence_sha256']!=by[pid,'delepref_unlimited',.2]['edited_sequence_sha256']:
                raise ValueError('Zero-evidence exact fallback changed')
        for endpoint in ENDPOINTS:scarcity.append(comparison(ids,method,'rag_unlimited',.2,endpoint,analysis='E5_secondary'))
    pools=defaultdict(list)
    for r in rows:
        if r['evaluation_role']=='full_pool':pools[r['protein_id'],'RAG' if r['method'].startswith('pool_RAG_') else 'original'].append(r)
    if len(pools)!=24:raise ValueError('Full12WT two-pool cohort differs')
    pool_rows=[]
    for (pid,source),candidates in sorted(pools.items()):
        selected=by[pid,'rag_unlimited' if source=='RAG' else 'delepref_unlimited',.2]
        if not any(r['edited_sequence_sha256']==selected['edited_sequence_sha256'] and r['correspondence']['wt_to_edit']==selected['correspondence']['wt_to_edit'] for r in candidates):
            raise ValueError('Original selected plan absent from its complete candidate pool')
        for endpoint in ENDPOINTS:
            pool_rows.append(dict(protein_id=pid,pool=source,endpoint=endpoint,
                distinct_sequences=len({r['edited_sequence_sha256'] for r in candidates}),
                **pool_summary([r['common_structure'].get(endpoint) for r in candidates],selected['common_structure'].get(endpoint))))
    pool_effects=[];pool_map={(r['protein_id'],r['pool'],r['endpoint']):r for r in pool_rows}
    pool_ids={pid for pid,_ in pools}
    for endpoint in ENDPOINTS:
        for field in ['best','selected','regret']:
            pairs=[]
            for pid in sorted(pool_ids):
                a=pool_map[pid,'RAG',endpoint].get(field);b=pool_map[pid,'original',endpoint].get(field)
                pairs.append((pid,a-b if a is not None and b is not None else None))
            pool_effects.append(dict(analysis='E7_descriptive',endpoint=endpoint,field=field,
                contrast='RAG_pool minus original_pool; lower regret is better',**effect(pairs,components,protocol['bootstrap_seed'])))
    quota=[]
    for endpoint in ENDPOINTS:quota.append(comparison(pool_ids,'rag_preserve_original_winner','rag_unlimited',.2,endpoint,analysis='E7_quota_descriptive'))
    outputs=dict(per_output_metrics=rows,prediction_identity_audit=audit,budget_table=budget_table,budget_comparisons=budget_effects,
        mechanism_comparisons=control_effects,evidence_scarcity_comparisons=scarcity,
        full_pool_per_WT=pool_rows,full_pool_comparisons=pool_effects,quota_comparisons=quota)
    for name,value in outputs.items():write(out/(name+'.json'),value)
    done=dict(status='ALL1864_ESMFOLD_PREDECLARED_EXTENSION_SUMMARIES_COMPLETE_SECOND_PREDICTOR_PENDING',
        logical=1864,physical=len(artifacts),missing_or_failed=sum(s!='SUCCEEDED' for s in statuses.values()),
        primary_logical_metrics_reused=700,additional_metric_jobs=len(jobs),
        E1b_WT=40,E4_WT=len(e4),effective_joint_WT=len(valid_joint),effective_shape_WT=len(valid_shape),E5_WT=20,E7_WT=12,
        secondary_and_descriptive_not_replacement_primary=True,no_output_reselection=True,new_structure_predictions=0,
        outputs={name+'.json':sha(out/(name+'.json')) for name in outputs},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else evaluate(a.root)
