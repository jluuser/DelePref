"""Final immutable output, metric, statistics, predictor and dispatch closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from recomb_scope_screen_20261017 import read,sha,write


def components(general,applications):
    rows=general+applications;parent={r['protein_id']:r['protein_id'] for r in rows};seen={}
    def find(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in rows:
        scope='general' if 'stratum' in r else r['dataset'];pid=r['protein_id']
        tags=[('WT',r['sequence_sha256'])]
        if scope=='general':
            tags += [(k,r[k]) for k in ['sequence_homology_component','panel_component','uniprot_accession'] if r.get(k)]
            tags += [('CATH_H',h) for h in r.get('cath_h_superfamilies',[])]
        elif scope=='Binder20':
            tags += [('native_source',r['native_reference_sha256'])]
            tags += [('shared_chain',c['sequence_sha256']) for c in r['partners']]
            tags += [('shared_chain',r['sequence_sha256'])]
        for kind,value in tags:
            key=scope,kind,json.dumps(value,sort_keys=True)
            if key in seen:union(pid,seen[key])
            else:seen[key]=pid
    return {p:find(p) for p in parent}


def release(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    cat=root/'reports/recomb_benign_prediction_catalog_20261017'
    target=root/'configs/recomb_benign_confirmation_freeze_20261017.json'
    if target.exists():raise FileExistsError('final release already exists')
    done=read(cat/'completion.json')
    for n,h in done['outputs'].items():
        if sha(cat/n)!=h:raise ValueError('catalog output changed')
    closure={}
    for path,h in done['source_sha256'].items():
        p=Path(path) if Path(path).is_absolute() else root/path
        if sha(p)!=h:raise ValueError('catalog source changed')
        closure[str(p)]=h
    for predictor in ['esmfold','boltz2']:
        report=read(root/'reports/delepref_v8_predictors'/(predictor+'_preflight.json'))
        for f in report['files']+report['implementation']+report.get('standard_amino_acid_assets',[]):
            if sha(f['path'])!=f['sha256']:raise ValueError('predictor assets changed')
            closure[f['path']]=f['sha256']
    general=read(root/'reports/recomb_benign_scope_20261017/general_panel.json')
    app=read(root/'reports/recomb_benign_application_constructs_20261017/panel.json')
    protocol=dict(status='FIXED_WHOLE_COHORT_MULTI_ENDPOINT_PROTOCOL',main_WT=44,budget_mechanism_WT=20,scarcity_WT=10,
        second_predictor_WT=8,binding_interface_cases=12,enzyme_cases=11,logical_outputs=867,
        bootstrap_seed=20261016,bootstrap_replicates=10000,numeric_equality_tolerance=1e-8,
        component_by_protein_id=components(general,app),
        components='same WT, original homology/panel/UniProt/CATH H relationships for general; common source/shared construct chain for binding cases; no outcome-based grouping',
        effect_estimand='construct macro paired difference; component-resampling bootstrap with within-component cases together; intervals encompass metric uncertainty',
        primary_metrics=['retained_ca_lddt','native_contact_recovery','full_WT_fixed_correspondence_TM'],
        reference_denominator='fixed complete observed-native WT relationships; certain deleted endpoints zero; unknown prediction/mapping endpoints bounded, never silently discarded',
        correspondence='all-optimal unit-edit stable consensus for Raygun; explicit provenance for deletion methods; no structure-dependent sequence remapping',
        ambiguous_retained_lddt='retained membership/neighbour denominators uncertain: conservative[0,1], stable-subset local geometry separately diagnostic with mapping coverage',
        TM='official USalign -I fixed known correspondence and -u full WT length; missing upper bound conditional on this known-CA fit; not an unknown-structure reoptimized bound',
        binding='observed-native heavy-atom contacts<5A; fixed design and partner mappings; recovery/Jaccard intervals for unknown mapping or coordinates; unresolved native region coverage separate',
        enzyme='annotated catalytic identity retention, CA-neighbour native contact recovery, and catalytic CA pair distance MAE with pair count; no claim of activity',
        evidence='all primary44 and all preregistered20/10/8; three metrics independently; changed-decision/coverage plots diagnostic; no oracle or biological functional assertion',
        historical_predictions=758,structure_prediction_limit=None,
        compute=dict(structure_call_limit=None,structure_categories=dict(main_or_application_final_output=None,retry=None),
            structure_retries_max=1,max_task_GPUs=2,GPU_only_Slurm=True),
        technical_failures='preserve failures and original logical denominators; at most one audited technical retry; two consecutive worker failures stop',
        application_cache='exact full canonical chain/weights/profile/seed/sample only; current release predicts application outputs afresh rather than use incomplete legacy provenance')
    write(cat/'statistical_protocol.json',protocol)
    paths=['analysis/recomb_benign_fold_worker_20261017.py','analysis/recomb_benign_structure_evaluate_20261017.py',
        'analysis/recomb_benign_metrics_20261017.py','analysis/recomb_benign_binding_native_20261017.py',
        'analysis/recomb_benign_structure_release_20261017.py','analysis/delepref_v8_structure_metrics.py',
        'slurm/recomb_benign_final_structures_20261017.sbatch',
        'analysis/recomb_edit_correspondence_20261017.py','code/build_pairs.py','code/common.py','code/adapters/structure_adapter.py',
        'code/prediction_journal.py','code/budget_ledger.py','reports/recomb_rag_decision_audit_20261017/per_condition.json',
        'reports/recomb_rag_decision_audit_20261017/completion.json']
    paths += ['reports/recomb_benign_prediction_catalog_20261017/'+n for n in done['outputs']]
    paths += ['reports/recomb_benign_prediction_catalog_20261017/completion.json','reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json']
    for p in paths:closure[str(root/p)]=sha(root/p)
    for old in read(cat/'compatible_cache.json').values():
        if sha(old['record_path'])!=old['record_sha256'] or sha(old['artifact_path'])!=old['artifact_sha256']:raise ValueError('compatible cache changed')
        closure[old['record_path']]=old['record_sha256'];closure[old['artifact_path']]=old['artifact_sha256']
    tests=[]
    for name in ['test_recomb_benign_metrics.py','test_recomb_benign_prediction_interfaces.py','test_prediction_journal.py']:
        run=subprocess.run([sys.executable,'-m','unittest','discover','-s','analysis/tests','-p',name,'-v'],cwd=root,capture_output=True,text=True)
        tests.append(dict(test=name,returncode=run.returncode,output=run.stdout+run.stderr))
        if run.returncode:raise RuntimeError('prediction release tests failed: '+name+' '+run.stderr)
    write(cat/'release_tests.json',tests);closure[str(cat/'release_tests.json')]=sha(cat/'release_tests.json')
    frozen=dict(status='COMPLETE_SCREENED_EXPERIMENT_RELEASE',fold_allowed=True,all_selected_outputs_frozen=True,
        main_WT=44,logical_requests=867,unique_physical=651,verified_compatible_cache=232,planned_missing_calls=419,
        complete_structure_prediction_limit=None,max_task_concurrent_GPUs=2,historical_calls=758,
        files=[dict(path=p,sha256=h) for p,h in sorted(closure.items())],
        predictor_profiles=str(cat/'predictor_profiles.json'),statistical_protocol=str(cat/'statistical_protocol.json'),
        original_weights_and_shared_environments_unchanged=True,paper_paused=True)
    write(target,frozen)
    write(cat/'release_completion.json',dict(status='RELEASED_FOR_MISSING_FINAL_PREDICTIONS',
        confirmation_freeze_sha256=sha(target),logical_requests=867,missing_calls=419,verified_cache=232,
        tests_passed=True,new_structure_calls=0,job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(read(cat/'release_completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    release(p.parse_args().root)
