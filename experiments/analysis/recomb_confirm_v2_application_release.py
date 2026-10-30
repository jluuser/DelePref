"""Release the complete fixed application experiment only after all CPU prerequisites pass."""
import argparse
from pathlib import Path
import os
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_application_catalog_v2 import BASE as CAT,digest

CONFIRM=Path('configs/recomb_confirm_v2_application_confirmation_freeze.json')
METRIC=Path('reports/recomb_confirm_v2_application_metric_preflight')


def release(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm release only')
    if (root/CONFIRM).exists():raise FileExistsError('Application complete confirmation already released')
    catalog=read(root/CAT/'completion.json');parsed=read(root/CAT/'native_parser_completion.json')
    metric=read(root/METRIC/'completion.json');qa=read(root/'reports/recomb_confirm_v2_execution/application_software_QA.json')
    if catalog['logical']!=2160 or parsed['unresolved'] or metric['cases']!=140 or qa['passed_tests']!=16:
        raise ValueError('Application catalog/native parser/metric/software prerequisites incomplete')
    files={}
    for source in ['configs/recomb_confirm_v2_application_catalog_inputs_v2_freeze.json',
        'configs/recomb_confirm_v2_application_catalog_parser_v2_freeze.json',
        'configs/recomb_confirm_v2_application_metric_preflight_freeze.json']:
        f=read(root/source)
        for p,h in {**f['files'],**f.get('native_reference_sha256',{})}.items():
            path=Path(p) if Path(p).is_absolute() else root/p
            if sha(path)!=h:raise ValueError('Frozen prerelease source/input changed: '+p)
            files[str(path)]=h
        files[str(root/source)]=sha(root/source)
    for p,h in qa['source_sha256'].items():
        if sha(root/p)!=h:raise ValueError('Software source changed after QA')
        files[str(root/p)]=h
    for p,h in catalog['outputs'].items():
        if sha(root/CAT/p)!=h:raise ValueError('Completed application catalog output changed')
        files[str(root/CAT/p)]=h
    for p,h in metric['outputs'].items():
        if sha(root/METRIC/p)!=h:raise ValueError('Completed native metric fixture changed')
        files[str(root/METRIC/p)]=h
    if sha(root/CAT/'native_parser_audit.json')!=parsed['native_parser_audit_sha256']:raise ValueError('Completed native parser audit changed')
    names=['analysis/recomb_confirm_v2_application_release.py','analysis/recomb_confirm_v2_application_predict.py',
        'analysis/recomb_confirm_v2_application_evaluate.py','analysis/recomb_confirm_v2_application_metrics.py',
        'analysis/recomb_confirm_v2_general_predict.py','analysis/recomb_confirm_v2_general_evaluate.py',
        'analysis/recomb_confirm_v2_structure_alignment.py','analysis/recomb_benign_metrics_20261017.py',
        'analysis/recomb_benign_catalytic_identity_20261017.py','analysis/delepref_v8_structure_metrics.py',
        'code/prediction_journal.py','code/budget_ledger.py',
        'slurm/recomb_confirm_v2_application_Boltz.sbatch','slurm/recomb_confirm_v2_application_evaluate.sbatch',
        'slurm/recomb_confirm_v2_application_release.sbatch',
        'reports/recomb_confirm_v2_execution/application_software_QA.json',
        str(CAT/'completion.json'),str(CAT/'native_parser_completion.json'),str(CAT/'native_parser_audit.json'),
        str(METRIC/'completion.json'),'reports/delepref_v8_predictors/boltz2_preflight.json',
        'provenance/usalign_20261017/USalign']
    for p in names:files[str(root/p)]=sha(root/p)
    profiles=read(root/CAT/'predictor_profiles.json');logical=read(root/CAT/'logical_requests.json')
    physical=read(root/CAT/'physical_requests.json');msas=read(root/CAT/'MSA_manifest.json')
    if len(logical)!=2160:raise ValueError('Final full application matrix differs')
    for m in msas:
        path=root/m['CSV_relative_path']
        if sha(path)!=m['CSV_sha256']:raise ValueError('Actual materialized MSA changed')
        files[str(path)]=m['CSV_sha256']
    needed=set()
    for r in physical:
        if digest(profiles[r['profile_name']])!=r['profile_sha256']:raise ValueError('Predictor physical profile differs')
        p=root/r['model_input_relative_path']
        if sha(p)!=r['model_input_sha256']:raise ValueError('Materialized complex input changed')
        files[str(p)]=r['model_input_sha256']
        for item in read(p)['sequences']:
            if 'ligand' in item:needed.update(item['ligand']['ccd'])
    preflight=read(root/'reports/delepref_v8_predictors/boltz2_preflight.json')
    for f in preflight['files']+preflight['implementation']+preflight.get('standard_amino_acid_assets',[]):
        if sha(f['path'])!=f['sha256']:raise ValueError('Predictor asset or implementation changed')
        files[f['path']]=f['sha256']
    for code in needed:
        p=Path('/public/home/zhangyangroup/chengshiz/.boltz/mols')/(code+'.pkl');files[str(p)]=sha(p)
    components={};repeat={}
    for cohort in ['Binder60','Enzyme80']:
        panel=read(root/('reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json'))
        for p in panel:
            components[p['protein_id']]=cohort+'|'+str(p['combined_component'] if cohort=='Binder60' else p['evaluation_sequence_component'])
            files[p['native_assembly_reference']]=p['native_assembly_reference_sha256']
        p=root/('reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json');s=read(p)
        repeat[cohort]=sorted(r['protein_id'] for r in s['budget10pct_and_three_structure_seeds']);files[str(p)]=sha(p)
    for row in logical:
        maps=row['native_to_prediction_chains']
        if len({m['native_instance_id'] for m in maps})!=len(maps):raise ValueError('Assembly-instance provenance is not unique')
    protocol=dict(status='COMPLETE_APPLICATION80_60_MAIN_BUDGET_AND_REPEAT_PROTOCOL',
        methods=['delepref_unlimited','rag_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native'],
        primary_endpoint=dict(Enzyme80='complete_catalytic_AA_identity',Binder60='interface_contact_recovery'),
        primary_comparison='RAG minus original DelePref, whole cohort20pct, fixed main seed20261017',
        component_by_protein_id=components,bootstrap_replicates=10000,bootstrap_seed=20261016,ordinary_CI=.95,
        repeat_subset_by_cohort=repeat,structure_seeds=[20261017,20261018,20261019],
        repeat_analysis='mean within WT/method across all3fixed seeds before component inference; WT calibration same',
        no_best_seed_selection=True,no_pseudo_replication=True,unknown_bounds_and_failures_retained=True,
        main_roster=dict(Binder60=60,Enzyme80=80),WT_first_for_calibration=True,
        phase_order='WT seed17 -> main20pct seed17 -> budget and fixed20pct seed18/19 extension',
        calibrated_core='Boltz2 r3/200, source-reviewed real MSAs, no native coordinate templates or affinity',
        no_model_candidate_or_selector_change=True,
        compute=dict(structure_call_limit=None,structure_categories=dict(application_Binder60=None,application_Enzyme80=None,retry=None),structure_retries_max=1),
        GPU_concurrency='existing total task GPU<=2, never launch a third allocation',
        structural_metrics_not_functional_assay=True)
    write(root/CAT/'statistical_protocol.json',protocol);files[str(root/CAT/'statistical_protocol.json')]=sha(root/CAT/'statistical_protocol.json')
    write(root/CONFIRM,dict(status='COMPLETE_APPLICATION_SEQUENCE_CONTEXT_PREDICTOR_METRIC_STATISTICS_FREEZE',
        fold_allowed=True,all_selected_outputs_frozen=True,structure_prediction_limit=None,total_logical_requests=2160,
        physical_requests=len(physical),generation_checkpoint_sha256='1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d',
        files=[dict(path=p,sha256=h) for p,h in sorted(files.items())],
        application_native_CCD_representation_limits_preserved=True,
        GPU_dispatch_requires_separate_live_capacity_check=True,new_training=False))
    done=dict(status='APPLICATION_ALL_CPU_INPUT_AND_SOFTWARE_PREREQUISITES_PASS_GPU_RELEASE_FROZEN_NOT_DISPATCHED',
        logical=2160,physical=len(physical),confirmation_freeze_sha256=sha(root/CONFIRM),
        GPU_jobs_submitted_by_this_release=0,new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'])
    write(root/CAT/'release_completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);release(p.parse_args().root)
