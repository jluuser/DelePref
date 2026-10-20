"""Reuse retained rebuttal aggregate evidence without resampling or retraining models."""
from collections import Counter
import csv
import json
from pathlib import Path
import os
from recomb_expanded_external_prepare_20261017 import ROOT,sha,write
SOURCE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit/results/experiments')
BASE='reports/recomb_legacy_evidence_20261017'


def main():
    packages={
      'v3_15_decoder_ablation':['decoder_metrics.csv','cath_decoder_metrics.csv','decoder_paired_bootstrap_ci.csv'],
      'v3_16_binder_coldstart_same_scorer':['binder_method_metrics.csv','binder_paired_bootstrap_ci.csv'],
      'v3_19_teacher_gate_tie_sensitivity':['v3_7_weight_sensitivity_index.csv','gate_tie_pair_summary.csv','gate_tie_candidate_summary.csv','gate_tie_paired_bootstrap_ci.csv'],
      'v3_25_binder50_fair_contact_comparison':['binder50_method_summary.csv','binder50_threshold_summary.csv','binder50_paired_bootstrap_ci.csv'],
      'v3_26_binder50_complete_metrics':['binder50_complete_method_summary.csv','binder50_complete_paired_bootstrap_ci.csv']}
    manifest=[];summaries={};runs={}
    for label,names in packages.items():
        folder=SOURCE/('student_v3_step004000_'+label)
        runs[label]=json.loads((folder/'run_summary.json').read_text())
        manifest.append(dict(path=str(folder/'run_summary.json'),sha256=sha(folder/'run_summary.json')))
        summaries[label]={}
        for name in names:
            path=folder/name
            with path.open() as f:
                reader=csv.DictReader(f);rows=list(reader);columns=reader.fieldnames
            manifest.append(dict(path=str(path),sha256=sha(path),rows=len(rows),columns=columns))
            summaries[label][name]=rows
    weights=summaries['v3_19_teacher_gate_tie_sensitivity']['v3_7_weight_sensitivity_index.csv']
    assert len(weights)==8 and all(r['post_hoc_not_training_causal'].lower()=='true' for r in weights)
    # All definitions and intervals retained, including unstable signs and poor agreement.
    weight_summary=[{k:r[k] for k in ['definition','global_weight','retained_weight','junction_weight',
        'mean_spearman_with_canonical','mean_top1_agreement_with_canonical','paired_all_rows','paired_sign_stable_rows',
        'post_hoc_not_training_causal','status']} for r in weights]
    write(BASE+'/aggregate_tables.json',summaries);write(BASE+'/original_run_summaries.json',runs)
    write(BASE+'/source_manifest.json',manifest);write(BASE+'/teacher_weight_summary.json',weight_summary)
    binder=summaries['v3_26_binder50_complete_metrics']['binder50_complete_method_summary.csv']
    write(BASE+'/binder_absolute_metric_rows.json',[r for r in binder if any(k in r['metric'] for k in
        ['contact','interface','clash','strict','relaxed','confidence','span','PAE','pae'])])
    write(BASE+'/completion.json',dict(status='RETAINED_COMPONENT_TEACHER_AND_BINDER_AGGREGATES_REUSED',job_id=os.environ.get('SLURM_JOB_ID'),
        packages=len(packages),tables=sum(len(v) for v in summaries.values()),source_files=len(manifest),teacher_utility_definitions=8,
        all_indexed_weight_rows_retained=True,all_paired_interval_rows_retained=True,
        old_span_limit_version='Historical q<=3; not new unlimited-span RAG',
        teacher_sensitivity='Post-hoc score-definition/gate/tie sensitivity; no causal retraining robustness claim',
        binder='Existing50 complexes and all aggregate failure/absolute metrics; no new RAG complex results or measured affinity assigned',
        positional_proxy='Historical central60% proxy remains archived, excluded from new biological-core claims',
        generation_contribution='Matched same-Scorer historical comparisons; keep their actual budget/K and protocol classes',
        candidate_structure_oracle_available=False,new_model_training=0,new_PLM_forwards=0,new_structure_calls=0,
        source_manifest_sha256=sha(ROOT/BASE/'source_manifest.json')))
    print(json.dumps(json.loads((ROOT/BASE/'completion.json').read_text())),flush=True)


if __name__=='__main__':main()
