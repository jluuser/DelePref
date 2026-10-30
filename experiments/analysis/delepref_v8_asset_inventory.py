"""Inventory existing AAAI evidence; metadata only, no model or outcome selection."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
EXPS={
 'teacher_gate_tie_sensitivity':'student_v3_step004000_v3_19_teacher_gate_tie_sensitivity',
 'training_homology_audit':'student_v3_step004000_v3_18_homology_leakage_audit',
 'coldstart_same_scorer':'student_v3_step004000_v3_16_binder_coldstart_same_scorer',
 'decoder_ablation':'student_v3_step004000_v3_15_decoder_ablation',
 'direct_PLM_baselines':'student_v3_step004000_v3_20_direct_plm_cross_teacher',
 'cross_family_encoder':'student_v3_step004000_v3_21_prott5_crossfamily_audit'}
out={}
for name,folder in EXPS.items():
    p=BASE/'results/experiments'/folder/'run_summary.json'
    if not p.is_file():
        out[name]=dict(path=str(p),exists=False);continue
    raw=p.read_bytes();d=json.loads(raw)
    out[name]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),exists=True,
        summary={k:d[k] for k in ['status','schema_version','execution','canonical_reproduction','cohort_counts',
          'claim_boundary','grid','bootstrap','negative_instability_and_coverage_reductions_retained'] if k in d})
p=BASE/'data/processed/enzyme_v4_public_measured/enzyme_standardized_measured_variants.csv'
with p.open(newline='') as f:
    reader=csv.DictReader(f);columns=reader.fieldnames;count=sum(1 for _ in reader)
report=dict(status='INVENTORY_ONLY_NOT_SCIENTIFIC_REVALIDATION',experiments=out,
    original_measured_table=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=count,columns=columns),
    duplicate_release_and_mapping_protocol='Keep fixed v6 primary 8-condition/1317-observation view; recovered legacy table is reproduction only.',
    new_GPU_jobs=0,new_PLM_forwards=0,paper_changes=False)
target=ROOT/'reports/delepref_v8/inherited_asset_inventory.json'
with target.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(report_path=str(target),existing_experiments=sum(x['exists'] for x in out.values()),original_measured_rows=count)))
