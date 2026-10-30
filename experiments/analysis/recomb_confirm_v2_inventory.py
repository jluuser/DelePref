"""Metadata-only candidate inventory. No unreviewed sequence is released for models."""
import argparse
from collections import Counter
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

GROUPS=Path('/public/nfs-data/keyuan.zhou/evocompress/results/iclr_completion_20260910/E1_E4/Raygun_native/groups')
FIELDS=('mcsa_id','enzyme_name','uniprot_accession','reviewed','pdb_id','auth_chain_id','assembly_id',
        'ec_numbers','ec_class','rcsb_title','rcsb_keywords','rcsb_keyword_text','experimental_methods',
        'resolution','initial_release_date','pdb_entity_id','coordinate_coordinate_valid',
        'coordinate_selected_label_asym_id','coordinate_CA_coverage','interface_class','audit_class',
        'source_organism','organism','taxonomy','design_chain','partner_chain','design_entity_id',
        'partner_entity_id','binder_type','source_dataset','category','source')


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Metadata-only CPU Slurm required')
    out=root/'reports/recomb_confirm_v2_inventory'
    if (out/'completion.json').exists():raise FileExistsError('Inventory already complete')
    old_general=read(root/'reports/recomb_benign_scope_20261017/general_audit.json')
    old_app=read(root/'reports/recomb_benign_application_scope_20261017/audit.json')
    old_sha={r['WT_sha256'] for r in old_general}
    rows=[]
    for cohort,prefix in [('enzyme','E2_'),('binding_interface','E3_')]:
        for path in sorted(GROUPS.glob(prefix+'*.json')):
            g=read(path);ref=g['reference']
            metadata={k:ref[k] for k in FIELDS if k in ref}
            rows.append(dict(cohort=cohort,candidate_id=g['group_key'],protein_id=g['protein_id'],
                source_file=str(path),source_file_sha256=sha(path),
                WT_sha256=g.get('wt_sequence_sha256'),WT_length=len(g.get('wt_sequence','')),
                partner_present=bool(g.get('partner_sequence')),metadata=metadata,
                available_reference_metadata_keys=sorted(k for k in ref if 'sequence' not in k.lower()),
                prior_other_project_baseline_results_exist=True,
                source_scope='UNREVIEWED_METADATA_ONLY_NO_MODEL_USE',
                eligibility_for_new_confirmation='pending_scope_construct_family_and_development_overlap_audit'))
    candidates=read(root/'reports/delepref_v8_panel/candidate_universe.json')
    native_audit={r['candidate_id']:r for r in read(root/'reports/delepref_v8_native_panel/reference_audit.json')}
    general=[]
    for r in candidates:
        general.append({k:r[k] for k in ('candidate_id','protein_id','sequence_sha256','stratum','domain_id',
            'uniprot_id','accession','pdb_id','chain_id','coordinate_path','structure_path','source_file',
            'sequence_homology_component','cath_h_superfamily') if k in r})
        general[-1].update(WT_length=len(r.get('sequence','')),previously_in_original100=r['sequence_sha256'] in old_sha,
            native_audit=native_audit.get(r['candidate_id'],{}),
            source_scope='UNREVIEWED_METADATA_ONLY_NO_MODEL_USE')
    write(out/'application_candidate_metadata.json',rows)
    write(out/'general_candidate_metadata.json',general)
    write(out/'historical_scope_recovery_queue.json',[
        dict(cohort='general',protein_id=r['protein_id'],WT_sha256=r['WT_sha256'],
            reason=r['reason'],annotation_files=r.get('annotation_files',[]),
            model_use_allowed=False,previous_model_outputs_exist=True)
        for r in old_general if r['status']=='HOLD']+[
        dict(cohort='application',protein_id=r['protein_id'],WT_sha256=r['WT_sha256'],
            reason=r.get('reason'),model_use_allowed=False,previous_model_outputs_exist=True)
        for r in old_app if r['status']=='HOLD'])
    report=dict(status='METADATA_INVENTORY_COMPLETE_NOT_MODEL_INPUT_RELEASE',
        application_candidates=dict(Counter(r['cohort'] for r in rows)),general_candidates=len(general),
        general_outside_original100=sum(not r['previously_in_original100'] for r in general),
        general_outside100_with_existing_native_PASS=sum(not r['previously_in_original100'] and
            r['native_audit'].get('status')=='PASS' for r in general),
        raw_sequences_saved=False,variant_outputs_exported=False,new_model_calls=0,
        new_positive_scope_decisions=0,job_id=os.environ['SLURM_JOB_ID'],
        files={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'completion.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
