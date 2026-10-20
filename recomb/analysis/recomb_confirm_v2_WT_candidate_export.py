"""Materialize only metadata-approved ordinary WTs for native/homology audits."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

FZ=Path('configs/recomb_confirm_v2_WT_candidate_export_freeze.json')
OUT=Path('reports/recomb_confirm_v2_WT_candidate_audit_inputs')
CATH=Path('/public/nfs-data/keyuan.zhou/evocompress/data/cath_s40_v1/inventory/eligible_domains.json')
ENZYME=Path('/public/nfs-data/keyuan.zhou/evocompress/data/enzyme_120_v1/coordinate_audit_v1/coordinate_pass.jsonl')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('WT candidate export already frozen')
    files=['analysis/recomb_confirm_v2_WT_candidate_export.py','analysis/recomb_confirm_v2_diagnostics.py',
        'reports/recomb_confirm_v2_expansion_review/CATH_metadata_review.json',
        'reports/recomb_confirm_v2_expansion_review/MCSA_metadata_review.json',
        'reports/recomb_confirm_v2_expansion_review/completion.json',
        'reports/recomb_confirm_v2_cohort_prescreen/application_prescreen.json',
        'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json',
        'reports/recomb_benign_scope_20261017/general_audit.json',
        'reports/recomb_benign_application_scope_20261017/audit.json',
        'reports/recomb_benign_application_constructs_20261017/panel.json',
        'slurm/recomb_confirm_v2_WT_candidate_export.sbatch']
    write(root/FZ,dict(status='ORDINARY_WT_EXPORT_FOR_INPUT_AUDITS_ONLY_FROZEN',
        files={p:sha(root/p) for p in files},external_files={str(p):sha(p) for p in [CATH,ENZYME]},
        permitted=['WT_sequence_materialization','native_mapping','homology_audit'],
        deletion_generation_allowed=False,PLM_encoding_allowed=False,structure_prediction_allowed=False))
    print(sha(root/FZ))


def execute(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen WT audit source changed: '+p)
    for p,h in f['external_files'].items():
        if sha(p)!=h:raise ValueError('Cached input inventory changed')
    out=root/OUT
    if (out/'completion.json').exists():raise FileExistsError('WT candidates already materialized')
    historical={r['WT_sha256'] for r in read(root/'reports/recomb_benign_scope_20261017/general_audit.json')+
                read(root/'reports/recomb_benign_application_scope_20261017/audit.json')}
    for p in read(root/'reports/recomb_benign_application_constructs_20261017/panel.json'):
        historical.update(c['sequence_sha256'] for c in p.get('partners',[]))
    cath={r['domain_id']:r for r in read(CATH)}
    enzymes={}
    for line in ENZYME.read_text().splitlines():
        r=__import__('json').loads(line);enzymes[r['candidate_id']]=r
    candidates=[];rejected=[]
    def add(row):
        seq=row['sequence'];h=hashlib.sha256(seq.encode()).hexdigest()
        if h!=row['WT_sha256']:raise ValueError('Source/metadata WT SHA mismatch')
        if h in historical:
            rejected.append(dict(candidate_id=row['candidate_id'],reason='historical_WT_or_partner'));return
        if not 50<=len(seq)<=1022 or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in seq):
            rejected.append(dict(candidate_id=row['candidate_id'],reason='length_or_nonstandard_AA'));return
        row.update(WT_sequence_export_allowed_for_input_audits=True,model_use_allowed=False,
                   native_and_family_release_pending=True)
        candidates.append(row)
    for r in read(root/'reports/recomb_confirm_v2_expansion_review/CATH_metadata_review.json'):
        if r['metadata_review_status']!='PASS_SOURCE_AND_FUNCTION_METADATA':continue
        c=cath[r['domain_id']]
        add(dict(candidate_id='CATH|'+r['domain_id'],role='general_domain',sequence=c['sequence'],WT_sha256=r['WT_sha256'],
            source_scope_review_status=r['metadata_review_status'],source_scope_review_file='reports/recomb_confirm_v2_expansion_review/CATH_metadata_review.json',
            cath_family=r['family'],cath_class=r['cath_class'],domain_id=r['domain_id'],
            archive_member=r['archive_member'],native_source='cached_CATH_S40_archive'))
    for r in read(root/'reports/recomb_confirm_v2_expansion_review/MCSA_metadata_review.json'):
        if r['metadata_review_status']!='PASS_SOURCE_FUNCTION_AND_CANONICAL_IDENTITY':continue
        e=enzymes[r['candidate_id']]
        add(dict(candidate_id='MCSA|'+str(r['candidate_id']),role='enzyme',sequence=e['sequence'],WT_sha256=r['WT_sha256'],
            source_scope_review_status=r['metadata_review_status'],source_scope_review_file='reports/recomb_confirm_v2_expansion_review/MCSA_metadata_review.json',
            uniprot_accession=r['uniprot_accession'],mcsa_id=r['mcsa_id'],enzyme_name=r['enzyme_name'],
            ec_numbers=r.get('ec_numbers'),source_family=r.get('sequence_homology_component'),
            native_reference_path=r.get('reference_structure_path'),native_reference_chain=r.get('reference_chain_id'),
            native_reference_pdb=r.get('reference_pdb_id'),catalytic_residue_count=r.get('catalytic_residue_count'),
            all_catalytic_residues_resolved=r.get('all_catalytic_residues_resolved'),
            source_candidate_id=r['candidate_id'],native_source='cached_MCSA_coordinate_audit'))
    inventory={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json')}
    for r in read(root/'reports/recomb_confirm_v2_cohort_prescreen/application_prescreen.json'):
        if r['cohort']!='binding_interface' or r.get('existing_interaction_category')=='antibody_antigen':continue
        if r.get('design_source_status')!='METADATA_SCOPE_PASS' or r.get('partner_source_status')!='METADATA_SCOPE_PASS':continue
        item=inventory[r['candidate_id']];path=Path(item['source_file'])
        if sha(path)!=item['source_file_sha256']:raise ValueError('Cached PPI input source changed')
        g=read(path);m=g['reference']
        add(dict(candidate_id='PPIchain|'+r['candidate_id'],role='general_chain',sequence=g['wt_sequence'],
            WT_sha256=g['wt_sequence_sha256'],source_scope_review_status='BOTH_PPI_PARTNERS_METADATA_SCOPE_PASS',
            source_scope_review_file='reports/recomb_confirm_v2_cohort_prescreen/application_prescreen.json',
            description=r.get('design_description'),interaction_category=r.get('existing_interaction_category'),
            native_reference_path=m.get('reference_coordinate_path'),native_reference_pdb=m.get('pdb_id'),
            native_reference_chain=m.get('compressed_auth_chain_id'),native_source='cached_PPI_coordinate_audit',
            full_SEQRES_continuity_not_yet_revalidated=True))
    write(out/'WT_candidates.json',candidates);write(out/'rejected_before_model_use.json',rejected)
    with (out/'WT_candidates.fasta').open('w',encoding='ascii',newline='\n') as handle:
        for i,r in enumerate(candidates):
            r['audit_query_id']='v2WT_%04d'%i
            handle.write('>'+r['audit_query_id']+'\n'+r['sequence']+'\n')
    write(out/'WT_candidates.json',candidates)
    report=dict(status='ORDINARY_WT_INPUT_AUDIT_CANDIDATES_MATERIALIZED_NOT_MODEL_RELEASE',
        candidates=len(candidates),roles=dict(Counter(r['role'] for r in candidates)),rejected=len(rejected),
        deletion_generation_allowed=False,PLM_encoding_allowed=False,structure_prediction_allowed=False,
        freeze_sha256=sha(root/FZ),new_model_calls=0,job_id=os.environ['SLURM_JOB_ID'],
        files={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.suffix in ('.json','.fasta')})
    write(out/'completion.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();(prepare if a.prepare else execute)(a.root)
