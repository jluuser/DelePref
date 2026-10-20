"""Export only reviewed Binder/legacy sequences for numbering and exposure audits."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_binder_components')


def main(root):
    out=root/BASE
    if (out/'input_completion.json').exists():raise FileExistsError('Binder audit inputs already exported')
    source=root/'reports/recomb_confirm_v2_binder_native_v2/qualified_native_cases.json'
    done=read(source.parent/'completion.json')
    if sha(source)!=done['output_sha256']:raise ValueError('Native candidate release changed')
    scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    # Use stored identity fields only for historical exclusions; never export excluded sequences.
    old_scope=read(root/'reports/recomb_benign_scope_20261017/general_audit.json')+read(root/'reports/recomb_benign_application_scope_20261017/audit.json')
    excluded_ids={r['protein_id'] for r in old_scope if r['status']=='EXCLUDE'}
    blocked={r['WT_sha256'] for r in old_scope if r['status']=='EXCLUDE'}
    old_mixed=root/'reports/recomb_general100_applications_20261017/panel.json'
    for r in read(old_mixed):
        if r['protein_id'] not in excluded_ids:continue
        blocked.add(r['sequence_sha256'])
        for p in r.get('partners',[]):
            if p.get('sequence_sha256'):blocked.add(p['sequence_sha256'])
    queries={};cases=[];held=[]
    def add(seq,use):
        h=hashlib.sha256(seq.encode()).hexdigest()
        if h in blocked:raise ValueError('Previously excluded participant must not enter audit sequence queries')
        if set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Nonstandard audit sequence')
        if h not in queries:queries[h]=dict(sequence=seq,sequence_sha256=h,uses=[])
        queries[h]['uses'].append(use);return h
    for row in read(source):
        if scope[row['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':raise ValueError('Unreviewed design target')
        if sha(row['native_reference'])!=row['native_reference_sha256']:raise ValueError('Native reference changed')
        ref=read(row['native_reference'])
        if any(c['sequence_sha256'] in blocked or c['full_entity_sha256'] in blocked for c in ref['chains']):
            held.append(dict(native_case_key=row['native_case_key'],candidate_id=row['candidate_id'],reason='stored_prior_excluded_participant_hash'));continue
        copy=dict(row);participants=[]
        for c in ref['chains']:
            h=add(c['sequence'],dict(kind='candidate',case=row['native_case_key'],role=c['role'],PDB=row['PDB'],chain=c['chain_id']))
            participants.append(dict(role=c['role'],chain=c['chain_id'],sequence_sha256=h,full_entity_sha256=c['full_entity_sha256']))
        copy['participants']=participants;cases.append(copy)
    # All legacy query material is from the positively reviewed corrected constructs.
    old_general=read(root/'reports/recomb_benign_scope_20261017/general_panel.json')
    old_app=read(root/'reports/recomb_benign_application_constructs_20261017/panel.json')
    for row in old_general+old_app:
        add(row['sequence'],dict(kind='development',protein_id=row['protein_id'],role='design',dataset=row.get('dataset','General44')))
        for p in row.get('partners',[]):
            add(p['sequence'],dict(kind='development',protein_id=row['protein_id'],role='partner',chain=p['chain_id'],dataset=row.get('dataset')))
    for row in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json'):
        add(row['sequence'],dict(kind='General100',protein_id=row['protein_id'],role='design'))
    ordered=sorted(queries)
    for i,h in enumerate(ordered):queries[h]['query_id']='binderAudit_%04d'%i
    records=[queries[h] for h in ordered]
    for c in cases:
        for p in c['participants']:p['query_id']=queries[p['sequence_sha256']]['query_id']
    write(out/'case_inputs.json',cases);write(out/'query_inputs.json',records);write(out/'prior_scope_hash_holds.json',held)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'queries.fasta').open('w',encoding='ascii',newline='\n') as f:
        for q in records:f.write('>'+q['query_id']+'\n'+q['sequence']+'\n')
    numbering=[q for q in records if any(u['kind']=='development' or u['role']=='design_antibody_H' for u in q['uses'])]
    write(out/'numbering_inputs.json',numbering)
    report=dict(status='REVIEWED_BINDER_INPUT_AUDIT_SEQUENCES_ONLY_NOT_MODEL_RELEASE',cases=len(cases),
        unique_queries=len(records),numbering_queries=len(numbering),prior_scope_holds=len(held),
        blocked_hash_source='stored SHA fields only, no excluded residue columns used',
        new_model_calls=0,deletion_generation_allowed=False,PLM_encoding_allowed=False,
        permitted='existing HMM numbering and sequence identity/homology audit',
        files={n:sha(out/n) for n in ['case_inputs.json','query_inputs.json','queries.fasta','numbering_inputs.json','prior_scope_hash_holds.json']},
        source_sha256=sha(source),export_source_sha256=sha(Path(__file__)))
    write(out/'input_completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
