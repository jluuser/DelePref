"""Reuse passed native audits and repair only nine derived coordinate sources."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import time
from recomb_confirm_v2_diagnostics import read,sha,write
import recomb_confirm_v2_binder_native as original

BASE=Path('reports/recomb_confirm_v2_binder_native_v2')
FZ=Path('configs/recomb_confirm_v2_binder_native_freeze_v2.json')
STAGE=original.STAGE


def reuse_report(root,case,path,coordinate,entities):
    r=read(path)
    if r['native_case_key']!=case['native_case_key'] or r['case_id']!=case['candidate_id']:raise ValueError('Partial audit identity differs')
    if r['native_file_sha256']!=sha(coordinate):raise ValueError('Partial audit native source differs')
    design=r['chains'][0];partners=r['chains'][1:]
    if design['chain_id']!=case['prospective_design_chain'] or not design['role']=='design_antibody_H':raise ValueError('Partial design role differs')
    for c in r['chains']:
        if hashlib.sha256(c['sequence'].encode()).hexdigest()!=c['sequence_sha256']:raise ValueError('Partial chain sequence hash differs')
        if c['full_entity_sha256']!=entities[c['entity_id']]['metadata']['polymer_sequence_sha256']:raise ValueError('Reviewed polymer hash differs')
    if not 50<=len(design['sequence'])<=1022 or design['native_CA_coverage']<.85:raise ValueError('Partial heavy input criterion differs')
    if any(c['native_CA_coverage']<.8 for c in partners) or not r['observed_H_antigen_contacts']:raise ValueError('Partial partner/interface criterion differs')
    light=case.get('fixed_light_chain');light=None if light in (None,'','NA','None') else light
    import re
    scaffold='curated_antibody_heavy_chain_with_fixed_light' if light else 'curated_antibody_heavy_domain_without_light_in_construct'
    description=' '.join(str(x.get('description','')) for x in case['chain_evidence'] if x['auth_chain']==design['chain_id'])
    if not light and re.search(r'VHH|nanobody|single.domain antibody',description,re.I):scaffold='explicitly_annotated_VHH_or_nanobody'
    compatible=r['compatible_documented_assemblies']
    return dict(candidate_id=case['candidate_id'],native_case_key=case['native_case_key'],PDB=r['PDB'],model_use_allowed=False,
        status='NATIVE_HEAVY_PARTNER_IDENTITIES_AND_ANTIGEN_INTERFACE_PASS_CONTEXT_FAMILY_PENDING',
        design_chain=design['chain_id'],sequence=design['sequence'],WT_sha256=design['sequence_sha256'],WT_length=len(design['sequence']),
        design_CA_coverage=design['native_CA_coverage'],partner_count=len(partners),
        H_antigen_contacts=len(r['observed_H_antigen_contacts']),H_light_contacts=len(r['observed_H_light_contacts']),scaffold=scaffold,
        compatible_assembly_count=len(compatible),context_closure_known=any(not a['additional_polymer_labels'] for a in compatible),
        antigen_accessions=case.get('antigen_reference_accessions',[]),antigen_name_group=case['antigen_name_group'],
        native_reference=str(path),native_reference_sha256=sha(path),reused_passed_native_audit=True,
        antigen_sequence_hashes=[p['sequence_sha256'] for p in partners if p['role']=='fixed_antigen'])


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Native v2 release already prepared')
    m=root/STAGE/'coordinate_manifest_v2.json';manifest=read(m)
    if manifest['cases']!=351 or manifest['PDB']!=265:raise ValueError('Case scope changed during source repair')
    if sha(root/STAGE/'cases.json')!=manifest['original_case_manifest_sha256']:raise ValueError('Original cases differ')
    paths=['analysis/recomb_confirm_v2_binder_native_v2.py','analysis/recomb_confirm_v2_binder_native.py',
        'analysis/recomb_confirm_v2_enzyme_context.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_benign_binding_native_20261017.py','analysis/recomb_scope_screen_20261017.py',
        'slurm/recomb_confirm_v2_binder_native_v2.sbatch','analysis/tests/test_recomb_confirm_v2_binder_native.py',
        str(STAGE/'cases.json'),str(m.relative_to(root)),
        'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json',
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json']
    native={str(root/r['path']):r['sha256'] for r in manifest['records']}
    for p,h in native.items():
        if sha(p)!=h:raise ValueError('Corrected original native coordinate identity differs')
    partial={str(p):sha(p) for p in (root/'reports/recomb_confirm_v2_binder_native/references').glob('*.json')}
    write(root/FZ,dict(status='SAME_CASES_AND_NATIVE_CRITERIA_WITH_NINE_ORIGINAL_SOURCE_REPAIRS_FROZEN',
        files={p:sha(root/p) for p in paths},native_files=native,previous_passed_reference_files=partial,
        original_failed_job=1118792,source_repair='nine derived assemblies have entry XXXX; use original mmCIF by same PDB IDs',
        old_logs_freeze_sources_and_partial_references_preserved=True,new_targets=0,scientific_input_criteria_changed=False,
        model_use_allowed=False,new_structure_predictions=0))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(dict(freeze_sha256=sha(root/FZ),reusable_native_references=len(partial)),flush=True)


def work(item):
    root,pdb,cases,coordinate,entities,reusable=item
    out=[];remaining=[]
    for c in cases:
        path=Path(root)/'reports/recomb_confirm_v2_binder_native/references'/(c['native_case_key']+'.json')
        if str(path) in reusable:
            if sha(path)!=reusable[str(path)]:raise ValueError('Previously passed native reference changed')
            out.append(reuse_report(root,c,path,coordinate,entities))
        else:remaining.append(c)
    if remaining:
        original.BASE=BASE
        out.extend(original.audit_pdb((root,pdb,remaining,coordinate,entities)))
    return out


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Native repair source/input differs')
    for p,h in {**f['native_files'],**f['previous_passed_reference_files']}.items():
        if sha(p)!=h:raise ValueError('Native source or passed reference changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Native v2 already claimed; inspect job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    cases=read(root/STAGE/'cases.json');scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    groups=defaultdict(list)
    for c in cases:
        if scope[c['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':raise ValueError('Held case cannot be audited')
        groups[c['pdb'].upper()].append(c)
    coordinates={r['PDB']:str(root/r['path']) for r in read(root/STAGE/'coordinate_manifest_v2.json')['records']}
    entities=read(root/'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json')
    audit=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for batch in ex.map(work,[(str(root),pdb,cases,coordinates[pdb],entities.get(pdb,{}),f['previous_passed_reference_files']) for pdb,cases in sorted(groups.items())]):
            audit.extend(batch)
            if len(audit)%20<4:print('BINDER_NATIVE_V2_AUDITED '+str(len(audit))+'/351',flush=True)
    if len(audit)!=351:raise ValueError('Native audit scope incomplete')
    passed=[r for r in audit if r['status']=='NATIVE_HEAVY_PARTNER_IDENTITIES_AND_ANTIGEN_INTERFACE_PASS_CONTEXT_FAMILY_PENDING']
    write(out/'native_audit.json',audit);write(out/'qualified_native_cases.json',passed)
    report=dict(status='BINDER_NATIVE_IDENTITIES_INTERFACE_COMPLETE_CONTEXT_COMPONENTS_FINAL60_PENDING',
        cases=351,native_pass=len(passed),held=351-len(passed),
        reused_passed_native_references=sum(r.get('reused_passed_native_audit',False) for r in audit),
        hold_reasons=dict(Counter(r['reason'] for r in audit if 'reason' in r)),
        scaffold=dict(Counter(r['scaffold'] for r in passed)),
        with_documented_assembly=sum(r['compatible_assembly_count']>0 for r in passed),
        with_listed_context_closure=sum(r['context_closure_known'] for r in passed),
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,
        model_release_allowed=False,output_sha256=sha(out/'qualified_native_cases.json'))
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
