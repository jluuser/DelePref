"""Verify SAbDab natural chain constructs and antigen interfaces, with no models."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import os
from pathlib import Path
import re
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_benign_binding_native_20261017 import contact_pairs

BASE=Path('reports/recomb_confirm_v2_binder_native')
FZ=Path('configs/recomb_confirm_v2_binder_native_freeze.json')
STAGE=Path('external/recomb_confirm_v2_SAbDab_native_staging')


def restore_construct(full,observed,expected_full_hash):
    if hashlib.sha256(full.encode()).hexdigest()!=expected_full_hash:raise ValueError('Native/metadata full polymer sequence differs')
    if not observed:raise ValueError('No positive-occupancy native CA')
    first,last=min(observed),max(observed)
    if not 1<=first<=last<=len(full):raise ValueError('Native label sequence outside SEQRES')
    for position,aa in observed.items():
        if full[position-1]!=aa:raise ValueError('Observed native residue and SEQRES identity differ')
    seq=full[first-1:last]
    if set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Nonstandard or unspecified construct residue')
    return seq,first,last


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Binder native audit already frozen')
    done=read(root/STAGE/'completion.json');manifest=read(root/STAGE/'request_manifest.json')
    if done['disposition']!={'SUCCEEDED':len(manifest['PDB'])}:raise ValueError('Native coordinates not fully staged')
    if sha(root/STAGE/'request_manifest.json')!=done['request_manifest_sha256']:raise ValueError('Native request identity differs')
    paths=['analysis/recomb_confirm_v2_binder_native.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_confirm_v2_enzyme_context.py','analysis/recomb_benign_binding_native_20261017.py',
        'analysis/recomb_scope_screen_20261017.py','analysis/tests/test_recomb_confirm_v2_binder_native.py',
        'slurm/recomb_confirm_v2_binder_native.sbatch',str(STAGE/'cases.json'),str(STAGE/'request_manifest.json'),str(STAGE/'completion.json'),
        'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json',
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json']
    native={}
    for r in done['records']:
        p=Path(r['path']) if r.get('server_cached') else root/r['path']
        if sha(p)!=r['sha256']:raise ValueError('Staged/cached native coordinate identity differs')
        native[str(p)]=r['sha256']
    write(root/FZ,dict(status='ORDINARY_BINDER_NATIVE_CONSTRUCT_AUDIT_FROZEN',files={p:sha(root/p) for p in paths},native_files=native,
        constructs='SEQRES between first and last positive-occupancy CA; all internal missing residues retained',
        design_length=[50,1022],design_min_CA_coverage=.85,partner_min_CA_coverage=.8,
        primary_interface='designed H chain to documented antigen chains, heavy atoms strictly<5A; H-light reported separately',
        at_least_one_observed_H_antigen_contact_required_for_direct_H_shrinking_panel=True,
        native_alternate_atoms='highest occupancy, then unlabelled, then lexical alt ID; model1 only',
        all_source_roles_and_sequence_hashes_required=True,new_model_calls=0,model_release_allowed=False,
        biological_assembly_closure_and_clonotype_family_release_still_pending=True))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def audit_pdb(item):
    root,pdb,cases,coordinate,entities=item
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.SeqUtils import seq1
    from recomb_confirm_v2_enzyme_context import values,rows
    path=Path(coordinate);result=[]
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as h:d=MMCIF2Dict(h)
    entry=values(d,'_entry.id')
    if len(entry)!=1 or entry[0].upper()!=pdb:raise ValueError('Native PDB entry differs')
    polymers={r['entity_id']:r for r in rows(d,'_entity_poly.')}
    asym={r['id']:r['entity_id'] for r in rows(d,'_struct_asym.')}
    assembly={r['id']:r for r in rows(d,'_pdbx_struct_assembly.')}
    generation=rows(d,'_pdbx_struct_assembly_gen.')
    atoms={k.split('.',1)[1]:values(d,k) for k in d if k.startswith('_atom_site.')}
    count=len(atoms['id']);bychain=defaultdict(list)
    for i in range(count):
        if atoms.get('pdbx_PDB_model_num',['1']*count)[i]!='1':continue
        if not atoms['label_seq_id'][i].isdigit() or atoms['type_symbol'][i].upper() in ('H','D'):continue
        occupancy=float(atoms.get('occupancy',['1']*count)[i])
        if occupancy<=0:continue
        bychain[atoms['auth_asym_id'][i]].append(i)
    parts_cache={}
    def part(chain,role):
        if chain not in parts_cache:
            indices=bychain.get(chain,[])
            labels={atoms['label_asym_id'][i] for i in indices};eids={atoms['label_entity_id'][i] for i in indices}
            if len(labels)!=1 or len(eids)!=1:raise ValueError('Auth chain does not identify one polymer entity')
            eid=next(iter(eids));label=next(iter(labels));poly=polymers.get(eid,{})
            if poly.get('type')!='polypeptide(L)':raise ValueError('Named binder role is not a protein chain')
            full=''.join(poly['pdbx_seq_one_letter_code_can'].split());meta=entities.get(eid,{})
            expected=meta.get('metadata',{}).get('polymer_sequence_sha256')
            if not expected:raise ValueError('No reviewed full polymer identity metadata')
            observed={};chosen={}
            for i in indices:
                position=int(atoms['label_seq_id'][i]);aa=seq1(atoms['label_comp_id'][i],custom_map={'MSE':'M','PCA':'Q'},undef_code='X')
                if atoms['label_atom_id'][i]=='CA':
                    if position in observed and observed[position]!=aa:raise ValueError('Native CA identity microheterogeneity')
                    observed[position]=aa
                alt=atoms.get('label_alt_id',['.']*count)[i];occ=float(atoms.get('occupancy',['1']*count)[i])
                key=position,atoms['label_atom_id'][i];rank=(-occ,alt not in ('.','?'),alt)
                if key not in chosen or rank<chosen[key][0]:chosen[key]=rank,i,aa
            seq,first,last=restore_construct(full,observed,expected)
            selected=[];ca=[None]*len(seq)
            for (position,name),(_,i,aa) in sorted(chosen.items()):
                if not first<=position<=last:continue
                j=position-first
                if aa!=seq[j]:raise ValueError('Native atom amino-acid identity differs')
                xyz=[float(atoms[k][i]) for k in ['Cartn_x','Cartn_y','Cartn_z']]
                if not np.isfinite(xyz).all():raise ValueError('Nonfinite native atom')
                selected.append(dict(residue_index=j,name=name,element=atoms['type_symbol'][i].upper(),xyz=xyz))
                if name=='CA':ca[j]=xyz
            parts_cache[chain]=dict(chain_id=chain,native_label_chain=label,entity_id=eid,sequence=seq,
                sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),full_entity_sha256=expected,
                label_seq_start=first,label_seq_end=last,native_CA=ca,heavy_atoms=selected,
                known_CA=sum(v is not None for v in ca),unknown_CA=sum(v is None for v in ca),
                native_CA_coverage=sum(v is not None for v in ca)/len(seq))
        return dict(parts_cache[chain],role=role)
    for case in cases:
        report=dict(candidate_id=case['candidate_id'],native_case_key=case['native_case_key'],PDB=pdb,model_use_allowed=False)
        try:
            h=case['prospective_design_chain'];light=case.get('fixed_light_chain')
            light=None if light in (None,'','NA','None') else light
            antigen=list(dict.fromkeys(c.strip() for c in re.split(r'\s*\|\s*',case['antigen_chain']) if c.strip()))
            required=[h]+([light] if light else [])+antigen
            if len(required)!=len(set(required)):raise ValueError('Design/light/antigen chain-role collision')
            evidence={r['auth_chain']:r['scope'] for r in case['chain_evidence']}
            if any(not evidence.get(c,'').startswith('PASS_') for c in required):raise ValueError('Role lacks positive source metadata')
            design=part(h,'design_antibody_H');partners=([part(light,'fixed_light_chain')] if light else [])+[part(c,'fixed_antigen') for c in antigen]
            if not 50<=len(design['sequence'])<=1022 or design['native_CA_coverage']<.85:raise ValueError('Heavy construct length/CA coverage below input criterion')
            if any(p['native_CA_coverage']<.8 for p in partners):raise ValueError('Fixed partner CA coverage below input criterion')
            contacts=[];light_contacts=[]
            for p in partners:
                pairs=[dict(design_index=a,partner_chain=p['chain_id'],partner_index=b) for a,b in contact_pairs(design['heavy_atoms'],p['heavy_atoms'])]
                (contacts if p['role']=='fixed_antigen' else light_contacts).extend(pairs)
            if not contacts:raise ValueError('No observed direct heavy-chain antigen contact')
            labels={c['native_label_chain'] for c in [design]+partners};compatible=[]
            for aid,annotation in assembly.items():
                generators=[r for r in generation if r['assembly_id']==aid]
                members={a.strip() for r in generators for a in r['asym_id_list'].split(',')}
                if labels.issubset(members):
                    extra=members-labels
                    extra_polymers=[dict(label_chain=a,entity_id=asym[a],polymer_type=polymers[asym[a]].get('type'),
                        source_status=entities.get(asym[a],{}).get('scope_status','HOLD_METADATA_UNAVAILABLE'))
                        for a in sorted(extra) if a in asym and asym[a] in polymers]
                    compatible.append(dict(assembly_id=aid,annotation=annotation,generators=generators,additional_polymer_labels=extra_polymers))
            scaffold='curated_antibody_heavy_chain_with_fixed_light' if light else 'curated_antibody_heavy_domain_without_light_in_construct'
            description=' '.join(str(x.get('description','')) for x in case['chain_evidence'] if x['auth_chain']==h)
            if not light and re.search(r'VHH|nanobody|single.domain antibody',description,re.I):scaffold='explicitly_annotated_VHH_or_nanobody'
            ref=dict(case_id=case['candidate_id'],native_case_key=case['native_case_key'],PDB=pdb,
                native_file=str(path),native_file_sha256=sha(path),chains=[design]+partners,
                observed_H_antigen_contacts=contacts,observed_H_light_contacts=light_contacts,
                compatible_documented_assemblies=compatible,model_use_allowed=False,
                internal_SEQRES_continuity_preserved=True,antigen_and_light_interfaces_separate=True)
            dest=Path(root)/BASE/'references'/(case['native_case_key']+'.json');write(dest,ref)
            report.update(status='NATIVE_HEAVY_PARTNER_IDENTITIES_AND_ANTIGEN_INTERFACE_PASS_CONTEXT_FAMILY_PENDING',
                design_chain=h,sequence=design['sequence'],WT_sha256=design['sequence_sha256'],WT_length=len(design['sequence']),
                design_CA_coverage=design['native_CA_coverage'],partner_count=len(partners),
                H_antigen_contacts=len(contacts),H_light_contacts=len(light_contacts),scaffold=scaffold,
                compatible_assembly_count=len(compatible),context_closure_known=any(not a['additional_polymer_labels'] for a in compatible),
                antigen_accessions=case.get('antigen_reference_accessions',[]),antigen_name_group=case['antigen_name_group'],
                native_reference=str(dest),native_reference_sha256=sha(dest),
                antigen_sequence_hashes=[p['sequence_sha256'] for p in partners if p['role']=='fixed_antigen'])
        except (ValueError,RuntimeError,KeyError,IndexError,TypeError) as error:
            report.update(status='NATIVE_BINDER_INPUT_HOLD_NO_MODEL_USE',reason=str(error))
        result.append(report)
    return result


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen Binder source/evidence changed')
    for p,h in f['native_files'].items():
        if sha(p)!=h:raise ValueError('Native coordinate file changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Native Binder audit already claimed')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    cases=read(root/STAGE/'cases.json');scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    grouped=defaultdict(list)
    for c in cases:
        if scope[c['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':raise ValueError('Held target cannot enter native audit')
        grouped[c['pdb'].upper()].append(c)
    evidence=read(root/'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json')
    coordinates={r['PDB']:str(Path(r['path']) if r.get('server_cached') else root/r['path']) for r in read(root/STAGE/'completion.json')['records']}
    start=time.monotonic();audit=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for batch in ex.map(audit_pdb,[(str(root),pdb,rows,coordinates[pdb],evidence.get(pdb,{})) for pdb,rows in sorted(grouped.items())]):
            audit.extend(batch)
            if len(audit)%20<4:print('BINDER_NATIVE_AUDITED '+str(len(audit))+'/'+str(len(cases)),flush=True)
    passed=[r for r in audit if r['status']=='NATIVE_HEAVY_PARTNER_IDENTITIES_AND_ANTIGEN_INTERFACE_PASS_CONTEXT_FAMILY_PENDING']
    write(out/'native_audit.json',audit);write(out/'qualified_native_cases.json',passed)
    done=dict(status='BINDER_NATIVE_ROLES_CONTINUITY_INTERFACE_AUDIT_COMPLETE_FINAL60_PENDING',cases=len(cases),
        native_pass=len(passed),disposition=dict(Counter(r['status'] for r in audit)),
        hold_reasons=dict(Counter(r['reason'] for r in audit if 'reason' in r)),
        scaffold=dict(Counter(r['scaffold'] for r in passed)),
        cases_with_documented_assembly=sum(r['compatible_assembly_count']>0 for r in passed),
        cases_with_listed_context_closure=sum(r['context_closure_known'] for r in passed),
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,
        model_release_allowed=False,outputs={n:sha(out/n) for n in ['native_audit.json','qualified_native_cases.json']})
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
