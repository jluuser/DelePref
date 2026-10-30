"""Audit full protein assemblies for qualified ordinary enzymes, before final selection."""
import argparse
from collections import Counter
import gzip
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_enzyme_context import rows, values
from recomb_confirm_v2_assembly import expand_instances
from recomb_confirm_v2_binder_native import restore_construct

BASE=Path('reports/recomb_confirm_v2_enzyme_assemblies')
FZ=Path('configs/recomb_confirm_v2_enzyme_assemblies_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Enzyme assembly audit already frozen')
    names=['analysis/recomb_confirm_v2_enzyme_assemblies.py','analysis/recomb_confirm_v2_assembly.py',
        'analysis/recomb_confirm_v2_binder_native.py','analysis/recomb_confirm_v2_enzyme_context.py',
        'analysis/recomb_confirm_v2_diagnostics.py','slurm/recomb_confirm_v2_enzyme_assemblies.sbatch',
        'reports/recomb_confirm_v2_enzyme_context/context_survey.json',
        'reports/recomb_confirm_v2_enzyme_native/candidate_inputs.json',
        'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json',
        'external/recomb_confirm_v2_EC7_native_staging/candidate_WT.json',
        'reports/recomb_confirm_v2_EC7_input_audit/completion.json',
        'reports/recomb_confirm_v2_EC7_input_audit/native_reference.json',
        'provenance/recomb_confirm_v2_EC7_metadata/P04191.json',
        'provenance/recomb_confirm_v2_public_metadata/rcsb_entity/1SU4_1.json']
    contexts=[c for c in read(root/names[6]) if not c['context_flags']]
    refs={c['native_reference']:c['native_reference_sha256'] for c in contexts}
    ec7ref=root/'reports/recomb_confirm_v2_EC7_input_audit/native_reference.json'
    refs[str(ec7ref)]=sha(ec7ref)
    native={read(p)['native_file']:read(p)['native_file_sha256'] for p in refs}
    write(root/FZ,dict(status='ORDINARY_ENZYME_FULL_PROTEIN_ASSEMBLY_AUDIT_FROZEN',
        files={p:sha(root/p) for p in names},native_references=refs,native_files=native,
        target_homomers='every equivalent gene copy uses the same canonical WT and subsequently the same edited sequence',
        partner_construct='observed native bounds with all internal SEQRES residues retained; positive source required',
        primary_reference='original audited catalytic native chain, identity operation preferred',
        chemical_context_mapping_still_pending=True,model_release_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def ec7_context(root):
    q=read(root/'external/recomb_confirm_v2_EC7_native_staging/candidate_WT.json')
    done=read(root/'reports/recomb_confirm_v2_EC7_input_audit/completion.json')
    if not done['status'].startswith('ORDINARY_EC7_NATIVE_AND_INPUT_EXPOSURE_PASS'):
        raise ValueError('EC7 native and exposure criteria not passed')
    meta=read(root/'provenance/recomb_confirm_v2_EC7_metadata/P04191.json')
    if meta['scope_flags'] or meta['metadata']['canonical_sequence_sha256']!=q['WT_sha256']:
        raise ValueError('EC7 source/identity changed')
    row=dict(q,ec_numbers=['7.2.2.10'],native_reference_pdb='1SU4',native_reference_chain='A',
        native_reference_path=str(root/q['native_path']),evaluation_sequence_component=q['candidate_id'],
        source_family='EC7_P04191',training_domain_like_exposure=True,
        annotation_tier='reviewed UniProtKB Active site, separate from MCSA complete catalytic networks')
    context=dict(candidate_id=q['candidate_id'],PDB='1SU4',assembly_id='1',native_design_label_chains=['A'],
        native_target_entity_ids=['1'],native_reference=str(root/'reports/recomb_confirm_v2_EC7_input_audit/native_reference.json'),
        native_reference_sha256=done['native_reference_sha256'],required_polymer_entities=[dict(entity_id='1',is_target_gene_entity=True)],
        canonical_cofactor_annotations=[c for c in meta['metadata']['functional_comments'] if c['commentType']=='COFACTOR'],
        canonical_subunit_annotations=[c for c in meta['metadata']['functional_comments'] if c['commentType']=='SUBUNIT'])
    metadata={'1':read(root/'provenance/recomb_confirm_v2_public_metadata/rcsb_entity/1SU4_1.json')}
    return row,context,metadata


def build(row,context,reference,entities):
    path=Path(reference['native_file'])
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as h:d=MMCIF2Dict(h)
    if values(d,'_entry.id')[0].upper()!=context['PDB']:raise ValueError('Native entry identity differs')
    polymers={p['entity_id']:p for p in rows(d,'_entity_poly.')}
    asym={p['id']:p['entity_id'] for p in rows(d,'_struct_asym.')}
    generation=[g for g in rows(d,'_pdbx_struct_assembly_gen.') if g['assembly_id']==context['assembly_id']]
    members={s.strip() for g in generation for s in g['asym_id_list'].split(',')}
    required={label:asym[label] for label in members if asym[label] in polymers}
    if not generation or not set(context['native_design_label_chains'])<=set(required):raise ValueError('Declared target assembly not closed')
    atoms={k.split('.',1)[1]:values(d,k) for k in d if k.startswith('_atom_site.')}
    n=len(atoms['id']);indices={label:[] for label in required}
    for i in range(n):
        label=atoms['label_asym_id'][i]
        if label not in indices or not atoms['label_seq_id'][i].isdigit():continue
        if atoms.get('pdbx_PDB_model_num',['1']*n)[i]!='1' or atoms['type_symbol'][i].upper() in ('H','D'):continue
        if float(atoms.get('occupancy',['1']*n)[i])>0:indices[label].append(i)
    parts=[]
    for label,eid in sorted(required.items()):
        poly=polymers[eid];meta=entities.get(eid,{})
        if poly.get('type')!='polypeptide(L)':raise ValueError('Required nonprotein polymer has no current input mapping')
        full=''.join(poly['pdbx_seq_one_letter_code_can'].split())
        expected=meta.get('metadata',{}).get('polymer_sequence_sha256')
        if not expected or hashlib.sha256(full.encode()).hexdigest()!=expected:raise ValueError('Native/reviewed full entity identity differs')
        target=eid in context['native_target_entity_ids']
        ids=meta.get('metadata',{}).get('polymer_identifiers',{}).get('uniprot_ids',[])
        if not target and row['uniprot_accession'] in ids:raise ValueError('Same gene split across polymer entities; canonical processed-fragment input requires explicit mapping')
        declared=next((p for p in context['required_polymer_entities'] if p['entity_id']==eid),None)
        if declared is None:raise ValueError('Assembly requires an unreviewed entity')
        if not target and not declared.get('source_status','').startswith('PASS_'):raise ValueError('Fixed partner source not positively reviewed')
        chosen={};observed={};auth=set()
        for i in indices[label]:
            pos=int(atoms['label_seq_id'][i]);aa=seq1(atoms['label_comp_id'][i],custom_map={'MSE':'M','PCA':'Q'},undef_code='X')
            auth.add(atoms['auth_asym_id'][i])
            if atoms['label_atom_id'][i]=='CA':
                if pos in observed and observed[pos]!=aa:raise ValueError('Native CA residue microheterogeneity')
                observed[pos]=aa
            alt=atoms.get('label_alt_id',['.']*n)[i];occ=float(atoms.get('occupancy',['1']*n)[i])
            key=pos,atoms['label_atom_id'][i];rank=(-occ,alt not in ('.','?'),alt)
            if key not in chosen or rank<chosen[key][0]:chosen[key]=rank,i,aa
        if len(auth)!=1:raise ValueError('Label chain to auth chain correspondence not unique')
        if target:
            seq=row['sequence'];mapping=reference['canonical_to_entity'];inverse={j+1:i for i,j in enumerate(mapping) if j is not None and 0<=j<len(full) and seq[i]==full[j]}
        else:
            seq,first,last=restore_construct(full,observed,expected);inverse={j:j-first for j in range(first,last+1)}
        if set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Unspecified/nonstandard prediction input residue')
        selected=[];ca=[None]*len(seq)
        for (pos,name),(_,i,aa) in sorted(chosen.items()):
            if pos not in inverse:continue
            j=inverse[pos]
            if aa!=seq[j]:raise ValueError('Native/input amino-acid identity differs')
            xyz=[float(atoms[k][i]) for k in ('Cartn_x','Cartn_y','Cartn_z')]
            if not np.isfinite(xyz).all():raise ValueError('Nonfinite native atom')
            selected.append(dict(residue_index=j,name=name,element=atoms['type_symbol'][i].upper(),xyz=xyz))
            if name=='CA':ca[j]=xyz
        if not target and sum(x is not None for x in ca)/len(seq)<.8:raise ValueError('Fixed partner native CA coverage below input criterion')
        if target and next(iter(auth))==reference['native_chain']:
            ca=reference['native_CA']  # Retain the already audited native CA choice, including unknowns.
        parts.append(dict(chain_id=next(iter(auth)),native_label_chain=label,entity_id=eid,
            role='design_enzyme' if target else 'fixed_protein_partner',sequence=seq,
            sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),full_entity_sha256=expected,
            native_CA=ca,heavy_atoms=selected))
    expanded,_=expand_instances(path,context['assembly_id'],parts)
    primary=[p for p in expanded if p['target_gene_copy'] and p['source_native_chain']==reference['native_chain']]
    if not primary:raise ValueError('Original catalytic reference chain absent from expanded assembly')
    primary=min(primary,key=lambda p:(not p['operation_is_identity'],p['predictor_chain_id']))
    if not primary['operation_is_identity']:raise ValueError('Original native catalytic reference requires additional explicit transform mapping')
    if primary['sequence_sha256']!=row['WT_sha256']:raise ValueError('Target gene WT changed')
    annotation=next(a for a in rows(d,'_pdbx_struct_assembly.') if a['id']==context['assembly_id'])
    declared=annotation.get('oligomeric_count','')
    if declared.isdigit() and int(declared)!=len(expanded):raise ValueError('Expanded protein count differs from declared assembly')
    return dict(candidate_id=row['candidate_id'],PDB=context['PDB'],assembly_id=context['assembly_id'],
        assembly_annotation=annotation,primary_target_chain=primary['predictor_chain_id'],chains=expanded,
        target_gene_copy_count=sum(p['target_gene_copy'] for p in expanded),catalytic=reference['catalytic'],
        canonical_cofactor_annotations=context['canonical_cofactor_annotations'],
        canonical_subunit_annotations=context['canonical_subunit_annotations'],
        native_nonpolymer_components=rows(d,'_pdbx_entity_nonpoly.'),
        canonical_native_reference=context['native_reference'],canonical_native_reference_sha256=context['native_reference_sha256'],
        native_file=str(path),native_file_sha256=reference['native_file_sha256'],
        native_coordinates_for_metrics_only_not_templates=True,chemical_context_profile_pending=True,
        model_release_allowed=False)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    frozen=read(root/FZ)
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen enzyme assembly input/source changed: '+p)
    for p,h in {**frozen['native_references'],**frozen['native_files']}.items():
        if sha(p)!=h:raise ValueError('Frozen native reference changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Assembly audit claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    rows_byid={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_enzyme_native/candidate_inputs.json')}
    evidence=read(root/'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json')
    work=[(rows_byid[c['candidate_id']],c,evidence[c['PDB']]) for c in read(root/'reports/recomb_confirm_v2_enzyme_context/context_survey.json') if not c['context_flags']]
    work.append(ec7_context(root));audit=[];qualified=[]
    for row,context,entities in work:
        d=dict(candidate_id=row['candidate_id'],PDB=context['PDB'])
        try:
            native=build(row,context,read(context['native_reference']),entities)
            dest=out/'references'/(row['WT_sha256']+'.json');write(dest,native)
            item=dict(row,native_assembly_reference=str(dest),native_assembly_reference_sha256=sha(dest),
                target_gene_copies=native['target_gene_copy_count'],protein_instances=len(native['chains']),
                assembly_id=native['assembly_id'],full_protein_context_AA=sum(len(p['sequence']) for p in native['chains']),
                annotation_tier=row.get('annotation_tier','MCSA complete catalytic network'),
                model_use_allowed=False,WT_encoding_allowed=False)
            qualified.append(item);d.update(status='DOCUMENTED_PROTEIN_ASSEMBLY_PASS_CHEMICAL_CONTEXT_AND_FINAL80_PENDING',
                protein_instances=item['protein_instances'],target_gene_copies=item['target_gene_copies'],reference=str(dest))
        except (ValueError,RuntimeError,KeyError,IndexError,TypeError) as e:
            d.update(status='ASSEMBLY_INPUT_HOLD_NO_MODELS',reason=str(e))
        audit.append(d);print('ENZYME_ASSEMBLY_AUDITED '+str(len(audit))+'/'+str(len(work)),flush=True)
    write(out/'assembly_audit.json',audit);write(out/'qualified_protein_constructs.json',qualified)
    done=dict(status='ENZYME_DOCUMENTED_PROTEIN_ASSEMBLY_AUDIT_COMPLETE_FINAL80_AND_CHEMICAL_PROFILE_PENDING',
        candidates=len(audit),protein_constructs_pass=len(qualified),
        EC_first_digit=dict(Counter((r['ec_numbers'] or ['?'])[0].split('.')[0] for r in qualified)),
        hold_reasons=dict(Counter(r['reason'] for r in audit if 'reason' in r)),
        sequence_components=len({r['evaluation_sequence_component'] for r in qualified}),
        outputs={n:sha(out/n) for n in ['assembly_audit.json','qualified_protein_constructs.json']},
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,model_release_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
