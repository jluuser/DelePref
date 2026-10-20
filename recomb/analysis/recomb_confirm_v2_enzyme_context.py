"""Survey documented native assembly/cofactor context without releasing models."""
import argparse
from collections import Counter
import gzip
import os
from pathlib import Path
import time
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_enzyme_context')
FZ=Path('configs/recomb_confirm_v2_enzyme_context_freeze.json')


def values(d,key):
    v=d.get(key,[])
    return [v] if isinstance(v,str) else v


def table(d,prefix):
    return {k[len(prefix):]:values(d,k) for k in d if k.startswith(prefix)}


def rows(d,prefix):
    columns=table(d,prefix)
    if not columns:return []
    n=max(map(len,columns.values()))
    if any(len(v)!=n for v in columns.values()):raise ValueError('Inconsistent mmCIF loop columns: '+prefix)
    return [{k:v[i] for k,v in columns.items()} for i in range(n)]


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Enzyme context survey already frozen')
    paths=['analysis/recomb_confirm_v2_enzyme_context.py','analysis/recomb_confirm_v2_diagnostics.py',
        'slurm/recomb_confirm_v2_enzyme_context.sbatch',
        'reports/recomb_confirm_v2_enzyme_native/candidate_inputs.json',
        'reports/recomb_confirm_v2_enzyme_native/qualified_native_inputs.json',
        'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json',
        'reports/recomb_confirm_v2_context_evidence/canonical_enzyme_context_notes.json',
        'reports/recomb_confirm_v2_context_evidence/completion.json']
    native=read(root/'configs/recomb_confirm_v2_enzyme_native_freeze.json')['native_files']
    write(root/FZ,dict(status='DOCUMENTED_ENZYME_ASSEMBLY_AND_COFACTOR_SURVEY_FROZEN',
        files={p:sha(root/p) for p in paths},native_files=native,new_model_calls=0,
        permitted='Native input/context qualification only; no new enzyme model release',
        future_homomer_rule='all copies of the target gene use the same edited sequence; distinct partners fixed for all methods',
        no_native_coordinates_as_prediction_templates=True))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen context evidence/source changed')
    approved=read(root/'reports/recomb_confirm_v2_enzyme_native/qualified_native_inputs.json')
    candidates={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_enzyme_native/candidate_inputs.json')}
    evidence=read(root/'reports/recomb_confirm_v2_context_evidence/entity_context_metadata.json')
    notes=read(root/'reports/recomb_confirm_v2_context_evidence/canonical_enzyme_context_notes.json')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Context survey claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    results=[];start=time.monotonic()
    for record in approved:
        row=candidates[record['candidate_id']];nativepath=Path(row['native_reference_path'])
        if sha(nativepath)!=f['native_files'][str(nativepath)]:raise ValueError('Native coordinate file changed')
        reference=read(record['native_reference'])
        if sha(record['native_reference'])!=record['native_reference_sha256']:raise ValueError('Native catalytic audit changed')
        with (gzip.open(nativepath,'rt') if nativepath.suffix=='.gz' else nativepath.open()) as handle:d=MMCIF2Dict(handle)
        assembly=str(reference.get('reference_assembly') or '')
        assemblies={r['id']:r for r in rows(d,'_pdbx_struct_assembly.')}
        generation=[r for r in rows(d,'_pdbx_struct_assembly_gen.') if r['assembly_id']==assembly]
        asym={r['id']:r['entity_id'] for r in rows(d,'_struct_asym.')}
        polymers={r['entity_id']:r for r in rows(d,'_entity_poly.')}
        labels=set()
        for auth,label,atom,model,position in zip(values(d,'_atom_site.auth_asym_id'),values(d,'_atom_site.label_asym_id'),
                values(d,'_atom_site.label_atom_id'),values(d,'_atom_site.pdbx_PDB_model_num'),values(d,'_atom_site.label_seq_id')):
            if auth==row['native_reference_chain'] and atom=='CA' and model=='1' and position.isdigit():labels.add(label)
        target_entities={asym[label] for label in labels}
        membership={a.strip() for r in generation for a in r['asym_id_list'].split(',')}
        required_entities={asym[a] for a in membership if a in asym and asym[a] in polymers}
        ep=evidence.get(row['native_reference_pdb'].upper(),{})
        context=[]
        for eid in sorted(required_entities):
            meta=ep.get(eid,{})
            own=eid in target_entities
            status='PASS_EXISTING_APPROVED_TARGET_ENTITY' if own else meta.get('scope_status','HOLD_PARTNER_METADATA_UNAVAILABLE')
            context.append(dict(entity_id=eid,polymer_type=polymers[eid].get('type'),is_target_gene_entity=own,
                asym_labels=sorted(a for a in membership if asym.get(a)==eid),source_status=status,
                source_metadata_sha256=meta.get('evidence_sha256'),
                description=meta.get('metadata',{}).get('entity',{}).get('pdbx_description'),
                canonical_length=len(''.join(polymers[eid].get('pdbx_seq_one_letter_code_can','').split()))))
        cofactor=[c for c in notes[row['candidate_id']].get('functional_comments',[]) if c.get('commentType')=='COFACTOR']
        subunit=[c for c in notes[row['candidate_id']].get('functional_comments',[]) if c.get('commentType')=='SUBUNIT']
        nonpoly=rows(d,'_pdbx_entity_nonpoly.')
        chemical={r['id']:dict(name=r.get('name'),type=r.get('type')) for r in rows(d,'_chem_comp.')}
        components=[dict(r,chemical=chemical.get(r.get('comp_id'),{})) for r in nonpoly]
        reasons=[]
        if len(labels)!=1 or len(target_entities)!=1:reasons.append('design_chain_label_or_entity_not_unique')
        if assembly not in assemblies or not generation or not labels.issubset(membership):reasons.append('declared_native_assembly_membership_not_established')
        if any(not r['source_status'].startswith('PASS_') for r in context):reasons.append('necessary_partner_source_review_pending')
        if any(r['polymer_type']!='polypeptide(L)' for r in context):reasons.append('nonprotein_polymer_context_requires_explicit_input_mapping')
        results.append(dict(candidate_id=row['candidate_id'],PDB=row['native_reference_pdb'].upper(),
            WT_sha256=row['WT_sha256'],uniprot_accession=row['uniprot_accession'],
            evaluation_sequence_component=row['evaluation_sequence_component'],source_family=row.get('source_family'),
            ec_numbers=row.get('ec_numbers'),WT_length=len(row['sequence']),native_CA_coverage=reference['native_CA_coverage'],
            assembly_id=assembly,assembly_annotation=assemblies.get(assembly),assembly_generation=generation,
            native_design_label_chains=sorted(labels),native_target_entity_ids=sorted(target_entities),
            required_polymer_entities=context,canonical_cofactor_annotations=cofactor,
            canonical_subunit_annotations=subunit,native_nonpolymer_components=components,
            reaction_compounds=reference.get('reaction_compounds'),
            context_flags=reasons,model_use_allowed=False,cofactor_CCD_and_assembly_instance_mapping_pending=True,
            native_reference=record['native_reference'],native_reference_sha256=record['native_reference_sha256']))
    write(out/'context_survey.json',results)
    done=dict(status='NATIVE_ENZYME_CONTEXT_SURVEY_COMPLETE_CONSTRUCT_PROTOCOL_AND_FINAL80_PENDING',
        inputs=len(results),assembly_membership_and_partner_source_pass=sum(not r['context_flags'] for r in results),
        context_flags=dict(Counter(x for r in results for x in r['context_flags'])),
        with_canonical_cofactor_annotation=sum(bool(r['canonical_cofactor_annotations']) for r in results),
        with_subunit_annotation=sum(bool(r['canonical_subunit_annotations']) for r in results),
        oligomeric_counts=dict(Counter(str((r['assembly_annotation'] or {}).get('oligomeric_count','unknown')) for r in results)),
        sequence_components=len({r['evaluation_sequence_component'] for r in results}),
        EC_first_digit=dict(Counter(sorted({e.split('.')[0] for e in r.get('ec_numbers') or []})[0] if r.get('ec_numbers') else 'unknown' for r in results)),
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,
        model_release_allowed=False,output_sha256=sha(out/'context_survey.json'))
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
