"""Audit canonical enzyme/native/catalytic identity before construct/model release."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import gemmi
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_edit_correspondence_20261017 import consensus

BASE=Path('reports/recomb_confirm_v2_enzyme_native')
FZ=Path('configs/recomb_confirm_v2_enzyme_native_freeze.json')
SOURCE=Path('/public/nfs-data/keyuan.zhou/evocompress/data/enzyme_120_v1/coordinate_audit_v1/coordinate_pass.jsonl')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Enzyme native audit already prepared')
    eligible=root/'reports/recomb_confirm_v2_native_audit_release/eligible_WT_native_audit_inputs.json'
    general=root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json'
    used={r['evaluation_sequence_component'] for r in read(general)}
    rows=[r for r in read(eligible) if r['role']=='enzyme' and r['evaluation_sequence_component'] not in used]
    scopes={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    if any(scopes[r['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING' for r in rows):
        raise ValueError('Functional HOLD cannot enter audit')
    write(root/BASE/'candidate_inputs.json',rows)
    paths=['analysis/recomb_confirm_v2_enzyme_native.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_edit_correspondence_20261017.py','slurm/recomb_confirm_v2_enzyme_native.sbatch',
        str(eligible.relative_to(root)),str(general.relative_to(root)),
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json',str(BASE/'candidate_inputs.json')]
    native={r['native_reference_path']:sha(r['native_reference_path']) for r in rows}
    write(root/FZ,dict(status='ORDINARY_ENZYME_NATIVE_AUDIT_ONLY_FROZEN',files={p:sha(root/p) for p in paths},
        external_files={str(SOURCE):sha(SOURCE)},native_files=native,
        mapping='exact unique entity segment; otherwise all-optimal unit-edit consensus, preserve matching AA only',
        min_native_CA_coverage=.85,all_annotated_catalytic_sites_identity_auth_position_and_CA_required=True,
        no_shape_or_model_outcomes_used=True,unknown_coordinates_preserved=True,
        construct_assembly_cofactor_review_still_required=True,model_use_allowed=False))
    print(dict(candidates=len(rows),freeze_sha256=sha(root/FZ)),flush=True)


def audit_one(item):
    row,raw,minimum=item
    result=dict(candidate_id=row['candidate_id'],WT_sha256=row['WT_sha256'],new_model_calls=0)
    try:
        seq=row['sequence']
        if raw['selection_outputs_used'] or raw['coordinate_audit']!='pass':raise ValueError('Wrong native provenance')
        if hashlib.sha256(raw['sequence'].encode()).hexdigest()!=row['WT_sha256']:raise ValueError('Canonical source identity differs')
        structure=gemmi.read_structure(row['native_reference_path']);chain=structure[0].find_chain(row['native_reference_chain'])
        if chain is None:raise ValueError('Reference chain absent')
        entity=structure.get_entity_of(chain.get_polymer())
        if entity is None or not entity.full_sequence:raise ValueError('Full SEQRES entity absent')
        full=''.join(gemmi.find_tabulated_residue(r).one_letter_code.upper() for r in entity.full_sequence)
        starts=[i for i in range(len(full)-len(seq)+1) if full[i:i+len(seq)]==seq]
        if len(starts)==1:
            mapping=list(range(starts[0],starts[0]+len(seq)));backend='unique_exact_entity_segment';ambiguous=0
        else:
            align=consensus(seq,full);mapping=align['wt_to_edit'];backend=align['method'];ambiguous=align['ambiguous_positions']
        inverse={j:i for i,j in enumerate(mapping) if j is not None and seq[i]==full[j]}
        xyz=[None]*len(seq);residues={};residue_map=[]
        for residue in chain:
            if residue.entity_type==gemmi.EntityType.NonPolymer or residue.label_seq is None:continue
            j=int(residue.label_seq)-1
            if j not in inverse:continue
            i=inverse[j];aa=gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
            if aa!=seq[i]:continue
            ca=[a for a in residue if a.name=='CA' and a.element.name=='C']
            if not ca:continue
            if xyz[i] is not None:raise ValueError('Multiple native residues map to same canonical position')
            atom=ca[0];xyz[i]=[atom.pos.x,atom.pos.y,atom.pos.z];residues[i]=residue
            residue_map.append(dict(wt_index=i,label_seq=residue.label_seq,auth_number=residue.seqid.num,
                insertion_code=residue.seqid.icode,residue_name=residue.name,altloc=atom.altloc))
        known=sum(p is not None for p in xyz);coverage=known/len(seq)
        sites=[]
        for s in raw['catalytic_residues']:
            i=int(s['position'])-1
            if not 0<=i<len(seq) or seq[i]!=s['sequence_code']:raise ValueError('Catalytic canonical index/code differs')
            residue=residues.get(i)
            if residue is None or int(s['auth_residue_number'])!=residue.seqid.num:
                raise ValueError('Catalytic native auth position or CA not established')
            aa=gemmi.find_tabulated_residue(str(s['code']).upper()).one_letter_code.upper()
            if aa!=seq[i]:raise ValueError('Catalytic amino-acid annotation identity differs')
            sites.append(dict(wt_index=i,WT_residue=seq[i],auth_number=residue.seqid.num,
                insertion_code=residue.seqid.icode,source_position_1based=s['position'],
                functional_atoms=s.get('functional_atoms',[]),source_status='manually_annotated_MCSA_coordinate_pass'))
        if not sites or len(sites)!=raw['catalytic_residue_count']:raise ValueError('Incomplete catalytic annotation roster')
        if coverage<minimum:raise ValueError('Native canonical CA coverage below input criterion')
        reference=dict(WT_sha256=row['WT_sha256'],WT_length=len(seq),native_CA=xyz,catalytic=sites,
            residue_mapping=residue_map,alignment_backend=backend,ambiguous_canonical_positions=ambiguous,
            canonical_to_entity=mapping,native_file=row['native_reference_path'],native_file_sha256=sha(row['native_reference_path']),
            native_chain=row['native_reference_chain'],known_CA_count=known,unknown_CA_count=len(seq)-known,
            native_CA_coverage=coverage,reference_assembly=raw.get('reference_assembly'),
            reaction_compounds=raw.get('reaction_compounds'),
            construct_assembly_cofactor_review_pending=True,model_use_allowed=False)
        result.update(status='NATIVE_AND_ALL_CATALYTIC_IDENTITIES_PASS_CONTEXT_PENDING',
            native_CA_coverage=coverage,alignment_backend=backend,annotated_sites=len(sites),reference=reference)
    except (ValueError,RuntimeError,KeyError,IndexError,TypeError) as e:
        result.update(status='NATIVE_INPUT_HOLD_NO_MODEL_USE',reason=str(e))
    return result


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Native enzyme input/source changed')
    for p,h in {**f['external_files'],**f['native_files']}.items():
        if sha(p)!=h:raise ValueError('Frozen native source changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Native enzyme audit claimed; inspect instead')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    rows=read(out/'candidate_inputs.json');allowed={r['source_candidate_id'] for r in rows};raw={}
    for line in SOURCE.read_text().splitlines():
        d=json.loads(line)
        if d['candidate_id'] in allowed:raw[d['candidate_id']]=d
    start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for d in ex.map(audit_one,[(r,raw[r['source_candidate_id']],f['min_native_CA_coverage']) for r in rows]):
            results.append(d)
            print('ENZYME_NATIVE_AUDITED '+str(len(results))+'/'+str(len(rows)),flush=True)
    references=[];audit=[]
    for d in results:
        item=dict(d);reference=item.pop('reference',None)
        if reference:
            path=out/'references'/(item['WT_sha256']+'.json');write(path,reference)
            item.update(native_reference=str(path),native_reference_sha256=sha(path));references.append(item)
        audit.append(item)
    write(out/'native_audit.json',audit);write(out/'qualified_native_inputs.json',references)
    done=dict(status='ENZYME_NATIVE_CATALYTIC_AUDIT_COMPLETE_CONTEXT_AND_FINAL80_RELEASE_PENDING',
        candidates=len(rows),native_pass=len(references),disposition=dict(Counter(d['status'] for d in results)),
        rejection_reasons=dict(Counter(d.get('reason') for d in results if d['status']=='NATIVE_INPUT_HOLD_NO_MODEL_USE')),
        seconds=time.monotonic()-start,new_model_calls=0,new_structure_calls=0,model_use_allowed=False,
        job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=sha(root/FZ),audit_sha256=sha(out/'native_audit.json'))
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
