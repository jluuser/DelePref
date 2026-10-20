"""Restore internal sequence continuity in reviewed Binder input constructs."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
from recomb_application_scope_20261017 import values
from recomb_scope_screen_20261017 import read,sha,write


def build(root):
    source=root/'reports/recomb_general100_applications_20261017/panel.json'
    review=root/'reports/recomb_benign_application_scope_20261017/audit.json'
    allowed={r['protein_id'] for r in read(review) if r['status']=='RETAIN'}
    out=root/'reports/recomb_benign_application_constructs_20261017';out.mkdir(exist_ok=False)
    panel,changes=[],[]
    for old in read(source):
        if old['protein_id'] not in allowed:continue
        row=copy.deepcopy(old)
        if row['dataset']=='Binder20':
            p=Path(row['native_reference_path'])
            if sha(p)!=row['native_reference_sha256']:raise ValueError('native changed')
            with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:d=MMCIF2Dict(f)
            canonical=dict(zip(values(d,'_entity_poly.entity_id'),values(d,'_entity_poly.pdbx_seq_one_letter_code_can')))
            parts=[dict(chain_id=row['design_chain'],sequence=row['sequence'],role='design')]
            parts.extend(dict(p,role='partner') for p in row['partners'])
            mappings=[]
            for part in parts:
                residues={};entities=set()
                for chain,atom,idx,comp,entity in zip(values(d,'_atom_site.auth_asym_id'),values(d,'_atom_site.label_atom_id'),
                    values(d,'_atom_site.label_seq_id'),values(d,'_atom_site.label_comp_id'),values(d,'_atom_site.label_entity_id')):
                    if chain==part['chain_id'] and atom=='CA' and idx.isdigit() and entity in canonical:
                        residues.setdefault(int(idx),seq1(comp,undef_code='X'));entities.add(entity)
                if len(entities)!=1 or not residues:raise ValueError('ambiguous native polymer')
                observed=''.join(residues[i] for i in sorted(residues))
                if observed!=part['sequence']:raise ValueError('old construct is not the documented observed CA sequence')
                full=''.join(canonical[next(iter(entities))].split())
                first,last=min(residues),max(residues)
                restored=full[first-1:last]
                if set(restored)-set('ACDEFGHIKLMNPQRSTVWYX'):raise ValueError('noncanonical polymer requires review')
                if any(full[i-1]!=a for i,a in residues.items()):raise ValueError('SEQRES/observed residue conflict')
                digest=hashlib.sha256(restored.encode()).hexdigest()
                changed=restored!=part['sequence']
                mappings.append(dict(chain_id=part['chain_id'],role=part['role'],native_label_seq_start=first,
                    native_label_seq_end=last,observed_positions=len(residues),restored_length=len(restored),
                    restored_internal_missing_count=len(restored)-len(residues),sequence_sha256=digest,changed=changed))
                if part['role']=='design':
                    row.update(sequence=restored,sequence_length=len(restored),sequence_sha256=digest,
                               target_length=len(restored)-round(len(restored)*row['budget']))
                else:
                    for partner in row['partners']:
                        if partner['chain_id']==part['chain_id']:partner.update(sequence=restored,sequence_sha256=digest)
            row['native_construct_mappings']=mappings
            row['construct_revision']='internal_SEQRES_continuity_with_original_observed_boundaries'
            row['legacy_construct_identity_unchanged']=not any(m['changed'] for m in mappings)
            changes.append(dict(protein_id=row['protein_id'],chains=mappings,
                                old_complex_cache_eligible=row['legacy_construct_identity_unchanged']))
        else:
            row['construct_revision']='canonical_enzyme_WT_unchanged_native_alignment_pending'
        panel.append(row)
    write(out/'panel.json',panel);write(out/'binder_construct_audit.json',changes)
    summary=dict(status='REVIEWED_APPLICATION_CONSTRUCTS_VERSIONED_NO_GENERATION',WT=len(panel),
        Binder_WT=len(changes),changed_Binder_cases=sum(not x['old_complex_cache_eligible'] for x in changes),
        changed_design_chains=sum(m['changed'] and m['role']=='design' for x in changes for m in x['chains']),
        changed_partner_chains=sum(m['changed'] and m['role']=='partner' for x in changes for m in x['chains']),
        enzyme_WT_unchanged=11,source_files={str(source.relative_to(root)):sha(source),str(review.relative_to(root)):sha(review)},
        panel_sha256=sha(out/'panel.json'),old_inputs_modified=False,model_results_used=False,
        new_model_calls=0,new_structure_predictions=0,application_generation_released=False,
        next='Use new construct hashes for new retrieval/encoding; reject old complex caches for corrected cases; complete enzyme native correspondence')
    write(out/'completion.json',summary);print(json.dumps(summary))


if __name__=='__main__':build(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
