"""Native atom references for reviewed contiguous binding-interface constructs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from recomb_scope_screen_20261017 import read, sha, write

BASE = Path('reports/recomb_benign_binding_native_20261017')


def contact_pairs(design_atoms, partner_atoms, cutoff=5.):
    """Unique residue pairs from observed heavy atoms; fixed input residue indices."""
    if not design_atoms or not partner_atoms:
        return []
    d = np.asarray([a['xyz'] for a in design_atoms],dtype=float)
    p = np.asarray([a['xyz'] for a in partner_atoms],dtype=float)
    result = set()
    for start in range(0,len(d),128):
        distances2 = np.sum((d[start:start+128,None,:]-p[None,:,:])**2,axis=2)
        a,b = np.nonzero(distances2 < cutoff**2)
        result.update((design_atoms[start+i]['residue_index'],partner_atoms[j]['residue_index']) for i,j in zip(a,b))
    return [list(pair) for pair in sorted(result)]


def build(root):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.SeqUtils import seq1
    from recomb_application_scope_20261017 import values
    panel_path = root/'reports/recomb_benign_application_constructs_20261017/panel.json'
    expected = 'e5688ff6918f1af5a96bb5c85eda17c3506f2d45e31ac832bf99fcafd04e8f57'
    if sha(panel_path) != expected:
        raise ValueError('reviewed contiguous application panel changed')
    approved = {p['protein_id'] for p in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if p['status']=='RETAIN'}
    out = root/BASE
    if out.exists():
        raise FileExistsError('native references already created')
    references = []; sources = {}
    for row in read(panel_path):
        if row['dataset'] != 'Binder20':
            continue
        if row['protein_id'] not in approved:
            raise ValueError('unreviewed binding construct')
        p = Path(row['native_reference_path'])
        if sha(p) != row['native_reference_sha256']:
            raise ValueError('native mmCIF changed')
        sources[str(p)] = sha(p)
        with gzip.open(p,'rt') if p.suffix=='.gz' else p.open() as f:
            d = MMCIF2Dict(f)
        atom = {k.split('.',1)[1]: v for k,v in d.items() if k.startswith('_atom_site.')}
        n = len(atom['id']); chains = []
        parts = [dict(chain_id=row['design_chain'],role='design',sequence=row['sequence'])]
        parts += [dict(c,role='partner') for c in row['partners']]
        mappings = {m['chain_id']:m for m in row['native_construct_mappings']}
        for part in parts:
            chain = part['chain_id']; sequence = part['sequence']; m = mappings[chain]
            if hashlib.sha256(sequence.encode()).hexdigest() != m['sequence_sha256'] or len(sequence) != m['restored_length']:
                raise ValueError('native/input chain identity mismatch')
            selected = {}; labels = set()
            for i in range(n):
                label = atom['label_seq_id'][i]
                if atom['auth_asym_id'][i] != chain or not label.isdigit() or atom.get('pdbx_PDB_model_num',['1']*n)[i] != '1':
                    continue
                index = int(label)-m['native_label_seq_start']
                if not 0 <= index < len(sequence):
                    continue
                element = atom['type_symbol'][i].upper()
                if element in ('H','D'):
                    continue
                aa = seq1(atom['label_comp_id'][i],custom_map={'MSE':'M'},undef_code='X')
                if aa != sequence[index]:
                    raise ValueError('observed native residue identity mismatch')
                labels.add(atom['label_asym_id'][i])
                occupancy = float(atom.get('occupancy',['1']*n)[i])
                if occupancy <= 0:
                    continue
                alt = atom.get('label_alt_id',['.']*n)[i]
                name = atom['label_atom_id'][i]
                rank = (-occupancy,alt not in ('.','?'),alt)
                key = index,name
                if key not in selected or rank < selected[key][0]:
                    xyz = [float(atom[k][i]) for k in ['Cartn_x','Cartn_y','Cartn_z']]
                    if not np.isfinite(xyz).all():
                        raise ValueError('nonfinite native coordinate')
                    selected[key] = rank,dict(residue_index=index,name=name,element=element,xyz=xyz)
            if len(labels) != 1:
                raise ValueError('ambiguous native polymer chain')
            atoms = [value[1] for key,value in sorted(selected.items())]
            ca = [None]*len(sequence)
            for a in atoms:
                if a['name'] == 'CA':
                    ca[a['residue_index']] = a['xyz']
            chains.append(dict(native_chain_id=chain,role=part['role'],sequence=sequence,
                sequence_sha256=m['sequence_sha256'],native_label_chain=next(iter(labels)),
                native_label_seq_start=m['native_label_seq_start'],native_CA=ca,heavy_atoms=atoms,
                known_CA=sum(x is not None for x in ca),unknown_CA=sum(x is None for x in ca)))
        design = chains[0]; contacts = []
        for partner in chains[1:]:
            contacts += [dict(design_index=a,partner_chain=partner['native_chain_id'],partner_index=b)
                for a,b in contact_pairs(design['heavy_atoms'],partner['heavy_atoms'])]
        references.append(dict(protein_id=row['protein_id'],WT_sha256=row['sequence_sha256'],chains=chains,
            observed_native_heavy_atom_contacts=contacts,contact_cutoff_A=5.,native_file_sha256=row['native_reference_sha256'],
            native_unresolved_regions_are_unknown=True,legacy_complex_construct_unchanged=row['legacy_construct_identity_unchanged']))
    if len(references) != 12:
        raise ValueError('binding panel size changed')
    out.mkdir(); write(out/'native_references.json',references)
    sources[str(Path(__file__).resolve())] = sha(Path(__file__))
    report = dict(status='REVIEWED_BINDING_NATIVE_ATOMS_AND_CHAIN_CORRESPONDENCE_VERIFIED',WT=12,
        chains=sum(len(r['chains']) for r in references),full_input_internal_continuity_preserved=True,
        all_observed_residue_identities_match=True,unknown_coordinates_preserved=True,
        native_contact_definition='observed design-partner heavy-atom minimum distance strictly below5A; unresolved native contacts unobserved, coverage separately reported',
        alternate_atom_rule='highest occupancy per named atom; ties prefer unlabelled then lexicographic alt identifier; model1 only',
        native_contact_counts={r['protein_id']:len(r['observed_native_heavy_atom_contacts']) for r in references},
        source_sha256=sources,panel_sha256=expected,output_sha256=sha(out/'native_references.json'),
        new_model_calls=0,new_structure_predictions=0)
    write(out/'completion.json',report); print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
