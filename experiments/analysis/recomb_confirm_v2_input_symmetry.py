"""Keep only native operator maps that preserve the actual protein/MSA/chemical graph."""
from collections import Counter


def filter_input_role_maps(allowed_maps,request,model_input):
    proteins={c['id']:c for c in request['chains']}
    ligands={s['ligand']['id']:s['ligand'] for s in model_input['sequences'] if 'ligand' in s}
    identity={k:k for k in proteins};bonds=[r['bond'] for r in model_input.get('constraints',[])]
    cross_ligand=any(b['atom1'][0] in ligands and b['atom2'][0] in ligands and b['atom1'][0]!=b['atom2'][0] for b in bonds)
    def component_signatures(mapping):
        signatures=[]
        for cid,ligand in ligands.items():
            internal=[];anchors=[]
            for bond in bonds:
                a,b=bond['atom1'],bond['atom2']
                if a[0]==cid and b[0]==cid:internal.append(tuple(sorted((tuple(a[1:]),tuple(b[1:])))))
                elif a[0]==cid and b[0] in proteins:anchors.append((a[1],a[2],mapping[b[0]],b[1],b[2]))
                elif b[0]==cid and a[0] in proteins:anchors.append((b[1],b[2],mapping[a[0]],a[1],a[2]))
            codes=ligand['ccd'];codes=tuple(codes) if isinstance(codes,list) else (codes,)
            signatures.append((codes,tuple(sorted(internal)),tuple(sorted(anchors))))
        protein_bonds=[]
        for bond in bonds:
            a,b=bond['atom1'],bond['atom2']
            if a[0] in proteins and b[0] in proteins:
                protein_bonds.append(tuple(sorted(((mapping[a[0]],*a[1:]),(mapping[b[0]],*b[1:])))))
        return Counter(signatures),Counter(protein_bonds)
    native=component_signatures(identity);kept=[];rejected=[]
    for mapping in allowed_maps:
        if set(mapping)!=set(proteins) or set(mapping.values())!=set(proteins):raise ValueError('Role map does not cover all physical protein copies')
        reason=None
        if any((proteins[c]['sequence'],proteins[c]['msa_csv_sha256'])!=(proteins[q]['sequence'],proteins[q]['msa_csv_sha256']) for c,q in mapping.items()):
            reason='actual_protein_sequence_or_MSA_not_equivalent'
        elif mapping!=identity and cross_ligand:reason='cross_ligand_graph_equivalence_not_established_identity_kept'
        elif component_signatures(mapping)!=native:reason='actual_covalent_chemical_attachment_graph_not_equivalent'
        if reason:rejected.append(dict(mapping=mapping,reason=reason))
        else:kept.append(mapping)
    if not kept or kept[0]!=identity:raise ValueError('The explicit identity map must remain available')
    return kept,dict(native_operator_maps=len(allowed_maps),input_equivalent_maps=len(kept),rejected=rejected,
        full_nonspatial_input_graph_checked=True,no_prediction_or_interface_score_used=True)
