"""Verify explicit covalent endpoints against Boltz2's unified bond table."""
from collections import Counter


def verify_native_covalent_bonds(structure, constraints, covalent_type):
    def expected_key(bond):
        return tuple(sorted(tuple(bond[k]) for k in ('atom1', 'atom2')))

    expected = Counter(expected_key(x['bond']) for x in constraints)
    actual = Counter()
    for bond in structure.bonds:
        if int(bond['res_1']) == int(bond['res_2']):
            continue  # Standard CCD intramolecular bonds are also in this table.
        endpoints = []
        for index in (1, 2):
            chain = structure.chains[int(bond['chain_' + str(index)])]
            residue = int(bond['res_' + str(index)]) - int(chain['res_idx']) + 1
            atom = structure.atoms[int(bond['atom_' + str(index)])]
            endpoints.append((str(chain['name']), residue, str(atom['name'])))
        if int(bond['type']) != int(covalent_type):
            raise ValueError('Unexpected inter-residue bond type in native schema')
        actual[tuple(sorted(endpoints))] += 1
    if actual != expected:
        raise ValueError('Native covalent endpoints differ: missing=%s extra=%s' %
                         (dict(expected - actual), dict(actual - expected)))
    return sum(actual.values())
