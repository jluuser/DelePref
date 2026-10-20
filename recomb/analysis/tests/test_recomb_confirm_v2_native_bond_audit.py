"""Technical API regression checks, without neural model calls."""
import unittest
from types import SimpleNamespace
from recomb_confirm_v2_native_bond_audit import verify_native_covalent_bonds


class BondAuditTest(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            chains=[dict(name='A', res_idx=0), dict(name='L0001', res_idx=20)],
            atoms=[dict(name='ND2'), dict(name='C1'), dict(name='O1')],
            bonds=[dict(chain_1=1, chain_2=1, res_1=20, res_2=20,
                        atom_1=1, atom_2=2, type=1),
                   dict(chain_1=0, chain_2=1, res_1=4, res_2=20,
                        atom_1=0, atom_2=1, type=9)])
        self.expected = [dict(bond=dict(atom1=['A', 5, 'ND2'], atom2=['L0001', 1, 'C1']))]

    def test_internal_bonds_are_not_explicit_connections(self):
        self.assertEqual(verify_native_covalent_bonds(self.data, self.expected, 9), 1)

    def test_wrong_residue_is_rejected(self):
        self.data.bonds[1]['res_1'] = 5
        with self.assertRaises(ValueError):
            verify_native_covalent_bonds(self.data, self.expected, 9)

    def test_missing_or_extra_connection_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_native_covalent_bonds(self.data, [], 9)
        self.data.bonds.pop()
        with self.assertRaises(ValueError):
            verify_native_covalent_bonds(self.data, self.expected, 9)


if __name__ == '__main__':
    unittest.main()
