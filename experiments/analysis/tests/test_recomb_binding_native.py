import unittest
from recomb_benign_binding_native_20261017 import contact_pairs


class NativeContactTests(unittest.TestCase):
    def test_residue_pair_not_atom_pair_count(self):
        a=[dict(residue_index=0,xyz=[0,0,0]),dict(residue_index=0,xyz=[.1,0,0])]
        b=[dict(residue_index=2,xyz=[1,0,0]),dict(residue_index=3,xyz=[5,0,0])]
        self.assertEqual(contact_pairs(a,b),[[0,2],[0,3]])

    def test_strict_cutoff_and_empty(self):
        self.assertEqual(contact_pairs([dict(residue_index=0,xyz=[0,0,0])],
            [dict(residue_index=0,xyz=[5,0,0])]),[])
        self.assertEqual(contact_pairs([],[]),[])


if __name__ == '__main__':
    unittest.main()
