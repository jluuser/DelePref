import hashlib
import unittest
from recomb_confirm_v2_binder_native import restore_construct
from recomb_benign_binding_native_20261017 import contact_pairs


class NativeConstructTests(unittest.TestCase):
    def test_internal_missing_positions_remain_in_sequence(self):
        seq='ACDEFGHIKL';h=hashlib.sha256(seq.encode()).hexdigest()
        restored,start,end=restore_construct(seq,{2:'C',4:'E',9:'K'},h)
        self.assertEqual((restored,start,end),('CDEFGHIK',2,9))

    def test_version_or_chain_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError):restore_construct('ACDE',{1:'A',4:'E'},'0'*64)

    def test_observed_residue_conflict_rejected(self):
        h=hashlib.sha256(b'ACDE').hexdigest()
        with self.assertRaises(ValueError):restore_construct('ACDE',{1:'A',3:'N'},h)

    def test_H_light_contact_is_not_antigen_contact(self):
        atom=lambda x:dict(residue_index=0,xyz=[x,0,0])
        self.assertEqual(contact_pairs([atom(0)],[atom(1)]),[[0,0]])
        self.assertEqual(contact_pairs([atom(0)],[atom(10)]),[])


if __name__=='__main__':unittest.main()
